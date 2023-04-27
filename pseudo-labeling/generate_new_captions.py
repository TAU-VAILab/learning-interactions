
import pandas as pd
import os
from transformers import pipeline
import time
from names import names2random_names, masks2random_names
from transformers import logging as transformers_logging
import warnings
import sqlite3
from tqdm.auto import tqdm
import numpy as np
from nltk import word_tokenize
import logging

def generate_from_prompt(generator, prompt, min_length=1, max_length=200, absolute_max_len=2048):
    
    prompt_toks = generator.tokenizer.tokenize(prompt)
    
    input_length = len(prompt_toks)
    
    if input_length >= absolute_max_len:
        logging.warning(f'Prompt length exceeds {absolute_max_len} tokens; not generating')
        return None
    
    output = generator(
        prompt,
        do_sample=True,
        max_length=min(input_length + max_length, 2048),
        min_length=input_length + min_length,
        top_p=0.95,
        temperature=0.7,
        no_repeat_ngram_size=3
    )
    
    if len(output) > 0:
        output_text = output[0].get('generated_text', '')
        return output_text[len(prompt):].strip()

def contains_names(text, names):
    toks = set(word_tokenize(text))
    return all(N in toks for N in names)

def basic_filter(text, names):
    ok = True
    ok = ok and len(text) > 0 and len(names) > 1
#     ok = ok and all(name in text for name in names)
    ok = ok and contains_names(text, names)
    return ok

def main():

    transformers_logging.set_verbosity_error() # suppress padding warnings

    db_fn = 'output/generated_interactions.db'
    assert os.path.exists(db_fn), f'Missing DB: {db_fn}'
    data_fn = 'output/ww_parses.csv'
    assert os.path.exists(data_fn), f'Missing data file: {data_fn}'

    print("Loading model...")
    start = time.time()
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)
    print('Replacing EOS token ID. Original:', generator.model.config.eos_token_id)
    NEWLINE_ID = generator.tokenizer('\n').input_ids[0]
    generator.model.config.eos_token_id = NEWLINE_ID # replaced from 50256 so generation ends at newline
    generator.model.config.pad_token_id = generator.model.config.eos_token_id
    end = time.time()
    print(f'Model loaded; {end - start:.2f}s elapsed')

    print("Loading interactions...")
    with sqlite3.connect(db_fn) as con:
        intrs = pd.read_sql('SELECT * FROM data', con).interaction
    print(len(intrs), f"interactions ({intrs.nunique()} unique)")

    print("Loading parsed WW data...")
    ww_df = pd.read_csv(data_fn)
    ww_df['interactions'] = ww_df.parses.str.split(';')
    print(f"Loaded: {len(ww_df)} rows")

    def get_prompt(n_examples=10, interaction=None):
        # if no interaction passed, a random one is used
        ww_samples = ww_df.sample(n_examples)
        ww_intrs = ww_samples.interactions.apply(np.random.choice)
        
        output = ''
        for i in range(n_examples):
            ww_intr = ww_intrs.iloc[i]
            ww_cap = ww_samples.encoded.iloc[i]
            ww_intr, ww_cap = names2random_names(ww_intr, ww_cap)
            output += 'Caption of image showing ' + ww_intr + ':\n'
            output += ww_cap + '\n\n'
        
        if interaction is None:
                interaction = intrs.sample(1).iloc[0]
        interaction, names = masks2random_names(interaction, unisex=True)
        
        output += 'Caption of image showing ' + interaction + ':\n'
        
        return interaction, output, names
    
    def caption_gen():
        while True:
            I, p, N = get_prompt()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # suppress sequential inference warnings
                text = generate_from_prompt(generator, p)
            if text is not None and text != '':
                if basic_filter(text, N):
                    yield I, text, N

    output_fn = 'output/generated_captions.db'
    exists = os.path.exists(output_fn)
    with sqlite3.connect(output_fn) as con:
        cur = con.cursor()
        if exists:
            print("Appending to existing DB:", output_fn)
        else:
            print("Creating DB:", output_fn)
            os.makedirs('output', exist_ok=True)
            cur.execute('''
                CREATE TABLE data (
                    interaction text,
                    caption text,
                    names text
                )
                ''')
        
        n = len(pd.read_sql('SELECT * FROM data', con))
        print("Number of existing records:", n)                

        gen = caption_gen()
        pbar = tqdm(gen)
        for I, text, N in tqdm(gen):

            names = ';'.join(N)

            pbar.set_description(text[:50] + '...')

            query = 'INSERT INTO data (interaction, caption, names) VALUES (?, ?, ?)'
            params = (I, text, names)
            cur.execute(query, params)
            con.commit()


if __name__ == "__main__":
    main()