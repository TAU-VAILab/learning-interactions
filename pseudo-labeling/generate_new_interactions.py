
import pandas as pd
import os
from transformers import pipeline
import time
from names import encode_names, mask_names
import re
from transformers import logging as transformers_logging
import logging
import warnings
import sqlite3
from tqdm.auto import tqdm

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
        temperature=1.,
        no_repeat_ngram_size=3
    )
    
    if len(output) > 0:
        output_text = output[0].get('generated_text', '')
        return output_text[len(prompt):].strip()

def check_output(output):
    if output is None:
        return False
    if not ('Alex' in output and 'Bailey' in output):
        return False
    if not re.search('^Alex .* Bailey', output):
        return False
    if not len(output.split('Alex')) == 2:
        return False
    if not len(output.split('Bailey')) == 2:
        return False
    
    # heuristic to avoid named entities:
    # don't allow uppercase letters besides Alex/Bailey
    # (note: this avoids interactions with >2 names)
    noab = output.replace('Alex', '').replace('Bailey', '')
    if noab != noab.lower():
        return False
    
    # heuristic: require -ing word (hopefully verb)
    if 'ing' not in output:
        return False

    if output.endswith(' the') or output.endswith(' a'):
        return False
    
    return True

def main():

    transformers_logging.set_verbosity_error() # suppress padding warnings

    input_fn = 'output/cc_news_parsed.csv'
    assert os.path.exists(input_fn), f'Missing data; generate this first: {input_fn}'

    print("Loading model...")
    start = time.time()
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)
    print('Replacing EOS token ID. Original:', generator.model.config.eos_token_id)
    NEWLINE_ID = generator.tokenizer('\n').input_ids[0]
    generator.model.config.eos_token_id = NEWLINE_ID # replaced from 50256 so generation ends at newline
    generator.model.config.pad_token_id = generator.model.config.eos_token_id
    end = time.time()
    print(f'Model loaded; {end - start:.2f}s elapsed')

    print("Loading data...")
    df = pd.read_csv(input_fn)
    print(len(df), 'parsed CC-News captions with interactions')
        
    intrs = pd.Series(x for item in df.parsed for x in item.split(';'))
    print(len(intrs), 'CC-News interactions')

    def get_intr_prompt(n_examples=10):
        return '\n'.join(intrs.sample(n_examples).apply(encode_names, unisex=True).to_list()) + '\n'

    def interaction_gen():
        while True:
            prompt = get_intr_prompt()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # suppress sequential inference warnings
                output = generate_from_prompt(generator, prompt)
            if check_output(output):
                yield mask_names(output, unisex=True)
    
    output_fn = 'output/generated_interactions.db'
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
                    interaction TEXT
                )
                ''')
        
        n = len(pd.read_sql('SELECT * FROM data', con))
        print("Number of existing records:", n)                

        gen = interaction_gen()
        pbar = tqdm(gen)
        for intr in tqdm(gen):
            pbar.set_description(intr)
            query = 'INSERT INTO data (interaction) VALUES (?)'
            params = (intr,)
            cur.execute(query, params)
            con.commit()

if __name__ == "__main__":
    main()