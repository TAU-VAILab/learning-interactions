
import pandas as pd
import os
import time
import sqlite3
from tqdm.auto import tqdm
from transformers import BartForSequenceClassification, BartTokenizer
import torch
import spacy
from functools import lru_cache
import re

banned_words = [
    'photo', 'image', 'picture', 'in this', 'In this'
]
banned_regex = '|'.join([fr'\b{x}\b' for x in banned_words])

def main():
    db_fn = 'output/generated_captions.db'
    assert os.path.exists(db_fn), f'Missing DB: {db_fn}'

    print("Loading models...")
    start = time.time()
    nlp = spacy.load('en_core_web_trf')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    model.to('cuda')
    model.eval()
    end = time.time()
    print(f'Models loaded; {end - start:.2f}s elapsed')

    @torch.no_grad()
    def score(premise, hypothesis):
        input_ids = tokenizer.encode(premise, hypothesis, return_tensors='pt').to('cuda')
        logits = model(input_ids)[0]
        probs = logits.softmax(dim=1)[0]
        return probs.cpu().numpy()

    print("Loading synthetic captions...")
    with sqlite3.connect(db_fn) as con:
        df = pd.read_sql('SELECT * FROM data', con)
    print(len(df), "records")

    tqdm.pandas(desc='NLI scoring')
    ents = df.progress_apply(lambda row: score(row['caption'], row['interaction']), axis=1)

    df['pc'] = ents.apply(lambda x: x[0])
    df['pn'] = ents.apply(lambda x: x[1])
    df['pe'] = ents.apply(lambda x: x[2])

    print(f"Filtering {len(df)} rows with NLI scores")
    df = df[df.pe > 0.5].copy()
    print(f"Remaining: {len(df)} rows")

    print("Loading ...")
    
    def has_verb(text):
        doc = nlp(text)
        return any(tok.pos_ == 'VERB' for tok in doc)
    tqdm.pandas(desc='Filtering to require a verb')
    has_verb = df.interaction.progress_apply(has_verb)
    df = df[has_verb].copy()
    print(f"Remaining: {len(df)} rows")

    print("Removing misformatted data...")
    df.interaction = df.interaction.str.replace("’", "'")
    tqdm.pandas(desc="Detecting unknown names")
    unknown_name = df.progress_apply(
        lambda row: len([
        w for w in row['interaction'].split() if w != w.lower() and w not in row['names'].split(';')
            and not(w.endswith("'s") and w[:-2] in row['names'].split(';'))
        ]) > 0
        , axis=1
    )
    to_remove = df.interaction.str.contains(banned_regex, regex=True) | unknown_name
    df = df[~to_remove].copy()
    print(f"Remaining: {len(df)} rows")

    def mask_names(row):
        intr = row['interaction']
        names = row['names'].split(';')
        for name in names:
            intr = re.sub(rf'\b{name}\b', '_', intr)
        return intr
    tqdm.pandas(desc="Masking names")
    df['interaction'] = df.progress_apply(mask_names, axis=1)

    print("Removing items with too many names")
    too_many_people = df.interaction.str.contains('_.*_.*_', regex=True)
    df = df[~too_many_people].copy()
    print(f"Remaining: {len(df)} rows")

    PATTERNS = [
        '_ with _',
        '_ with _ and _',
        '_ with _ & _',
        '_ and _ with _',
        '_ & _ with _',
    ]
    for i in range(1, 10):
        x = ', '.join(['_'] * i)
        for y in ['and', '&']:
            PATTERNS += [x + f', {y} _']
            PATTERNS += [x + f' {y} _']
    # longest first, so they are replaced first:
    PATTERNS = sorted(PATTERNS, key=len, reverse=True)

    def normalize_subjects(intr):
        # e.g. "_ and _ shaking hands" => "_ shaking hands with _"
        for p in PATTERNS:
            if intr.startswith(p + ' '):
                return '_' + intr[len(p):] + ' with _'
        return intr

    tqdm.pandas(desc="Normalizing subjects")
    normed = df.interaction.progress_apply(normalize_subjects)
    print((df.interaction != normed).sum(), "will be modified")
    df.interaction = normed

    print("Removing invalid formats...")
    df.interaction = df.interaction.str.replace('^_ is ([^ ]*ing)', r'_ \1', regex=True)
    valid_format = df.interaction.str.match('^_ [^ ]*ing [^_]*_[^_]*$')
    df = df[valid_format].copy()
    print(f"Remaining: {len(df)} rows")


    
    encoded = df.interaction.str.replace('_', 'Adam', 1).str.replace('_', 'Bob', 1)
    @lru_cache(maxsize=None) # memoization => faster
    def rm_pps(text):
        doc = nlp(text)
        name_toks = {w for w in doc if w.text in {'Adam', 'Bob'}}
        prep_toks = {w for w in doc if w.pos_ == 'ADP'}
        pps = {x for p in prep_toks if not any(x in name_toks for x in p.subtree) for x in p.subtree}
        return nlp(''.join([
            x.text_with_ws for x in doc if x not in pps
        ])).text.strip()
    tqdm.pandas(desc="Removing PPs")
    encoded_rm_pps = encoded.progress_apply(rm_pps)
    df['interaction_no_pp'] = encoded_rm_pps.str.replace(r'\bAdam\b', '_', regex=True).str.replace(r'\bBob\b', '_', regex=True)

    out_fn = 'output/synthetic_captions.csv'
    print("Saving to:", out_fn)
    if os.path.exists(out_fn):
        print(f'Warning: overwriting {out_fn}')
    df.to_csv(out_fn, index=False)

if __name__ == "__main__":
    main()