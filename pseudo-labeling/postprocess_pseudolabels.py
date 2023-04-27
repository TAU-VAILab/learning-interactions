# from transformers import pipeline
# import torch
import pandas as pd
# import time
from tqdm.auto import tqdm
# from names import encode_names
# from transformers import logging as transformers_logging
# from torch.utils.data import Dataset
from argparse import ArgumentParser
import os
# from glob import glob

ing_words = {
    'during', 'thing', 'sing', 'ring', 'ding', 'king', 'zing', 'bling',
    'cling', 'bring', 'swing', 'wring', 'sting',
}

def contains_ing_verb_fast(text):
    return any(
        word.endswith('ing') and word not in ing_words
        and word != 'during'
        for word in text.split()
    )

banned_words = ['photo', 'image', 'picture']

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='output/phhi_raw.csv')
    parser.add_argument('--output', '-o', type=str, default='output/phhi.csv')
    parser.add_argument('--ignore', type=str, default='id_lists/ignore.txt')
    parser.add_argument('--hashes', type=str, default='id_lists/hashes.csv.gz')
    parser.add_argument('--wenda', type=str, default='../data/waldo_and_wenda.csv')
    parser.add_argument('--weight_exp', type=float, default=0.25)
    return parser.parse_args()


def main():
    args = get_opts()
    input_fn = args.input
    output_fn = args.output
    ignore_fn = args.ignore
    wenda_fn = args.wenda
    weight_exp = args.weight_exp

    assert os.path.exists(input_fn), f'Missing file: {input_fn}'
    assert os.path.exists(ignore_fn), f'Missing file: {ignore_fn}'
    assert os.path.exists(wenda_fn), f'Missing file: {wenda_fn}'

    print("Loading data from:", input_fn)
    df = pd.read_csv(input_fn, dtype={'id': object})
    print("Data loaded. Rows:", len(df))

    with open(ignore_fn, 'r') as f:
        ignore_ids = set(L.strip() for L in f.readlines())
    print(f"Removing ignored IDs ({len(ignore_ids)})...") # duplicates, samples with same artist or year-month-day as test samples
    df = df[~df.id.isin(ignore_ids)]
    print("Remaining rows:", len(df))

    wenda_df = pd.read_csv(wenda_fn, dtype={'id': object})
    test_ids = set(wenda_df[wenda_df.source == 'ww'].id)
    print(f"Removing test IDs ({len(test_ids)})...")
    df = df[~df.id.isin(test_ids)]
    print("Remaining rows:", len(df))

    print('Filtering out entries without an -ing verb...')
    # contains_ing_verb_fast
    tqdm.pandas(desc='Checking for verbs')
    has_verb = df.pseudolabel.apply(contains_ing_verb_fast)
    df = df[has_verb].copy()
    print('Remaining rows:', df.shape[0])

    print('Filtering out invalid formats...')
    valid_format = df.pseudolabel.str.match('^_ [^ ]*ing [^_]*_[^_]*$')
    df = df[valid_format].copy()
    print('Remaining rows:', df.shape[0])

    print('Filtering out banned words...')
    contains_banned_word = df.pseudolabel.str.contains('|'.join(banned_words))
    df = df[~contains_banned_word].copy()
    print('Remaining rows:', df.shape[0])

    print('Setting weights...')
    counts_dict = df.pseudolabel.value_counts().to_dict()
    text2weight = {
        v: (1 / c)
        for v, c in counts_dict.items()
    }
    weight = df.pseudolabel.map(text2weight)
    weight *= len(df) / weight.sum()
    df['weight'] = weight
    df.weight = df.weight ** weight_exp
    df.weight /= df.weight.mean()

    print('Adjusting weights based on repeated captions')
    print('Loading hash data from:', args.hashes)
    df_hashes = pd.read_csv(args.hashes, dtype='object')
    id2hash = df_hashes.set_index('id').hash.to_dict()
    df['hash'] = df.id.map(id2hash)
    assert df.hash.notna().all(), 'Missing ID(s) in hashes'
    print('Calculating multipliers...')
    h2v = df.hash.value_counts().to_dict()
    multipliers = df.hash.apply(lambda h: 1 / h2v[h])
    print('Adjusting weights by multipliers...')
    df.weight *= multipliers
    del df['hash']

    print("Saving to:", output_fn)
    df.to_csv(output_fn, index=False)
    print("done")

if __name__ == "__main__":
    main()