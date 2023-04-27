import os
import json
import pandas
from argparse import ArgumentParser
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
import gensim.downloader as api
from transformers import BartForSequenceClassification, BartTokenizer
from bleurt.score import BleurtScorer
import torch

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--results', '-r', type=str, required=True)
    parser.add_argument('--waldo_and_wenda', '-w', type=str, default='../data/waldo_and_wenda.csv')
    parser.add_argument('--bleurt', '-b', type=str, required=True)
    return parser.parse_args()


def main():

    EVALUATE_AT = [1, 5, 8]
    MAX_AT = max(EVALUATE_AT)

    args = get_opts()

    assert os.path.exists(args.results), f'Missing results file: {args.results}'
    assert os.path.exists(args.waldo_and_wenda), f'Missing Wenda file: {args.waldo_and_wenda}'
    assert os.path.exists(args.bleurt), f'Missing BLEURT checkpoint: {args.bleurt}'

    with open(args.results, 'r') as f:
        res = json.load(f)
    
    wenda_df = pd.read_csv(args.waldo_and_wenda, dtype={'id': object})
    id2source = wenda_df.set_index('id').source.to_dict()

    res_ids = {x['ID'] for x in res.values()}
    wenda_ids = set(wenda_df.id)

    n_missing = len(wenda_ids - res_ids)
    extras = res_ids - wenda_ids
    assert len(extras) == 0, f'Unrecognized ID(s) (e.g. {list(extras)[0]})'
    if len(extras) > 0:
        print(f'Warning: missing {n_missing} Wenda IDs in results')
    id_counts = Counter(x['ID'] for x in res.values())
    if max(id_counts.values()) > 1:
        print(f'Warning: duplicated ID(s) in results (e.g. {id_counts.most_common()[0][0]})')

    print('Downloading evaluation models...')
    print('GloVe...')
    glove = api.load('glove-wiki-gigaword-200')
    print('NLI...')
    nli_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')
    nli_model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to('cuda').eval()
    print('BLEURT...')
    bleurt_scorer = BleurtScorer(args.bleurt)

    def get_bleurt_score(pred, label):
        references = [label.replace('_', 'person')]
        candidates = [pred.replace('_', 'person')]
        return bleurt_scorer.score(references=references, candidates=candidates)[0]

    @torch.no_grad()
    def get_nli_scores(pred, caption):

        input_ids = nli_tokenizer.encode(caption, pred, return_tensors='pt').to('cuda')
        logits = nli_model(input_ids)[0]
        probs = logits.softmax(dim=1)[0]

        pc, _, pe = probs

        return pe.item(), pc.item()

    def get_verb_sim(pred, label):
        v0 = pred.split()[1] if len(pred.split()) > 1 else ''
        v1 = label.split()[1]
        if v0 in glove and v1 in glove:
            return glove.similarity(v0, v1)
        return 0

    def score(ID, idx, pred, label, caption):
        bleurt_score = get_bleurt_score(pred, label)
        pe, pc = get_nli_scores(pred, caption)
        verb_sim = get_verb_sim(pred, label)
        return {
            'id': ID,
            'index': idx,
            'BL': bleurt_score,
            'pe': pe,
            'pc': pc,
            'sim': verb_sim
        }

    print('Running evaluation...')
    id2gt = wenda_df.set_index('id').to_dict()
    pbar = tqdm(res.items())
    scores = []
    for fn, item in pbar:
        ID = item['ID']
        pbar.set_description(f'Processing ID {ID}')

        preds = item['preds']
        assert len(preds) >= MAX_AT, f'Too few predictions ({len(preds)}) for ID: {ID}'
        preds = preds[:MAX_AT]

        label = id2gt['label'][ID]
        caption = id2gt['caption'][ID]

        scores += [
            score(ID, i, pred, label, caption)
            for i, pred in enumerate(preds)
        ]

    print('Calculating aggregate metrics')
    df = pd.DataFrame(scores)

    for AT in EVALUATE_AT:
        print(F'Scores@{AT}:')
        subdf = df[df['index'] < AT].copy().groupby('id').agg({
            'sim': max,
            'BL': max,
            'pe': max,
            'pc': min
        }).reset_index()
        subdf['source'] = subdf.id.map(id2source)
        subdf = subdf.groupby('source').agg({
            'sim': 'mean',
            'BL': 'mean',
            'pe': 'mean',
            'pc': 'mean'
        }).T
        subdf['overall'] = subdf.mean(axis=1)
        print(subdf)
        print()


if __name__ == '__main__':
    main()
