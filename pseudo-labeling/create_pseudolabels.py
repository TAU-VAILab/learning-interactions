from transformers import pipeline
import pandas as pd
import time
from tqdm.auto import tqdm
from names import encode_names
from transformers import logging as transformers_logging
from torch.utils.data import Dataset
from argparse import ArgumentParser
import os
from glob import glob

def get_opts():
    parser = ArgumentParser()
    
    parser.add_argument('--output', '-o', type=str, default='output/phhi_raw.csv')
    parser.add_argument('--inference_batch_size', '-b', type=int, default=4)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=20)
    parser.add_argument('--model', '-m', type=str, default='output/summarization_model')
    parser.add_argument('--whos_waldo_dir', '-ww', type=str, required=True)
    parser.add_argument('--multiple_detections_fn', type=str, default='id_lists/multiple_detection_ids.txt')
    return parser.parse_args()


def get_ww_ids(ww_dir, multiple_detection_ids):
    ids = {
        os.path.basename(x)
        for x in glob(os.path.join(ww_dir, '*'))
    } - {'splits'}

    ids &= multiple_detection_ids

    return ids

# use dataset for faster pipeline inference
class EncodedCaptionDataset(Dataset):

    def __init__(self, ww_dir, ids):
        self.ww_dir = ww_dir
        self.ids = ids

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        ID = self.ids[idx]
        with open(os.path.join(self.ww_dir, ID, 'caption.txt'), 'r') as f:
            cap = f.read().strip()
        cap_e = encode_names(cap)
        return cap_e

def main():

    transformers_logging.set_verbosity_error() # suppress long warnings about input length

    args = get_opts()

    output_fn = args.output
    bsz = args.inference_batch_size
    min_length = args.min_length
    max_length = args.max_length
    model_dir = args.model
    ww_dir = args.whos_waldo_dir
    multiple_detections_fn = args.multiple_detections_fn
    
    assert os.path.exists(model_dir), f'Missing model: {model_dir}'
    assert os.path.exists(ww_dir), f"Missing Who's Waldo data: {ww_dir}"
    assert os.path.exists(multiple_detections_fn), f'Missing file: {multiple_detections_fn}'

    with open(multiple_detections_fn, 'r') as f:
        multiple_detection_ids = set([L.strip() for L in f.readlines()])

    print('Loading summarization model from:', model_dir)
    start = time.time()
    pipe = pipeline('summarization', model=model_dir, device=0)
    pipe.model.eval()
    end = time.time()
    print(f'Summarization model loaded; {end - start:.2f}s elapsed')
    
    print('Retrieving WW IDs (with multiple detections)...')
    start = time.time()
    end = time.time()
    ids = list(get_ww_ids(ww_dir, multiple_detection_ids))
    print(f'WW IDs retrieved ({len(ids)} total); {end - start:.2f}s elapsed')

    ds = EncodedCaptionDataset(ww_dir, ids)


    interactions = [obj[0]['summary_text'] for obj in tqdm(
        pipe(ds, min_length=min_length, max_length=max_length, batch_size=bsz)
        , total=len(ids)
        , desc="Generating pseudolabels"
    )]
    
    df = pd.DataFrame({'id': ids, 'pseudolabel': interactions})

    print('Saving to:', output_fn)
    df.to_csv(output_fn, index=False)
    print('done')


if __name__ == '__main__':
    main()