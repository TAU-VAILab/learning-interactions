from data import TestData
from model import HHIModel
from model_clipcap import HHICCModel
from tqdm.auto import tqdm
from transformers import logging as transformers_logging
import os
import time
from config import load_config
from argparse import ArgumentParser
import json

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--clipcap', '-cc', type=str)
    parser.add_argument('--beam_k', '-bk', type=int, default=32)
    parser.add_argument('--top_k', '-tk', type=int, default=8)
    parser.add_argument('--img_dir', '-d', type=str, required=True)
    parser.add_argument('--verbose' , '-v', action='store_true')
    parser.add_argument('--output', '-o', type=str, required=True)
    return parser.parse_args()

def main():

    transformers_logging.set_verbosity_error() # suppress long warnings about missing pretrained weights

    args = get_opts()
    cfg = load_config()
    use_clipcap = args.clipcap is not None
    if use_clipcap:
        assert os.path.exists(args.clipcap), f'CLIPCap weights not found: {args.clipcap}'
    if os.path.exists(args.output):
        print(f"Warning: output filename exists; will be overwritten ({args.output})")

    print('Loading data...')
    data = TestData(
        cfg['data'],
        img_dir=args.img_dir
    )
    print(len(data.test_dataset), 'test samples')

    print(f'Loading model (use_clipcap={use_clipcap})...')
    start = time.time()
    ModelClass = HHICCModel if use_clipcap else HHIModel
    kwargs = data.special_tokens
    if use_clipcap:
        kwargs['clipcap_pretrained'] = args.clipcap
    model = ModelClass.load_from_checkpoint(
        args.checkpoint,
        model_cfg=cfg['model'],
        **kwargs)
    model.eval()
    end = time.time()
    print(f'Model loaded; {end - start:.2f}s elapsed')

    print(f'Using beam search with beam_k={args.beam_k} (saving top {args.top_k})')

    output = {}
    for B in tqdm(data.test_dataloader, desc="Running inference"):
        encoder_inputs = B['model_inputs']['encoder_inputs']
        bsz = encoder_inputs['pixel_values'].shape[0]
        assert bsz == 1, 'Batch size 1 required'
        fn = B['filename'][0]
        ID = os.path.splitext(os.path.basename(fn))[0]
        beams = model.run_beam_search(encoder_inputs=encoder_inputs, beam_k=args.beam_k)
        top_beams = beams[:args.top_k]
        top_texts = ['_ ' + data.decode(indices[1:-1]) for indices, _ in top_beams]
        top_scores = [score for _, score in top_beams]
        datum = {
            'ID': ID,
            'preds': top_texts,
            'scores': top_scores
        }
        if args.verbose:
            print(fn)
            print(datum)
            print()
        output[fn] = datum

    print("Saving to:", args.output)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print('done')

if __name__ == '__main__':
    main()
