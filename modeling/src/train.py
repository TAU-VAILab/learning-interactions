from data import WWData
from model import HHIModel
from model_clipcap import HHICCModel
import time
import pytorch_lightning as pl
import torch
from transformers import logging as transformers_logging
import os
from config import load_config
from argparse import ArgumentParser


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--whos_waldo_dir', '-ww', type=str, required=True)
    parser.add_argument('--phhi', '-p', type=str, default='../data/phhi.csv')
    parser.add_argument('--clipcap', '-cc', type=str)
    parser.add_argument('--n_workers', '-n', type=int, default=10)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--learning_rate', '-lr', type=int, default=1e-5)
    return parser.parse_args()


def main():

    transformers_logging.set_verbosity_error() # suppress long warnings about missing pretrained weights

    args = get_opts()
    cfg = load_config()

    use_clipcap = args.clipcap is not None
    if use_clipcap:
        assert os.path.exists(args.clipcap), f'CLIPCap weights not found: {args.clipcap}'

    print('Loading data...')
    data = WWData(
        cfg['data'],
        ww_dir=args.whos_waldo_dir,
        phhi_fn=args.phhi,
        n_workers=args.n_workers,
        batch_size=args.batch_size)
    print(len(data.train_dataset), 'train samples')

    print(f'Creating model (use_clipcap={use_clipcap})...')
    
    start = time.time()
    ModelClass = HHICCModel if use_clipcap else HHIModel
    kwargs = data.special_tokens
    if use_clipcap:
        kwargs['clipcap_pretrained'] = args.clipcap
    model = ModelClass(cfg['model'], lr=args.learning_rate, **kwargs)
    end = time.time()
    print(f'Model created; {end - start:.2f}s elapsed')

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=-1,
            save_last=True,
            save_on_train_epoch_end=True,
            every_n_epochs=1)
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        auto_select_gpus=True,
        precision=16,
        callbacks=callbacks
    )

    trainer.fit(model, data.train_dataloader)

    print('done')

if __name__ == '__main__':
    main()
