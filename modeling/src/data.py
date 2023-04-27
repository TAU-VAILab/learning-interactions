from glob import glob
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
import torch
from transformers import CLIPProcessor, GPT2TokenizerFast
import time
from torch.utils.data import DataLoader


def get_ww_ids(cfg, filter_by_detections=True):
    ww_dir = cfg['data']['ww_dir']
    ids = {
        os.path.basename(x)
        for x in glob(os.path.join(ww_dir, '*'))
    } - {'splits'}

    if filter_by_detections:
        fn = cfg['data']['ww_detections_fn']
        df_detections = pd.read_csv(fn, dtype={'id': object})
        matching = set(df_detections[df_detections.n_detections > 1]['id'])
        print(f'Filtering to only use >1 detection samples; {len(ids & matching)} left')
        ids &= matching

    return ids

class FolderData:

    def __init__(self, folder, load_img=False):
        self.folder = folder
        
        self.id = self.folder.split('/')[-1]
        
        self.files = glob(f'{folder}/*')
        
        if load_img:
            self.image = Image.open(f'{folder}/image.jpg')
    
            if self.image.mode != 'RGB':
                self.image = self.image.convert('RGB')


def load_id(cfg, ID, **kwargs):
    path = os.path.join(cfg['data']['ww_dir'], ID)
    return FolderData(path, **kwargs)

class DecoderTokenizer:

    def __init__(self, data_cfg):
        # GPT2 tokenizer needed for encoding target token ids or for decoding predictions
        print('Loading GPT2 tokenizer...')
        start = time.time()
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(data_cfg['gpt2_pretrained'])
        self.BOS = self.gpt2_tokenizer.bos_token_id
        self.EOS = self.gpt2_tokenizer.eos_token_id # same as BOS for GPT2
        self.bos_token = self.gpt2_tokenizer.bos_token
        self.eos_token = self.gpt2_tokenizer.eos_token
        assert data_cfg['pad_token'] in self.gpt2_tokenizer.vocab
        self.gpt2_tokenizer.pad_token = data_cfg['pad_token'] # new padding token: helps with verb objective
        self.PAD = self.gpt2_tokenizer.pad_token_id
        self.pad_token = self.gpt2_tokenizer.pad_token
        
        end = time.time()
        print(f'GPT2 tokenizer loaded; {end - start:.2f}s elapsed')

    def add_special_tokens(self, text):
        return self.bos_token + text + self.eos_token

class ImageDataset(IterableDataset):

    def __init__(self, df, ww_dir, batch_size, collate_fn):
        self.df = df
        self.ww_dir = ww_dir
        # ^ (if True, loads caption.txt files)

        self.batch_size = batch_size

        self.iters_per_epoch = self.df.shape[0] // self.batch_size
        # ^ pseudo-epoch: enough steps to possibly cover all samples

        self.collate_fn = collate_fn

    def __len__(self):
        return self.iters_per_epoch
        # ^ required for PL to display epochs
    
    def load_id(self, ID):
        path = os.path.join(self.ww_dir, ID)
        return FolderData(path, load_img=True)

    def __iter__(self):

        while True:
            rows = self.df.sample(self.batch_size)
            
            batch = []
            for _, row in rows.iterrows():

                ID = row.id
                F = self.load_id(ID)
                data = row.to_dict()
                data['image'] = F.image
            
                batch.append(data)

            yield self.collate_fn(batch)

class Collator:

    def __init__(self, data_cfg, dec_tokenizer):

        self.dec_tokenizer = dec_tokenizer

        # CLIP processor needed for data collation
        print('Loading CLIP processor...')
        start = time.time()
        self.input_processor = CLIPProcessor.from_pretrained(data_cfg['clip_pretrained'])
        end = time.time()
        print(f'CLIP processor loaded; {end - start:.2f}s elapsed')

    def collate(self, batch):
        batch_size = len(batch)

        data = {
            'batch_size': batch_size
        }
        keys = batch[0].keys() # can assume all samples have the same keys
        for k in keys:
            val_list =  [item[k] for item in batch]
            if k == 'image':
                encoder_inputs = self.input_processor(
                    images=val_list,
                    return_tensors='pt')
                data['model_inputs'] = {
                    'encoder_inputs': encoder_inputs
                }
            elif k == 'weight':
                # numeric data: can convert into tensor
                data[k] = torch.tensor(val_list)
            else:
                data[k] = val_list

        if 'pseudolabel' in keys: # train time
            texts = [item['pseudolabel'][2:] for item in batch] # [2:] to remove initial "_ " (predictable)
            texts_with_special = [
                self.dec_tokenizer.add_special_tokens(x)
                for x in texts
            ]

            tokens = self.dec_tokenizer.gpt2_tokenizer(texts_with_special, padding=True)
            token_ids = torch.tensor(tokens.input_ids, dtype=int)

            data['model_inputs']['input_ids'] = token_ids
            data['labels'] = token_ids
            # note HF API shifts labels by one within the model, so labels==input_ids

        return data


class WWData:
    def __init__(self, data_cfg, use_wenda=False, ww_dir=None, phhi_fn=None, n_workers=1, batch_size=1):

        assert ww_dir is not None, 'Missing WW dir'
        assert phhi_fn is not None, 'Missing pHHI filename'

        self.df_train = pd.read_csv(phhi_fn, dtype={'id': object})

        self.batch_size = batch_size
        self.n_workers = n_workers

        self.dec_tokenizer = DecoderTokenizer(data_cfg)
        self.special_tokens = {
            'BOS': self.dec_tokenizer.BOS,
            'EOS': self.dec_tokenizer.EOS,
            'PAD': self.dec_tokenizer.PAD
        }

        self.collator = Collator(data_cfg, dec_tokenizer=self.dec_tokenizer)

        print('Creating dataset & dataloader objects...')
        self.train_dataset = ImageDataset(
            self.df_train,
            ww_dir,
            self.batch_size,
            self.collator.collate)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=None, # batches are already collated in DS object
            num_workers=self.n_workers,
            pin_memory=True)

class TestDataset(Dataset):

    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img = Image.open(fn).convert('RGB')
        data = {
            'filename': fn,
            'image': img
        }
        return data

class TestData:
    def __init__(self, data_cfg, img_dir=None, batch_size=1, n_workers=0):

        assert img_dir is not None, 'Missing img_dir'

        self.img_dir = img_dir
        self.fns = glob(os.path.join(self.img_dir, '**', '*.jpg'), recursive=True)
        print(f'Test data: {len(self.fns)} .jpg files found')
        assert len(self.fns) > 0, f'No .jpg files found in image directory: {self.img_dir}'

        self.dec_tokenizer = DecoderTokenizer(data_cfg)
        self.special_tokens = {
            'BOS': self.dec_tokenizer.BOS,
            'EOS': self.dec_tokenizer.EOS,
            'PAD': self.dec_tokenizer.PAD
        }

        self.collator = Collator(data_cfg, dec_tokenizer=self.dec_tokenizer)

        print('Creating dataset & dataloader objects...')
        self.test_dataset = TestDataset(self.fns)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=True,
            collate_fn=self.collator.collate
        )

        self.decode = self.dec_tokenizer.gpt2_tokenizer.decode