import time
from datasets import load_dataset
from tqdm.auto import tqdm
import pandas as pd
from parser import InteractionParser
import os

patterns = [
    '(left)', '(right)', '(center)',
    ', left,', ', right,', ', center,', ', centre,',
    ', pictured,', 'Photo:', 'PHOTO: ', 'Photo by',
    'Image copyright',
    'Getty ', 'AP Photo', 'AP Image'
]

def main():

    print('Loading cc_news dataset...')
    start = time.time()
    dataset = load_dataset('cc_news', 'plain_text')
    end = time.time()
    print(f'cc_news dataset loaded; {end - start:.2f}s elapsed')

    data = []
    for i, text in enumerate(tqdm(dataset['train']['text'], desc='Searching in dataset')):
        for line in text.splitlines():
            if any(p in line for p in patterns):
                data.append({
                    'text': line,
                    'patterns': ';'.join([p for p in patterns if p in line])
                })
    df = pd.DataFrame(data)

    print(f'{df.shape[0]} captions found')

    print('Preprocessing captions...')
    df = df[df.text.str.len() < 1000].copy() # remove extremely long captions
    df['cleaned'] = df.text
    # remove some patterns
    df.cleaned = df.cleaned.str.replace(
        ', (left|right|center|centre), ', ' ', regex=True
    ).str.replace(r' \((left|right|center|centre)\)', ' '
    ).str.replace(r'\((AP ?)Photo.*\)', ''
    ).str.replace(r'\(Image.*\)', ''
    ).str.replace(r'\(Photo.*\)', ''
    ).str.replace(r'\(Credit.*\)', ''
    ).str.replace(r'\[(Featured )?Image.*\]', ''
    ).str.replace('\(?Getty Images\)?$', ''
    ).str.replace('Image copyright .* Image caption', ''
    ).str.replace('Photo: AF?PP?$', ''
    ).str.replace('FILE PHOTO:', ''
    ).str.replace('^Photo:', ''
    ).str.replace(r'Image \d+ of \d+', ' '
    ).str.replace(' +', ' ').str.strip()
    df = df[df.cleaned != ''].copy()

    print(f'{df.shape[0]} cleaned captions remaining')

    print('Loading interaction parser...')
    start = time.time()
    ip = InteractionParser(mask_names=True)
    end = time.time()
    print(f'Interaction parser loaded; {end - start:.2f}s elapsed')

    tqdm.pandas(desc='Parsing captions')
    df['parsed'] = df.cleaned.progress_apply(lambda text: list({desc for lemma, desc in ip.parse(text)}))

    df = df[df.parsed.str.len() > 0].copy()
    df.parsed = df.parsed.str.join(';')

    print(f'{df.shape[0]} captions contain at least one interaction')

    output_fn = 'output/cc_news_parsed.csv'
    os.makedirs('output', exist_ok=True)
    print('Saving to:', output_fn)
    df.to_csv(output_fn, index=False)

    print('done')

if __name__ == '__main__':
    main()