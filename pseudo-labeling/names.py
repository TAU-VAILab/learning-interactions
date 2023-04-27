import numpy as np
import os

for fn in ['dummy_names.txt', 'dummy_names_unisex.txt', 'first-names.txt']:
    assert os.path.exists(f'name_lists/{fn}'), f'Missing file: name_lists/{fn}'

with open('name_lists/dummy_names.txt', 'r') as f:
    NAMES = [x.strip() for x in f.readlines()]

with open('name_lists/dummy_names_unisex.txt', 'r') as f:
    UNISEX_NAMES = [x.strip() for x in f.readlines()]
    
with open('name_lists/first-names.txt', 'r') as f:
    FIRST_NAMES = [x.strip() for x in f.readlines()]

def encode_names(caption, unisex=False):
    txt = ''
    fragments = caption.split('[NAME]')
    for i, fragment in enumerate(fragments):
        txt += fragment
        if i != len(fragments) - 1:
            txt += (UNISEX_NAMES if unisex else NAMES)[i % len(NAMES)]
    return txt

def mask_names(text, unisex=False):
    for NAME in UNISEX_NAMES if unisex else NAMES:
        text = text.replace(NAME, '[NAME]')
    return text
    
def random_name():
    return np.random.choice(FIRST_NAMES)

def random_name_mapping(unisex=False):
    names = UNISEX_NAMES if unisex else NAMES
    chosen = np.random.choice(FIRST_NAMES, len(NAMES) + 1)
    mapping = {
        N: chosen[i]
        for i, N in enumerate(names)
    }
    mapping['Alice'] = chosen[-1]
    return mapping

def masks2random_names(text, return_names=True, unisex=True):
    mapping = random_name_mapping(unisex=unisex)
    keys = list(mapping.keys())
    s = text.split('[NAME]')
    output = ''
    names = []
    for i in range(len(s)):
        output += s[i]
        if i < len(s) - 1:
            K = keys[i % len(keys)]
            N = mapping[K]
            output += N
            names.append(N)
    if return_names:
        return output, names
    return output

def names2random_names(desc, text, unisex=False):
    mapping = random_name_mapping(unisex=unisex)
    output_desc = desc
    output = text
    for name in mapping:
        output_desc = output_desc.replace(name, mapping[name])
        output = output.replace(name, mapping[name])
    return output_desc, output