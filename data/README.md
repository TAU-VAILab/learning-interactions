# Accompanying Data

This folder contains the following data:

* `waldo_and_wenda.csv` – **Waldo and Wenda** benchmark for HHI understanding
* `imsitu-hhi.txt` – IDs for **imSitu-HHI** subset of the [imSitu](http://imsitu.org/) dataset
* `phhi.csv` – **pHHI** (pseudo-labels indicating HHI) for the [Who's Waldo dataset](https://whoswaldo.github.io)

See the sections below for instructions on using these, as well as download instructions for **synthetic caption data**.

## Waldo and Wenda

The file `waldo_and_wenda.csv` contains metadata and ground-truth annotations for the 1,000–item **Waldo and Wenda** HHI understanding benchmark. The `source` column indicates images from:

* `ww` – [Who's Waldo](https://whoswaldo.github.io/) (300 items)
* `cc` – [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/) (400 items, from val set)
* `coco` – [Microsoft COCO](https://cocodataset.org) (300 items, from val2014 set)

WW images can be obtained by requesting access to the WW dataset (see its homepage for details). CC and COCO images are available via the listed URLs ([CC source](https://huggingface.co/datasets/conceptual_captions)). We do not reproduce image files here; see each dataset for its respective licensing details and see below for the licensing of our additions.

The `caption` column provides ground-truth captions from the source datasets. Note that WW captions have named person entities replaced with an underscore, and COCO samples use the first reference from the original dataset as the listed caption.

The `id` column contains a unique identifier for each item. For those from WW and COCO, these are the original identifiers from those datasets. For items from CC, these are the first five digits of the MD5 hash of the corresponding image URL.

## imSitu-HHI

The file `imsitu-hhi.txt` lists the items from the [imSitu](http://imsitu.org/) dataset that comprise the **imSitu-HHI** subset as described in our paper.

## pHHI

The file `phhi.csv` contains HHI pseudo-labels for relevant items in the Who's Waldo dataset. These have been preprocessed as described in our paper, including to avoid overlap with the test items in Waldo and Wenda.

Alternatively, you may generate these yourself using the code in the [pseudo-labeling subrepo](../pseudo-labeling).

For image data, please request access to Who's Waldo as described above.

## Synthetic Caption Data

You may download the synthetic caption data `synthetic_captions.csv.gz` (used for training summarization model) at [this link](https://drive.google.com/drive/folders/1PT1CWLEdGhT72h2G96M6mo9pn1VlBItL?usp=sharing).

## License

Data from the [Who's Waldo](https://whoswaldo.github.io/), [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/), [Microsoft COCO](https://cocodataset.org), and [imSitu](http://imsitu.org/) datasets are licensed according to the licensing terms of each respective dataset. We license our data contributions (ground-truth pseudo-label annotations) under the non-commercial [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.