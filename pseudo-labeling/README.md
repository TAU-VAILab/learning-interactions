# Creating Pseudo-Labels

## Requirements

For scripts in this repo:
* Python >= 3.8
* In a new virtual environment: `pip install -r requirements.txt`
* Spacy model installation: `python -m spacy download en_core_web_trf`
* Download pre-parsed Who's Waldo caption data `ww_parses.csv.gz` ([link](https://drive.google.com/drive/u/1/folders/1PT1CWLEdGhT72h2G96M6mo9pn1VlBItL)) and extract to the `output/` folder.
  * Contains WW captions (only from train set) along with syntactic parsing interactions.
  * Download `first-names.txt` from the [random-name repo](https://github.com/dominictarr/random-name) and place in the `name_lists/` folder.

For training the summarization model:
* Recommended: use a separate environment from other scripts to avoid conflicting dependencies
* Download PyTorch `run_summarization.py` script from transformers library (tested with script from release tag v4.18.0, [on GitHub](https://github.com/huggingface/transformers/blob/31ec2cb2badfbdd4c1ac9c6c9b8a74e974984206/examples/pytorch/summarization/run_summarization.py)).
* Install libraries from accompanying `requirements.txt`

All scripts assume a single GPU.

## Synthetic Caption Data

To download pre-calculated data, download and extract `synthetic_captions.csv` as described in the [data documentation](../data/README.md), and place it in `output/`.

To generate this data yourself, run the following scripts in order:

* `parse_cc_news_captions.py`
   * Takes about 30 minutes.
* `generate_new_interactions.py`
   * Run for as long as you would like (terminate with ctrl-c) to generate the desired number of synthetic interactions.
* `generate_new_captions.py`
   * Run for as long as you would like (terminate with ctrl-c) to generate the desired number of synthetic captions.
* `filter_new_captions.py`

Outputs will be saved in `output/`, including the final caption data `output/synthetic_captions.csv`.

## Summarization Model

You may use pretrained weights for the summarization model, available [here](https://drive.google.com/drive/folders/1kZqjVPFnztWJeedSJ6FyOUZsZyLM7mi5?usp=sharing) (`summarization_model.tar.gz`).

If you would like to train the summarization model yourself, use `run_summarization.py` (see above for download instructions):

```
run_summarization.py \
--model_name_or_path t5-base \
--output_dir output/summarization_model \
--train_file output/synthetic_captions.csv \
--text_column caption \
--summary_column interaction_no_pp \
--do_train True \
--do_eval False \
--num_train_epochs 3 \
--per_device_train_batch_size 8 \
--logging_dir output/summarization_logs \
--save_total_limit 1 \
--source_prefix summarize:
```

## Inference to Create Pseudo-Labels

Run:
```
python create_pseudolabels.py -ww (path to Who's Waldo)
```

See `--help` for all arguments, including specifying the model weights (`-m`) and output filename (`-o`). By default output is saved to `output/phhi_raw.csv`.

## Postprocessing

Run:
```
python postprocess_pseudolabels.py 
```

See `--help` for all arguments, including specifying the input (`-i`) and output (`-o`) filenames. By default input is from `output/phhi_raw.csv` and output is saved to `output/phhi.csv`.

