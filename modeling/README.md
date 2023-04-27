# Modeling

## Requirements

These scripts require access to Who's Waldo and our pseudo-label data; see the accompanying [data subrepo](../data) for details.

To run scripts in this repo:
* Python >= 3.8
* In a new virtual environment: `pip install -r requirements.txt`

Fine-tuning CLIPCap requires pretrained model weights (either trained on CC or COCO); please download from the [official CLIPCap repo](https://github.com/rmokady/CLIP_prefix_caption) (MLP mapping network weights).

For evaluation on Waldo and Wenda:
* Download all images as `.jpg` files
* Put in directory (can be in subdirectories)
* Filenames: `WENDA_ID.jpg`

Evaluation also requires downloading the BLEURT-20 checkpoint, available from the [official BLEURT repo](https://github.com/google-research/bleurt).

All scripts assume a single GPU.

## Training

Run:
```
python src/train.py -ww (path to Who's Waldo) (-cc ...) (-e ...)
```

Selected flags: (see `--help` for more):
* Encdec by default, add `-cc` for CLIPCap followed by the location of pretrained CLIPCap weights
* `-e ...` to specify number of epochs (default: 1)
* Checkpoints are saved in relevant subdirectory of `lightning_logs/`.
* Uses CUDA by default, if available.

Other details:
* Uses pHHI from `../data/phhi.csv` by default (optionally pass filename with `-p`).
* Advanced settings can be modified in `config.yml`.

## Pretrained Checkpoint

You may also use our pretrained checkpoint for our best-performing model (CLIPCap fine-tuned on our pseudo-labels), available [here](https://drive.google.com/drive/folders/1kZqjVPFnztWJeedSJ6FyOUZsZyLM7mi5?usp=sharing) (`cc_ft.tar.gz`).

Please note that this checkpoint has been computed from scratch and may differ slightly from results in the visualization and paper due to randomness during training.

## Inference

For inference, run:
```
python src/test.py -d ... -o ... -c ... (-cc ...) (-v) 
```

Selected flags (see `--help` for more):
* `-d`: Pass directory containing `.jpg` image files to run inference on (searches recursively in directory)
* `-o`: output filename (json file)
* `-c`: Pass checkpoint (e.g. `lightning_logs/version_0/checkpoints/last.ckpt`)
* `-cc`: When using CLIPCap (see above)
* `-v`: verbose mode (display predictions as they are generated)

The predictions are saved to a JSON file (at the filename specified by `-o`) containing top beam search predictions for each image (listed in decreasing probability order).

## Evaluation

To evaluate results (calculating metrics on top 1, 5, and 8 beams), run:

```
python src/eval.py -r ... -b ...
```

Pass the results json filename with `-r`, and the BLEURT checkpoint (location of downloaded BLEURT-20 checkpoint) with `-b`.

By default this uses the Waldo and Wenda benchmark data at `../data/waldo_and_wenda.csv`, assuming that each results item has an ID field corresponding to an item in Waldo and Wenda.