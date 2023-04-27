# Learning Human-Human Interactions in Images from Weak Textual Supervision (ICCV 2023)

[Project Page](https://learning-interactions.github.io/) | [Paper](https://arxiv.org/abs/2304.14104) | [Interactive Visualization](https://tau-vailab.github.io/learning-interactions/web/viz.html)

This is the official repository for the paper: *Morris Alper and Hadar Averbuch-Elor (2023). Learning Human-Human Interactions in Images from Weak Textual Supervision. ICCV 2023*

## Accompanying Data

See [the data documentation](data) for information on using the accompanying data, including:
* **Waldo and Wenda** benchmark for HHI understanding
* IDs for **imSitu-HHI** subset of the [imSitu](http://imsitu.org/) dataset
* **pHHI** (pseudo-labels indicating HHI) for the [Who's Waldo dataset](https://whoswaldo.github.io)
* **Synthetic caption data** for training summarization model

## Creating Pseudo-Labels

See [the pseudo-labeling documentation](pseudo-labeling) for information on training the summarization model and using it to generate pseudo-labels for the Who's Waldo dataset. Alternatively, you may use pre-computed pseudo-labels (pHHI) – see above.

## Modeling (Training & Pretrained Checkpoint, Inference, Evaluation)

See [the modeling documentation](modeling) for information on training the HHI understanding model (or using a pretrained checkpoint), and running inference and evaluation.

## Licence

We release our code under the [MIT license](https://opensource.org/license/mit/). Please see [the data documentation](data) for licensing of accompanying data.

## Citation

If you find this code or our data helpful in your research or work, please cite the following paper.
```
@InProceedings{alper2023learning,
    author    = {Morris Alper and Hadar Averbuch-Elor},
    title     = {Learning Human-Human Interactions in Images from Weak Textual Supervision},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023}
}
```