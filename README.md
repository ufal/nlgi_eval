NLGI-Eval
=========

Using NLI ([Roberta](https://huggingface.co/roberta-large-mnli) from HuggingFace) for data-to-text NLG evaluation. 
Tested on [WebNLG](https://webnlg-challenge.loria.fr/challenge_2017/) and [E2E](https://github.com/tuetschek/e2e-eval) datasets.

This code accompanies the following paper:

Ondřej Dušek & Zdeněk Kasner (2020): [Evaluating Semantic Accuracy of Data-to-Text Generation with Natural Language Inference](https://www.aclweb.org/anthology/2020.inlg-1.19/). In _Proceedings of INLG_.

Usage
-----

### Installation

The code requires Python 3 and some additional packages (most importantly [Transformers](https://huggingface.co/), see [`requirements.txt`](./requirements.txt)). To install, clone this repository and then run:
```
pip install -r requirements.txt
```
We recommend to use a [virtualenv](https://virtualenv.pypa.io/en/latest/) for the installation.

### Running

Run `./nli_eval.py -h` for a list of available options.

Basic usage (default settings):
```
./nli_eval --type <type> input.tsv output.json
```
The type is either `webnlg` or `e2e`, based on the domain of the data.

### Data used

For WebNLG, we used the 2017 human evaluation results file. Run [`data/download_webnlg.sh`](data/download_webnlg.sh) to download it.

E2E data are taken from [the primary systems](https://github.com/tuetschek/e2e-eval), then concatenated and processed with [the slot error script](https://github.com/tuetschek/e2e-cleaning). The result is stored here in [`data/e2e.tsv`](data/e2e.tsv) for simplicity.

### Other datasets

To use this on a different dataset, you need to provide new templates for all predicates and link them to `TEMPLATE_PATHS` in the code. You also need to implement a data loading routine such as `parse_e2e` or `parse_webnlg`.


Citing
------

```
@inproceedings{dusek_evaluating_2020,
	address = {Online},
	title = {Evaluating {Semantic} {Accuracy} of {Data}-to-{Text} {Generation} with {Natural} {Language} {Inference}},
	booktitle = {Proceedings of the 13th {International} {Conference} on {Natural} {Language} {Generation} ({INLG} 2020)},
	author = {Dušek, Ondřej and Kasner, Zdeněk},
	month = dec,
	year = {2020},
}
```
