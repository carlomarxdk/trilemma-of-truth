# The Trilemma of Truth in Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2506.23921-b31b1b.svg)](https://arxiv.org/abs/2506.23921)
[![🤗 Datasets](https://img.shields.io/badge/🤗%20Datasets-trilemma--of--truth-yellow)](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Email](https://img.shields.io/badge/Email-g.savcisens@northeastern.edu-orange)](mailto:g.savcisens@northeastern.edu)
[![DOI](https://zenodo.org/badge/986600505.svg)](https://doi.org/10.5281/zenodo.15779092)

**This repository** is the codebase for [our paper](https://arxiv.org/abs/2506.23921) on evaluating factual reasoning in large language models.  
Here you’ll find everything needed to  
1. Generate and inspect our three Trilemma data sets (city locations, drug indications, word definitions),  
2. Run zero-shot prompts,  
3. Train and evaluate a suite of probe models (from mean-difference to our sAwMIL),  

**Abstract:** We often attribute human characteristics to large language models (LLMs) and claim that they "know" certain things. LLMs have an internal probabilistic knowledge that represents information retained during training. How can we assess the veracity of this knowledge? 
We examine two common methods for probing the veracity of LLMs and discover several assumptions that are flawed. To address these flawed assumptions, we introduce `sAwMIL` (short for Sparse Aware Multiple-Instance Learning), a probing method that utilizes the internal activations of LLMs to separate statements into *true*, *false*, and *neither*. `sAwMIL` is based on multiple-instance learning and conformal prediction. We evaluate `sAwMIL` on 5 validity criteria across 16 open-source LLMs, including both default and chat-based variants, as well as on 3 new datasets. Among the insights we provide are: (1) the veracity signal is often concentrated in the third quarter of an LLM's depth; (2) truth and falsehood signals are not always symmetric; (3) linear probes perform better on chat models than on default models; (4) nonlinear probes may be required to capture veracity signals for some LLMs with reinforcement learning from human feedback or knowledge distillation; and (5) LLMs capture a third type of signal that is distinct from true and false and is neither true nor false. These findings provide a reliable method for verifying what LLMs "know" and how certain they are of their probabilistic internal knowledge.

![Abstract Pipeline](./docs/figures/flow.svg)

---

## Table of Contents

- [The Trilemma of Truth in Large Language Models](#the-trilemma-of-truth-in-large-language-models)
  - [Table of Contents](#table-of-contents)
  - [📘 Repository Overview](#-repository-overview)
    - [What is included?](#what-is-included)
    - [What is not included?](#what-is-not-included)
    - [`sAwMIL` (Sparse Aware Multiple Instance Learning) Implementation](#sawmil-sparse-aware-multiple-instance-learning-implementation)
  - [⚡ Installation](#-installation)
  - [📝 Usage \& Examples](#-usage--examples)
    - [Run the Scripts](#run-the-scripts)
      - [0. Return full error log in `Hydra`](#0-return-full-error-log-in-hydra)
      - [1. Collect Hidden Activations](#1-collect-hidden-activations)
      - [2. Run zero-shot prompt (and collect scores)](#2-run-zero-shot-prompt-and-collect-scores)
      - [3. Train *sAwMIL* probe](#3-train-sawmil-probe)
        - [3.1. One-vs-all](#31-one-vs-all)
        - [3. Multiclass](#3-multiclass)
      - [4. Single Instance Probe](#4-single-instance-probe)
        - [4.1 Train *one-vs-all SVM* probe](#41-train-one-vs-all-svm-probe)
      - [4.2 Train *multiclass SVM* probe](#42-train-multiclass-svm-probe)
        - [4.3 Train the *mean-difference* probe](#43-train-the-mean-difference-probe)
    - [Task specification](#task-specification)
  - [🗂️ Dataset](#️-dataset)
    - [Structure](#structure)
    - [Load Data with `DataHandler`](#load-data-with-datahandler)
    - [Processed Data on Hugging Face 🤗](#processed-data-on-hugging-face-)
  - [✍️ How to Cite?](#️-how-to-cite)
  - [📝 To Do](#-to-do)
  - [📃 Licenses](#-licenses)

## 📘 Repository Overview

This repository contains the code used to generate the results presented in the paper. 
Along with the code, we provide the usage examples and results.

### What is included?

1. [datasets](datasets/) folder contains the datasets (e.g., statement) that we use. The subfolders contain the notebooks that we used to generate datasets, as well as generate the syntehtic entities and statements
2. [outputs/probes/prompt](outputs/probes/prompt) contains the scores for the *zero-shot prompting* (for every mode, dataset and instruction phrasing). These can be load using the `DataHandler` class. 
3. [outputs/probes/mean_diff](outputs/probes/mean_diff) contains an example of results for the *mean-difference* probe (`Llama-3-8b` model, `city_locations` dataset, based on the activations of the 7th decoder).
4. [configs](configs/) contains experiment configurations; `Hydra` uses these to run experiments.

### What is not included? 

TODO

### `sAwMIL` (Sparse Aware Multiple Instance Learning) Implementation

The code for the `sAwMIL` is partially based on the [garydoranjr/misvm](https://github.com/garydoranjr/misvm) repository (contains the `sbMIL` implementation for older versions of Python and [cvxopt](https://cvxopt.org/)). We adapt [MISVM](https://github.com/garydoranjr/misvm) code for `python=3.11.11` and `cvxopt=1.3.2`. The patched code for the `sAwMIL` is located in [probes/sawmil](probes/sawmil.py) script.

> [!NOTE]
> We plan to release a **standalone** package that implements `sAwMIL` and `sbMIL` using the [gurobipy](https://www.gurobi.com/) (closer to the end of June 2025).

## ⚡ Installation

Clone the repository:

```sh
git clone https://github.com/carlomarxdk/trilemma-of-truth.git
cd trilemma-of-truth
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Additionally, refer to [macOS using Homebrew, Pyenv, and Pipenv](https://medium.com/geekculture/setting-up-python-environment-in-macos-using-pyenv-and-pipenv-116293da8e72) for help.

Get HuggingFace **Access Tokens** for gated models:
> [!NOTE]
> If you intend to use LLMs, you need to update the `configs/model` files for some of the models. For example, in case of `base_gemma.yaml`, you need to update the `token` field with a valid Access Token, see [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). 
> Same applies to `base_llama`, `_llama-3-8b-med`, `_llama-3.1-8b-bio`.

## 📝 Usage & Examples

We use `Hydra` to run and manage our experiments. Refer to [Hydra Documentation](https://hydra.cc/docs/intro/) for help.

### Run the Scripts

#### 0. Return full error log in `Hydra`

In `Hydra` you can specify `HYDRA_FULL_ERROR=1` before each command. For example: 

```bash
HYDRA_FULL_ERROR=1 python run_zero_shot.py model=llama-3-8b 
```

#### 1. Collect Hidden Activations

To run experiments (e.g., train probes) on your machine, you need to collect hidden activations. The command below would collect hidden activations for every statement in the datasets, you only have to specify the name of the model, see [configs/activations.yamls](configs/activations.yaml) for more information on the attributes.

```bash
# To collect hidden activations for (every statement) specific model
python collect_activations.py model=llama-3-8b # see configs/activations.yaml for all the paramaters
```

After you collected the activations, you can load them using the code in [notebooks/load_and_split_dataset](notebooks/load_and_split_dataset.ipynb) notebook.

#### 2. Run zero-shot prompt (and collect scores)

You can collect the zero-shot prompting scores without having activations.

```bash
# Collect scores with the zero-shot prompting method (aka replies to multiple choice questions)
python run_zero_shot.py model=llama-3-8b variation=default batch_size=12 # see configs/probe_prompt.yaml for all the available paramaters
```

Note that we provide scores for every model in [outputs/probes/prompt](outputs/probes/prompt/) folder. We provide an example on how to load the scores from the zero-shot prompting in  [notebooks/load_and_split_dataset](notebooks/load_and_split_dataset.ipynb) notebook.

#### 3. Train *sAwMIL* probe
##### 3.1. One-vs-all

Note that you must collect activations before training this probe. Generally, you need to train three SVM probes: one with `task=0`, one with `task=1` and `task=2`, see [Task Specification](#task-specification).

```bash
# Train one-vs-all probe (an example without the hyperparameter search)
python run_training.py --config-name=probe_mil.yaml \
model=llama-3-8b datapack=city_locations probe=sawmil task=0 search=False 
```

##### 3. Multiclass
After you collect all the activations and train three one-vs-all `sAwMIL` probes, you can proceed with training the multiclass one.
The `run_mc_training.py` runs only with the `task=-1`.

```bash
python run_mc_training.py --config-name=probe_mil.yaml \
model=llama-3-8b datapack=city_locations probe=sawmil task=-1 search=False 
```

#### 4. Single Instance Probe

These probes use only the last token representation (instead of bags)
The **Single Instance Learning** probes use only representations of the last tokens (instead of the bags).

##### 4.1 Train *one-vs-all SVM* probe

Generally, you need to train three SVM probes: one with `task=0`, one with `task=1` and `task=2`, see [Task Specification](#task-specification).

```bash
python run_training.py --config-name=probe_sil.yaml \
model=llama-3-8b datapack=city_locations probe=svm task=1
```

#### 4.2 Train *multiclass SVM* probe
After you collect all the activations and train three one-vs-all `SVM` probes, you can proceed with training the multiclass one.
The `run_mc_training.py` runs only with the `task=-1`.

```bash
python run_mc_training.py --config-name=probe_mil.yaml \
model=llama-3-8b datapack=city_locations probe=svm task=-1
```

##### 4.3 Train the *mean-difference* probe

The mean-difference probe is trained to separate *true-vs-false*, thus, use `task=3` .

```bash
python run_training.py --config-name=probe_sil.yaml \
model=llama-3-8b datapack=city_locations probe=mean_diff task=3
```

### Task specification

You can train probe using different task configurations (see [misc/task.py](misc/task.py)). We have 5 tasks:

- **True-vs-All** (`task=0`): Separate *true* instances from all others (*false* and *neither*-valued cases);
- **False-vs-All** (`task=1`): Separate *false* instances from all others (*true* and *neither* cases);
- **Neither-vs-All** (`task=2`): Separate *neither* instances from all others (*true* and *false* cases);
- **True-vs-False** (`task=3`): Separate *true* and *false* cases (the *neither* statements are filtered out);
- **Multiclass** (`task=-1`): Multiclass setup, where labels correspond to `0=true`, `1=false` and `2=neither`.

## 🗂️ Dataset

The dataset scripts and files are located in the `datasets/` folder. This includes everything from data generation to the final preprocessed splits used in our experiments.

### Structure

1. `datasets/generators/`: Jupyter notebooks for data preprocessing and generation, along with *intermediate* data.
2. `datasets/generators/synthetic/`: Contains synthetic object/name lists (`*_raw.txt`) and manually filtered name list (`*_checked.csv`).
3. `datasets/`: Final preprocessed CSV files used to assemble the following datasets:
   - City Locations: `["city_locations.csv", "city_locations_synthetic.csv"]`
   - Medical Indications: `["med_indications", "med_indications_synthetic"]`
   - Word Definitions: `["word_instances", "word_types", "word_synonyms", "word_types_synthetic", "word_instances_synthetic", "word_synonyms_synthetic"]`

These datasets are used across our scripts to train probes and evaluate results.


### Load Data with `DataHandler`

You can load and assemble datasets using the `DataHandler` class:

```python
from data_handler import DataHandler

dh = DataHandler(
    model='llama-3-8b',
    datasets=['city_locations', 'city_locations_synthetic'],
    activation_type='full', # load the representation of all the tokens in each statement (alternatively, you can use `last`)
    with_calibration=True,    # Include a calibration set
    load_scores=False # if you run a zero-shot prompting with `default`, 
    #`shuffled` or `tf` template -- it will append these scores to the data (if they are calculated) 
)

dh.assemble(
    test_size=0.25,
    calibration_size=0.25,
    seed=42,
    exclusive_split=True      # Ensures entities don’t appear in multiple splits 
    # `True` would make the train, test and calibartion splits approximately split according to your specifications
    # in this case, test size is going to be approximatelly 25% of all the samples. 
)
```

For more usage examples, see the [notebooks/](notebooks/) folder.

### Processed Data on Hugging Face 🤗

The  final preprocessed datasets - including standardized splits - are also available on [Hugging Face Datasets](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth). These are ideal if you want to skip local preprocessing and directly load ready-to-use datasets into your workflow. They follow the same structure and splitting scheme we use internally. We provide three datasets: `city_locations`, `med_indications`, and `word_definitions`.

> [!IMPORTANT]
> **Note I:** These Hugging Face -- hosted datasets are *not* used in our experiments.  
> 
> **Note II**: All experiments in this repository (e.g., `collect_activations.py`, probe evaluations) rely on the `DataHandler` class, which assembles the datasets locally from the `datasets/` folder.
> 
> **Note III:** The calibration split is labeled as `validation`, following Hugging Face naming conventions (`train`, `validation`, `test`).

**How to use HF?** First, install the 🤗 Datasets and `pandas` libraries:

```bash
pip install datasets pandas
```

Then load the data with the `datasets` package. The dataset identifier is `carlomarxx/trilemma-of-truth`.

```python
from datasets import load_dataset

# 1. Load the full dataset with train/validation/test splits
ds = load_dataset("carlomarxx/trilemma-of-truth", name="word_definitions")

# Convert to pandas
df = ds["train"].to_pandas()

# Access the first example
print(ds["train"][0])

# 2. Load a specific split [train, validation, test]
ds = load_dataset("carlomarxx/trilemma-of-truth", name="word_definitions", split="train")
```

## ✍️ How to Cite?

```bibtex
@inproceedings{savcisens2024trilemma,
      title={The Trilemma of Truth in Large Language Models},
      author={Savcisens, Germans and Eliassi‐Rad, Tina},
      booktitle={arXiv preprint arXiv:2506.23921},
      year={2025}
    }
```

## 📝 To Do

> [!WARNING]
> We have refactored the code to improve readability. Please let us know if something does not work.

- [x] Check `run_zero_shot.py`
- [x] Check `collect_activations.py`
- [x] Check `run_training.py` for SIL probes (SVM and Mean Difference)
- [x] Check `run_training.py` for `sAwMIL`
- [x] Add the multiclass SIL and MIL script
- [ ] Check the multiclass SIL (SVM)
- [ ] Check the multiclass MIL (`sAwMIL`)
- [ ] Upload `llama-3-8b` activations for the `city_locations` dataset
- [ ] Check the script for interventions
- [ ] Check the script for the cross-dataset generalization
- [ ] Add scripts/notebooks for plot generation
- [x] Add examples: data loading 
- [x] Describe the contents of the repository

## 📃 Licenses

**Contacts**:

- [Germans Savcisens](https://savcisens.com/) (@carlomarxdk)
- [Tina Eliassi-Rad](https://eliassi.org/) (@eliassi)

> [!IMPORTANT]
> This **code** is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
> The **data** is licensed under the [Creative Commons Attribution 4.0 (CC BY 4.0)](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-4.0.md).
