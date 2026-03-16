# Rethinking Veracity in Large Language Models:  Three Flawed Assumptions, One New Probe


**This repository** is the codebase for evaluating factual reasoning in large language models.  
Here you’ll find everything needed to  

1. Generate and inspect our  datasets (city locations, drug indications, word definitions),  
2. Run zero-shot prompts,  
3. Train and evaluate a suite of probes.

**Abstract:** The public often attributes human-like qualities to large language models (LLMs), assuming that they ``know'' certain things. In reality, LLMs encode information retained during training as internal probabilistic knowledge. This study examines existing methods for probing the veracity of that knowledge and identifies three flawed underlying assumptions. To address these flaws, we introduce sAwMIL (Sparse-Aware Multiple-Instance Learning), a multiclass probing framework that combines multiple-instance learning with conformal prediction. sAwMIL leverages LLMs' internal representations to classify statements as *true*, *false*, or *neither*. We evaluate sAwMIL across 16 open-source LLMs, including default and chat-based variants, using three new curated datasets. Our results show that (1) common probing methods fail to provide a reliable and transferable veracity direction and, in some settings, perform worse than zero-shot prompting; (2) truth and falsehood are not encoded symmetrically; and (3) LLMs encode a third type of signal that is distinct from both true and false.

---

## Table of Contents

- [Rethinking Veracity in Large Language Models:  Three Flawed Assumptions, One New Probe](#rethinking-veracity-in-large-language-models--three-flawed-assumptions-one-new-probe)
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
        - [3.2 Multiclass](#32-multiclass)
      - [4. Single Instance Probe](#4-single-instance-probe)
        - [4.1 Train *one-vs-all SVM* probe](#41-train-one-vs-all-svm-probe)
        - [4.2 Train *multiclass SVM* probe](#42-train-multiclass-svm-probe)
        - [4.3 Train binary SIL baselines](#43-train-binary-sil-baselines)
      - [5. Extra](#5-extra)
        - [5.1 Generalization Performance](#51-generalization-performance)
        - [5.2 Interventions](#52-interventions)
    - [Task specification](#task-specification)
  - [🗂️ Dataset](#️-dataset)
    - [Structure](#structure)
    - [Load Data with `DataHandler`](#load-data-with-datahandler)
  - [📃 Licenses](#-licenses)

## 📘 Repository Overview

This repository contains the code used to generate the results presented in the paper. 
Along with the code, we provide the usage examples and results.

### What is included?

1. [datasets](datasets/) folder contains the datasets (e.g., statement) that we use. The subfolders contain the notebooks that we used to generate datasets, as well as generate the syntehtic entities and statements
2. [outputs/probes/prompt](outputs/probes/prompt) contains the scores for the *zero-shot prompting* (for every mode, dataset and instruction phrasing). These can be load using the `DataHandler` class.
3. [outputs/probes/mean_diff](outputs/probes/mean_diff) contains an example of results for the *mean-difference* probe (`Llama-3-8b` model, `city_locations` dataset, based on the activations of the 7th decoder).
4. [configs](configs/) contains experiment configurations; `Hydra` uses these to run experiments.
5. [outputs/activations/llama-3-8b](outputs/activations/llama-3-8b) contains activations for the `city_locations` dataset (13th decoder).
6. [outputs/probes](outputs/probes) contains example of coefficients and statistics for the probes trained on the `llama-3-8b` activations (`city_locations` dataset).


### What is not included?

1. Activations and the coefficients for the trained probes (we only include activations for the 13th decoder of the `llama-3-8b` model and `city_locations` dataset)
2. Full generated artifacts for every model/configuration run (for example, complete figure/table sets and all intermediate outputs).

Plot generation code is included in `analysis/make_plots.py`, `make_plots.ipynb`, and `make_tables.ipynb`.

### `sAwMIL` (Sparse Aware Multiple Instance Learning) Implementation

The code for the `sAwMIL` is partially based on the [garydoranjr/misvm](https://github.com/garydoranjr/misvm) repository (contains the `sbMIL` implementation for older versions of Python and [cvxopt](https://cvxopt.org/)). We adapt [MISVM](https://github.com/garydoranjr/misvm) code for `python=3.12` and `cvxopt=1.3.2`. The patched code for the `sAwMIL` is located in [probes/sawmil](probes/sawmil.py) script.



## ⚡ Installation

Install dependencies:

```sh
pip install -r requirements.txt
```

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

To run experiments (e.g., train probes) on your machine, you need to collect hidden activations. The command below would collect hidden activations for every statement in the datasets, you only have to specify the name of the model, see [configs/activations.yaml](configs/activations.yaml) for more information on the attributes.

```bash
# To collect hidden activations for (every statement) specific model
python collect_activations.py model=llama-3-8b 
# see configs/activations.yaml for all the paramaters
```


#### 2. Run zero-shot prompt (and collect scores)

You can collect the zero-shot prompting scores without having activations.

```bash
# Collect scores with the zero-shot prompting method (aka replies to multiple choice questions)
python run_zero_shot.py \
      model=llama-3-8b \
      variation=default \
      batch_size=12 
# see configs/probe_zeroshot.yaml for all the available paramaters
```

Note that we provide scores for every model in [outputs/probes/prompt](outputs/probes/prompt/) folder. We provide an example on how to load the scores from the zero-shot prompting in  [notebooks/load_and_split_dataset](notebooks/load_and_split_dataset.ipynb) notebook.

#### 3. Train *sAwMIL* probe

##### 3.1. One-vs-all

Note that you must collect activations before training this probe. Generally, you need to train three SVM probes: one with `task=0`, one with `task=1` and `task=2`, see [Task Specification](#task-specification).

```bash
# Train one-vs-all probe (an example without the hyperparameter search)
python run_training.py \
      model=llama-3-8b \
      datapack=city_locations \
      probe=sawmil \
      task=0 \
      search=True # True to activate the parameter search
```

##### 3.2 Multiclass

After you collect all the activations and train three one-vs-all `sAwMIL` probes, you can proceed with training the multiclass one.
The `run_training.py` runs only with the `task=-1`.

```bash
python run_training.py \
      model=llama-3-8b \
      datapack=city_locations \
      probe=sawmil \
      task=-1 \
      search=True
```


#### 4. Single Instance Probe

These probes use only the last token representation (instead of bags)
The **Single Instance Learning** probes use only representations of the last tokens (instead of the bags).

##### 4.1 Train *one-vs-all SVM* probe

Generally, you need to train three SVM probes: one with `task=0`, one with `task=1` and `task=2`, see [Task Specification](#task-specification).

```bash
python run_training.py \
      model=llama-3-8b \
      datapack=city_locations \
      probe=svm \
      task=1
```

##### 4.2 Train *multiclass SVM* probe

After you collect all the activations and train three one-vs-all `SVM` probes, you can proceed with training the multiclass one.
The `run_training.py` runs only with the `task=-1`.

```bash
python run_training.py \
      model=llama-3-8b \
      datapack=city_locations \
      probe=svm \
      run_debugging=False \ # True would run the training only on the 13th layer
      task=-1
```

##### 4.3 Train binary SIL baselines

The SIL binary baselines are trained to separate *true-vs-false*, thus, use `task=3`, these include `mean_diff`, `spca` and `ttpd`.

```bash
python run_training.py  \
      model=llama-3-8b \
      datapack=city_locations \
      probe=mean_diff \
      task=3
```

#### 5. Extra

##### 5.1 Generalization Performance

To check the performance of the probe on another dataset you can run `run_generalization.py`. It will load the probe trained on `datapack` and use the test split of the `datapack@datapack_test`.

```bash
python run_generalization.py \
      model=llama-3-8b \
      datapack=city_locations \
      datapack@datapack_test=med_indications \
      probe=sawmil \
      search=True \
      task=-1 # Generalization of the multiclass sawmil
```

Use the task nr that you used to train the probe. For example for `mean_diff` (or any other binary SIL), it is `task=3`.

##### 5.2 Interventions

The code for interventions is located in `run_intervention.py`.

```bash
python run_intervention.py \
      model=llama-3-8b \
      datapack=city_locations \
      task=0
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
## 📃 Licenses

> [!IMPORTANT]
> This **code** is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

1. This is research software. While we strive for correctness and reproducibility, please verify results for your specific use case.
2. GitHub Copilot and Claude contributed to code annotations, docstrings, and formatting. All algorithmic logic, methodological design, and scientific claims were developed and reviewed by the authors.