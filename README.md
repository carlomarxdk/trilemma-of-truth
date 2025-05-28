# Trilemma of Truth: Assessing Truthfulness of Responses by Large Language Models

## Overview

**Trilemma of Truth** is 

This repository aims to:
- 


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Installation

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

## Usage

How you use this project 

```sh
# For running scripts or applications
python main.py
```

## Dataset

You can find the full preprocessed dataset (with all splits) hosted on [Hugging Face 🤗 Datasets](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth).  
We provide three configurations: `city_locations`, `med_indications`, and `word_definitions`.
> **Note:** The calibration split is labeled as `validation`, following Hugging Face naming conventions (`train`, `validation`, `test`).

First, install the 🤗 Datasets library:

```bash
pip install datasets
```


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

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

---