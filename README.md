# Trilemma of Truth

## Overview

**Trilemma of Truth** is 


## Table of Contents

- [Trilemma of Truth](#trilemma-of-truth)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Dataset](#dataset)
  - [Citation](#citation)
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

You can find the full preprocessed datasets (with all splits) hosted on [Hugging Face 🤗 Datasets](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth).  
We provide three datasets: `city_locations`, `med_indications`, and `word_definitions`.
> **Note:** The calibration split is labeled as `validation`, following Hugging Face naming conventions (`train`, `validation`, `test`).

First, install the 🤗 Datasets and `pandas` libraries:

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

## Citation


## License

This **code** is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
The **data** is licensed under the [Creative Commons Attribution 4.0 (CC BY 4.0)](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/cc-by-4.0.md).

---