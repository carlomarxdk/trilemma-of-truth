import torch
import re
import numpy as np
import yaml
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from data_handler import DataHandler
import polars as pl
import os
import logging


log = logging.getLogger(__name__)

# Misc Functions


def should_process_layer(layer_id, cfg):
    """Determine if a given layer should be processed."""
    layer_range = np.quantile(
        cfg.model['layers'], cfg.layer_range, method="closest_observation")
    return (layer_range[0] <= layer_id <= layer_range[1])


# Other misc function


def return_layers(model, config):
    """
    Return the layers to save activations from (as a list)
    """
    return list(range(len(model.get_submodule(config["module"]).get_submodule(config["encoders"]))))


def load_data(args, datasets):
    """
    Load the data from the config file
    """
    dh = DataHandler(args.experiment, datasets,
                     args.agg, with_calibration=True, load_scores=True,
                     )
    dh.assemble(test_size=args.test_size, calibration_size=args.cal_size,
                seed=args.random_seed, exclusive_split=True)
    return dh


def save_checkpoint(scores, dataset, experiment, batch_index, suffix=""):
    """Save checkpoint with scores and the current batch index."""
    save_dir = f"outputs/checkpoints/{experiment}/{dataset}/{suffix}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'scores': scores, 'batch_index': batch_index},
               os.path.join(save_dir, "checkpoint.pt"))


def load_checkpoint(dataset, experiment):
    """Load checkpoint if it exists."""
    checkpoint_path = f"outputs/checkpoints/{experiment}/{dataset}/checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        return checkpoint['scores'], checkpoint['batch_index']
    return None, 0


def load_statements_from_multiple_data(dataset: List) -> List[str]:
    """
    Load the dataset from the mulitple files
    """
    data = list()
    for dataset in dataset:
        data.append(pl.read_csv(f"datasets/{dataset}.csv"))
    return flatten(pl.concat(data).select("statement").to_numpy())


def load_statements(dataset: List) -> List[str]:
    """
    Load the dataset from the file
    """
    return flatten(pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()), flatten(pl.read_csv(f"datasets/{dataset}.csv").select("correct").to_numpy())


def prepare_hf_model(model_config, device):
    """
    Prepare the HuggingFace model and tokenizerz for the experiment
    """
    if model_config["dtype"] == "float16":
        _dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config["model"], token=model_config["token"], torch_dtype=_dtype,  attn_implementation="eager", device_map={"": device})
    elif model_config["dtype"] == "float32":
        _dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config["model"], token=model_config["token"], torch_dtype="auto", device_map={"": device})
    else:
        raise ValueError("dtype must be either 'bfloat16' or 'float32'.")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_config["model"], token=model_config["token"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model

    return model, tokenizer


def load_config(key: str) -> List[int]:
    """
    Load the configuration from the file and return the values for the key
    """
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)[key]


def flatten(xss) -> List:
    """
    Flatten a list of lists
    """
    return [x for xs in xss for x in xs]


def translate_embedding(X, direction, target_coord: float, absolute: bool = True):
    """
    Translate the 'X' embedding in the direction  by 'new_coord' units (if abslute is True)
    Args:
    - X: The embedding to translate
    - direction: The direction to translate the embedding
    - new_coord: The new coordinate to move to
    Returns:
    - The translated embedding
    """
    # move in respect to the current coordinate
    curr_coord = torch.einsum("bsh, h -> bs", X, direction)
    delta = target_coord - curr_coord
    proj = torch.einsum("h, bs -> bsh", direction, delta)
    return X + proj


def amplify_embedding(X, direction, factor: float = 1.0):
    """
    Amplify the 'X' embedding in the direction by 'factor' units
    Args:
    - X: The embedding to amplify
    - direction: The direction to amplify the embedding
    - factor: The factor to amplify the embedding
    Returns:
    - The amplified embedding
    """
    proj = torch.einsum("h, bsh -> bsh", direction, X)
    proj = torch.sign(proj) * direction
    return X + factor * proj


def bootstrap_ci(metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05, random_state=None):
    """
    Calculate bootstrapped confidence intervals for a given scikit-learn metric.

    Parameters:
    - metric_func: A scikit-learn metric function (e.g., accuracy_score, precision_score).
    - y_true: Array-like of shape (n_samples,), Ground truth (correct) labels.
    - y_pred: Array-like of shape (n_samples,), Predicted labels, as returned by a classifier.
    - n_bootstraps: Number of bootstrap samples to generate (default: 1000).
    - alpha: Significance level for the confidence intervals (default: 0.05).
    - random_state: Random seed for reproducibility (default: None).

    Returns:
    - ci_lower: Lower bound of the confidence interval.
    - ci_upper: Upper bound of the confidence interval.
    """

    if random_state is not None:
        np.random.seed(random_state)

    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # Generate a bootstrap sample
        indices = np.random.randint(0, len(y_true), len(y_true))
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]

        # Calculate the metric for the bootstrap sample
        score = metric_func(y_true_bootstrap, y_pred_bootstrap)
        bootstrapped_scores.append(score)

    # Calculate the confidence interval
    ci_lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

    return metric_func(y_true, y_pred), ci_lower, ci_upper


def return_label(data):
    """
    Return labels from the dataframe
    """
    correct, real, fake, fictional, negated = data["correct"].values, data["real_object"].values, data[
        "fake_object"].values, data["fictional_object"].values, data["negation"].values
    combined = np.select(
        [
            (correct == 0) & (real == 1) & (fake == 0) & (fictional == 0),
            (correct == 1) & (real == 1) & (fake == 0) & (fictional == 0),
            (fake == 1) & (fictional == 0) & (real == 0),
            (correct == 0) & (fake == 1) | (correct == 0) & (fictional == 1),
            (correct == 1) & (fake == 1) | (correct == 1) & (fictional == 1),
        ],
        [0, 1, 4, 2, 3], default=4
    )
    return correct, real, fake, combined, negated, fictional


def normalize(X):
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def create_label_map(unique_labels, colors):
    """
    Create a label map for plots with unique labels and colors
    """
    return {label: color for label, color in zip(unique_labels, colors)}


def get_device():
    """
    Get the device to use for computation
    """
    if torch.cuda.is_available():
        # If CUDA is available, select the first CUDA device
        device = torch.device("cuda:0")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    # Check for MPS availability on supported macOS devices (requires PyTorch 1.12 or newer)
    elif torch.backends.mps.is_available():
        # If MPS is available, use MPS device
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        # Fallback to CPU if neither CUDA nor MPS is available
        device = torch.device("cpu")
        print("Using CPU")
    return device

def extract_number(filename):
    """
    Extract the number from a filename
    """
    match = re.search(r'layer_(\d+)', filename)
    return int(match.group(1)) if match else 0