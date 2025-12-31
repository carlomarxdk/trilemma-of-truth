"""Utility functions for Hydra configuration and model preparation.

This module provides helper functions for loading data, preparing models,
and managing Hydra configurations for experiments.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import polars as pl
import torch
from hydra import compose, initialize
from nnsight import LanguageModel
from omegaconf import OmegaConf
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import matthews_corrcoef as mcc
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_handler import DataHandler

log = logging.getLogger("utils")


class NpEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""

    def default(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



def drop_rows_with_tail_keep(
    arr, num_rows_to_keep: int, last_rows_to_keep: int = 2, random_seed: int = 42
):
    """Randomly select rows to keep while preserving the last N rows.

    Args:
        arr: 2D NumPy array.
        num_rows_to_keep: Total number of rows to keep. Must be >= 2.
        last_rows_to_keep: Number of rows from the end to always keep.
            Defaults to 2.
        random_seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        2D array with specified rows retained, with last rows always included.
    """
    # Separate the last two rows
    last_two_rows = arr[-last_rows_to_keep:]

    # Determine rows to sample (excluding the last two)
    remaining_rows = arr[:-last_rows_to_keep]
    # Subtract the last two rows from the total

    # Randomly sample the specified number of rows
    np.random.seed(random_seed)
    indices_to_keep = np.random.choice(
        len(remaining_rows), size=num_rows_to_keep - last_rows_to_keep, replace=False
    )
    sampled_rows = remaining_rows[indices_to_keep]

    # Combine sampled rows with the last two rows
    final_array = np.vstack([sampled_rows, last_two_rows])

    return final_array


def bootstrap_3_ci(
    metric_func, arg_1, arg_2, arg_3, n_bootstraps=1000, alpha=0.05, random_state=None
):
    """Calculate bootstrap confidence intervals for a 3-argument metric.

    Args:
        metric_func: A metric function accepting 3 arguments.
        arg_1: First argument for the metric function.
        arg_2: Second argument for the metric function.
        arg_3: Third argument for the metric function.
        n_bootstraps: Number of bootstrap samples. Defaults to 1000.
        alpha: Significance level for confidence intervals. Defaults to 0.05.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Tuple of (metric_value, ci_lower, ci_upper).
    """

    if random_state is not None:
        np.random.seed(random_state)

    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # Generate a bootstrap sample
        indices = np.random.randint(0, len(arg_1), len(arg_1))
        arg_1_bootstrap = arg_1[indices]
        arg_2_bootstrap = arg_2[indices]
        arg_3_bootstrap = arg_3[indices]

        # Calculate the metric for the bootstrap sample
        score = metric_func(arg_1_bootstrap, arg_2_bootstrap, arg_3_bootstrap)
        bootstrapped_scores.append(score)

    # Calculate the confidence interval
    ci_lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

    return metric_func(arg_1, arg_2, arg_3), ci_lower, ci_upper


def weighed_mcc(y_true, y_pred, coverage):
    """Calculate MCC weighted by coverage for abstention cases.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        coverage: Acceptance rate (fraction of non-abstained predictions).

    Returns:
        Weighted MCC score.
    """
    return mcc(y_true, y_pred) * coverage


def weighed_ami(y_true, y_pred, coverage):
    """Calculate AMI weighted by coverage for abstention cases.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        coverage: Acceptance rate (fraction of non-abstained predictions).

    Returns:
        Weighted AMI score.
    """
    return ami(y_true, y_pred) * coverage


def load_data(cfg):
    """Load and prepare data from configuration.

    Args:
        cfg: Configuration object with datapack and model settings.

    Returns:
        Configured DataHandler instance with loaded and split data.
    """
    if cfg.datapack.cal_size > 0:
        with_calibration = True
    else:
        with_calibration = False
    dh = DataHandler(
        model=cfg.model["name"],
        datasets=cfg.datapack["datasets"],
        dataset_path=cfg.setup["dataset_path"],
        activations_path=cfg.setup["activations_path"],
        output_path=cfg.setup["output_path"],
        activation_type=cfg.agg,
        with_calibration=with_calibration,
        load_scores=cfg.datapack["load_scores"],
    )
    dh.assemble(
        test_size=cfg.datapack["test_size"],
        calibration_size=cfg.datapack["cal_size"],
        seed=cfg.datapack["random_seed"],
        exclusive_split=cfg.datapack["exclusive_split"],
    )
    return dh


def load_data_with_test(cfg):
    """Load test data from configuration.

    Args:
        cfg: Configuration object with datapack_test and model settings.

    Returns:
        Configured DataHandler instance with loaded test data.
    """
    if cfg.datapack_test.cal_size > 0:
        with_calibration = True
    else:
        with_calibration = False
    log.warning(f"Test dataset path: {cfg.setup['dataset_path']}")
    dh = DataHandler(
        model=cfg.model["name"],
        datasets=cfg.datapack_test["datasets"],
        dataset_path=cfg.setup["dataset_path"],
        activations_path=cfg.setup["activations_path"],
        output_path=cfg.setup["output_path"],
        activation_type=cfg.agg,
        with_calibration=with_calibration,
        load_scores=cfg.datapack_test["load_scores"],
    )
    dh.assemble(
        test_size=cfg.datapack_test["test_size"],
        calibration_size=cfg.datapack_test["cal_size"],
        seed=cfg.datapack_test["random_seed"],
        exclusive_split=cfg.datapack_test["exclusive_split"],
    )
    return dh


def flatten(xss) -> list:
    """Flatten a list of lists into a single list.

    Args:
        xss: List of lists.

    Returns:
        Flattened list.
    """
    return [x for xs in xss for x in xs]


def load_statements(dataset: list) -> list[str]:
    """Load statements from a dataset CSV file.

    Args:
        dataset: Dataset name.

    Returns:
        List of statement strings.
    """
    return flatten(
        pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()
    )


def load_statements_with_targets(dataset: list) -> list[str]:
    """Load statements and correct labels from a dataset CSV file.

    Args:
        dataset: Dataset name.

    Returns:
        Tuple of (statements list, correct labels list).
    """
    return flatten(
        pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()
    ), flatten(pl.read_csv(f"datasets/{dataset}.csv").select("correct").to_numpy())


def get_device():
    """Detect and return the best available device for computation.

    Returns:
        torch.device instance (CUDA, MPS, or CPU).
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


def clear_device_cache(device: torch.device):
    """Clear device memory cache if supported.

    Args:
        device: torch.device instance to clear cache for.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # As of now, explicit cache clearing for MPS may not be necessary or available.
        # This check ensures that if torch.mps.empty_cache() is added in the future,
        # your code will use it; otherwise, it will safely do nothing.
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    # No need to clear cache for CPU


def return_layers(model, cfg):
    """Get list of layer indices from model configuration.

    Args:
        model: Model instance.
        cfg: Configuration object with model settings.

    Returns:
        List of layer indices.
    """
    return list(
        range(
            len(model.get_submodule(cfg.model.module).get_submodule(cfg.model.encoders))
        )
    )


def prepare_hf_model(cfg, device=None):
    """Prepare HuggingFace model and tokenizer for experiments.

    Args:
        cfg: Configuration object with model settings.
        device: Device to load model on. If None, uses cfg.device.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ValueError: If dtype is not 'float16' or 'float32'.
    """
    if device is None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(device)
    if cfg.model["dtype"] == "float16":
        _dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model["model"],
            token=cfg.model["token"],
            torch_dtype=_dtype,
            attn_implementation="eager",
            device_map={"": device},
        )
    elif cfg.model["dtype"] == "float32":
        _dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model["model"],
            token=cfg.model["token"],
            torch_dtype=_dtype,
            device_map={"": device},
        )
    else:
        raise ValueError("dtype must be either 'bfloat16' or 'float32'.")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model

    return model, tokenizer


def prepare_hf_tokenizer(cfg, device=None):
    """Prepare HuggingFace tokenizer without loading model.

    Args:
        cfg: Configuration object with model settings.
        device: Device specification (unused but kept for API consistency).

    Returns:
        Tuple of (None, tokenizer).
    """
    if device is None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(device)
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model

    return None, tokenizer


def prepare_nnsight(cfg):
    """Prepare NNSight LanguageModel and tokenizer for experiments.

    Args:
        cfg: Configuration object with model settings.

    Returns:
        Tuple of (model, tokenizer).
    """
    device = torch.device(cfg.device)
    # if cfg.model["dtype"] == "float16":
    #     _dtype = torch.float16
    #     model = AutoModelForCausalLM.from_pretrained(
    #         pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"],
    #         torch_dtype=_dtype,  attn_implementation="eager", device_map={"": device})
    # elif cfg.model["dtype"] == "float32":
    #     _dtype = torch.float32
    #     model = AutoModelForCausalLM.from_pretrained(
    #         pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"], torch_dtype="auto", device_map={"": device})
    # Load model and tokenizer
    if cfg.model["dtype"] == "float16" or cfg.model["dtype"] == "float32":
        _dtype = torch.float16
    model = LanguageModel(
        cfg.model["model"],
        token=cfg.model["token"],
        device_map={"": device},
        dispatch=True,
        torch_dtype=_dtype,
        offload_folder="offload",  # local dir to spill weights to
        offload_state_dict=True,
    )

    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.tokenizer.padding_side = "left"
    model.eval()
    model.requires_grad_(False)
    tokenizer = model.tokenizer
    return model, tokenizer


def return_label(data):
    """Extract labels from dataframe.

    Args:
        data: Pandas DataFrame with label columns.

    Returns:
        Tuple of (correct, real, negated) label arrays.
    """
    correct, real, negated = (
        data["correct"].values,
        data["real_object"].values,
        data["negation"].values,
    )
    return correct, real, negated


def normalize(X):
    """Normalize vectors to unit length.

    Args:
        X: 1D or 2D array to normalize.

    Returns:
        Normalized array.
    """
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def load_hydra_experiment(model, datapack, probe, config_name):
    """Load Hydra experiment configuration.

    Args:
        model: Model name override.
        datapack: Datapack name override.
        probe: Probe name override.
        config_name: Base configuration name.

    Returns:
        Configuration dictionary.
    """
    with initialize(version_base="1.1", config_path="configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[f"model={model}", f"datapack={datapack}", f"probe={probe}"],
        )
    return OmegaConf.to_container(cfg, resolve=True)
