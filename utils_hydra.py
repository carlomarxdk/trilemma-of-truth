from hydra import compose, initialize
from omegaconf import OmegaConf

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_mutual_info_score as ami,
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
    adjusted_rand_score as ari,
)
from typing import List
import polars as pl
from data_handler import DataHandler
import numpy as np
from nnsight import LanguageModel
import logging
import json
log = logging.getLogger('utils')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def drop_rows_with_tail_keep(arr, num_rows_to_keep: int, last_rows_to_keep: int = 2,  random_seed: int = 42):
    """
    Randomly selects a specific number of rows to keep from a 2D numpy array,
    ensuring the last two rows are always retained.

    Parameters:
    - arr (numpy.ndarray): 2D NumPy array.
    - num_rows_to_keep (int): Total number of rows to keep, including the last two rows.
                              Must be >= 2.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - numpy.ndarray: 2D array with specified rows retained, keeping the last two rows intact.
    """
    # Separate the last two rows
    last_two_rows = arr[-last_rows_to_keep:]

    # Determine rows to sample (excluding the last two)
    remaining_rows = arr[:-last_rows_to_keep]
    # Subtract the last two rows from the total

    # Randomly sample the specified number of rows
    np.random.seed(random_seed)
    indices_to_keep = np.random.choice(
        len(remaining_rows), size=num_rows_to_keep-last_rows_to_keep, replace=False)
    sampled_rows = remaining_rows[indices_to_keep]

    # Combine sampled rows with the last two rows
    final_array = np.vstack([sampled_rows, last_two_rows])

    return final_array


def bootstrap_3_ci(metric_func, arg_1, arg_2, arg_3, n_bootstraps=1000, alpha=0.05, random_state=None):
    """
    Calculate bootstrapped confidence intervals for a given scikit-learn metric that can accept multiple arguments.

    Parameters:
    - metric_func: A scikit-learn metric function (e.g., precision_score, f1_score).
    - n_bootstraps: Number of bootstrap samples to generate (default: 1000).
    - alpha: Significance level for the confidence intervals (default: 0.05).
    - random_state: Random seed for reproducibility (default: None).
    - *args: Positional arguments to be passed to the metric function. For example, y_true, y_pred, etc.
    - **kwargs: Keyword arguments to be passed to the metric function.

    Returns:
    - original_score: The metric value calculated on the original data.
    - ci_lower: Lower bound of the confidence interval.
    - ci_upper: Upper bound of the confidence interval.
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


def safe_bootstrap(metric, **kwargs):
    try:
        return bootstrap_ci(metric, **kwargs)
    except Exception:
        return (0, -1e-6, 1e-6)


def weighed_mcc(y_true, y_pred, coverage):
    '''MCC adapted for cases with Abstention.
    Args:
    y_true: np.array, true labels
    y_pred: np.array, predicted labels
    coverage: float, acceptance rate (fraction of predictions that are not abstained)
    Returns:
    float, weighed MCC
    '''
    return mcc(y_true, y_pred) * coverage


def weighed_ami(y_true, y_pred, coverage):
    '''AMI adapted for cases with Abstention
    Args:
    y_true: np.array, true labels
    y_pred: np.array, predicted labels
    coverage: float, acceptance rate (fraction of predictions that are not abstained)
    Returns:
    float, weighed AMI
    '''
    return ami(y_true, y_pred) * coverage


def load_data(cfg):
    """
    Load the data from the config file
    """
    if cfg.datapack.cal_size > 0:
        with_calibration = True
    else:
        with_calibration = False
    dh = DataHandler(cfg.model["name"], cfg.datapack["datasets"],
                     cfg.agg, with_calibration=with_calibration, load_scores=cfg.datapack[
                         "load_scores"],
                     )
    dh.assemble(test_size=cfg.datapack["test_size"], calibration_size=cfg.datapack["cal_size"],
                seed=cfg.datapack["random_seed"], exclusive_split=cfg.datapack["exclusive_split"])
    return dh


def load_data_with_test(cfg):
    """
    Load the data (TEST) from the config file
    """
    if cfg.datapack_test.cal_size > 0:
        with_calibration = True
    else:
        with_calibration = False
    dh = DataHandler(cfg.model["name"], cfg.datapack_test["datasets"],
                     cfg.agg, with_calibration=with_calibration, load_scores='default',
                     )
    dh.assemble(test_size=cfg.datapack_test["test_size"], calibration_size=cfg.datapack_test["cal_size"],
                seed=cfg.datapack_test["random_seed"], exclusive_split=cfg.datapack_test["exclusive_split"])
    return dh


def return_layers(model, cfg):
    """
    Return the layers to save activations from (as a list)
    """
    return list(range(len(model.get_submodule(cfg.model["module"]).get_submodule(cfg.model["encoders"]))))


def flatten(xss) -> List:
    """
    Flatten a list of lists
    """
    return [x for xs in xss for x in xs]


def load_statements(dataset: List) -> List[str]:
    """
    Load the dataset from the file
    """
    return flatten(pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy())


def load_statements_with_targets(dataset: List) -> List[str]:
    """
    Load the dataset from the file
    """
    return flatten(pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()), flatten(pl.read_csv(f"datasets/{dataset}.csv").select("correct").to_numpy())


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


def clear_device_cache(device: torch.device):
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
    """
    Return the layers to save activations from (as a list)
    """
    return list(range(len(model.get_submodule(cfg.model.module).get_submodule(cfg.model.encoders))))


def prepare_hf_model(cfg, device=None):
    """
    Prepare the HuggingFace model and tokenizerz for the experiment
    """
    if device is None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(device)
    if cfg.model["dtype"] == "float16":
        _dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"],
            torch_dtype=_dtype,  attn_implementation="eager", device_map={"": device})
    elif cfg.model["dtype"] == "float32":
        _dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"], torch_dtype="auto", device_map={"": device})
    else:
        raise ValueError("dtype must be either 'bfloat16' or 'float32'.")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model

    return model, tokenizer


def prepare_hf_tokenizer(cfg, device=None):
    """
    Prepare the HuggingFace model and tokenizerz for the experiment
    """
    if device is None:
        device = torch.device(cfg.device)
    else:
        device = torch.device(device)
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.model["model"], token=cfg.model["token"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model

    return None, tokenizer


def prepare_nnsight(cfg):
    """
    Prepare the NNSight model and tokenizer for the experiment
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
    if cfg.model["dtype"] == "float16":
        _dtype = torch.float16
    elif cfg.model["dtype"] == "float32":
        _dtype = torch.float16
    model = LanguageModel(
        cfg.model["model"], token=cfg.model["token"], device_map={"": device}, dispatch=True, torch_dtype=_dtype,
        offload_folder="offload",    # local dir to spill weights to
        offload_state_dict=True)

    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    model.tokenizer.padding_side = "left"
    model.eval()
    model.requires_grad_(False)
    tokenizer = model.tokenizer
    return model, tokenizer


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


def load_hydra_experiment(model, datapack, probe, config_name):
    with initialize(version_base="1.1", config_path="configs"):
        cfg = compose(config_name=config_name, overrides=[
                      f'model={model}', f'datapack={datapack}', f'probe={probe}'])
    return OmegaConf.to_container(cfg, resolve=True)
