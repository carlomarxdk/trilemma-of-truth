from __future__ import annotations
import json
import tempfile
import pickle
from typing import Dict
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import numpy as np
import joblib
import re
from sklearn.metrics import (
    recall_score as recall,
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
    adjusted_mutual_info_score as ami,
    adjusted_rand_score as ari,
)
from scipy.stats import energy_distance
import warnings


log = logging.getLogger(__name__)

# 
def available_layers(output_dir: Path | str) -> list[int]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifests"  

    layers = set()
    for f in manifest_path.glob("manifest_*.json"):
        m = re.search(r"manifest_(\d+)", f.name)
        if m: layers.add(int(m.group(1)))
    return sorted(layers)


# Misc Functions

def bootstrap_ci(metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05, random_state=None) -> Tuple[float, float, float]:
    """
    Calculate bootstrapped confidence intervals for a given scikit-learn metric.

    Args:
        metric_func: A scikit-learn metric function (e.g., accuracy_score, precision_score).
        y_true: Array-like of shape (n_samples,), Ground truth (correct) labels.
        y_pred: Array-like of shape (n_samples,), Predicted labels, as returned by a classifier.
        n_bootstraps: Number of bootstrap samples to generate (default: 1000).
        alpha: Significance level for the confidence intervals (default: 0.05).
        random_state: Random seed for reproducibility (default: None).
    Returns:
        mean: Mean value of the metric.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
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

def safe_bootstrap(metric, **kwargs) -> Tuple[float, float, float]:
    '''
    Safely compute bootstrap confidence intervals for a given metric.
    If an error occurs, return a default value of (0, -1e-6, 1e-6).
    Args:
        metric: Metric function to compute (e.g., accuracy_score, precision_score).
        **kwargs: Arguments to pass to the bootstrap_ci function.
    Returns:
        A tuple containing the mean, lower bound, and upper bound of the confidence interval.
    '''
    try:
        return bootstrap_ci(metric, **kwargs)
    except Exception:
        return (0, -1e-6, 1e-6)


def log_metric_binary(preds: np.ndarray, scores: np.ndarray, y_true: np.ndarray, mask: np.ndarray, cfg) -> Dict[str, float]:
    """
    Log the metrics to the Weights and Biases dashboard with prefix and return as a dictionary without prefix.
    """
    # yhat = probs.round()
    is_binary = len(np.unique(y_true)) == 2
    if not is_binary:
        log.warning("Not a binary classification problem.")
    is_ok = (len(np.unique(preds)) > 0) & (len(np.unique(preds)) < 4)
    assert is_ok, "Only binary classification is supported (or binary with abstention class '-1')."

    a_mask = (preds != -1).flatten()
    preds = preds.flatten()
    scores = scores.flatten()
    a_rate = np.sum(a_mask[mask]) / len(a_mask[mask])

    def wmcc(y_true, y_pred): return mcc(y_true, y_pred) *\
        a_rate

    def wami(y_true, y_pred): return ami(y_true, y_pred) *\
        a_rate

    def wari(y_true, y_pred): return ari(y_true, y_pred) *\
        a_rate

    full_mask = a_mask & mask

    binary_kwargs = dict(
        y_true=y_true[full_mask],
        y_pred=preds[full_mask],
        n_bootstraps=cfg.eval_params["n_bootstraps"]
    )

    # Get the values for each metric using the helper.
    mcc_val = safe_bootstrap(mcc,  **binary_kwargs)
    ami_val = safe_bootstrap(ami,  **binary_kwargs)
    ari_val = safe_bootstrap(ari,  **binary_kwargs)
    recall_val = safe_bootstrap(recall, **binary_kwargs)
    if np.equal(a_mask.mean(), 1):
        wmcc_val = mcc_val
        wami_val = ami_val
        wari_val = ari_val
        wrecall_val = recall_val
    else:
        wmcc_val = safe_bootstrap(wmcc, **binary_kwargs)
        wami_val = safe_bootstrap(wami, **binary_kwargs)
        wari_val = safe_bootstrap(wari,  **binary_kwargs)
        wrecall_val = safe_bootstrap(recall, **binary_kwargs)
    try:
        probs = scores[full_mask]
        x_min = probs.min()
        x_max = probs.max()

        # Apply min-max scaling
        probs_scaled = (probs - x_min) / (x_max - x_min)
        targets = y_true[full_mask]
        energy_val = energy_distance(
            probs_scaled[targets == 0], probs_scaled[targets == 1])
    except Exception as e:
        log.warning(
            f"Error calculating energy distance: {e}. Setting to 1000.")
        energy_val = 1000

    try:
        mAP_val = mAP(y_true[full_mask],
                      scores[full_mask])

    except:
        try:
            mAP_val = mAP(y_true[full_mask],
                          np.zeros_like(scores[full_mask]))
        except:
            try:
                mAP_val = mAP(y_true[mask],
                              np.zeros_like(scores[mask]))
            except:
                mAP_val = 0

    metric_with_ci = {
        "mcc": mcc_val,
        "ami": ami_val,
        "ari": ari_val,
        "wmcc": wmcc_val,
        "wami": wami_val,
        "wari": wari_val,
        "map": mAP_val,
        "wmap": mAP_val * a_rate,
        "energy": energy_val,
        "wenergy": energy_val * a_rate,
        "acceptance_rate": a_rate,
        "recall": recall_val,
        "wrecall": wrecall_val,
        "n": y_true[full_mask].shape[0],
    }
    return metric_with_ci


def log_metric_multiclass(preds: np.ndarray, scores: np.ndarray, y_true: np.ndarray, cfg) -> Dict[str, float]:
    """
    Log the metrics to the Weights and Biases dashboard with prefix and return as a dictionary without prefix.
    Args:
        preds: Predicted labels, as returned by a classifier.
        scores: Predicted scores or probabilities for each class.
        y_true: Ground truth (correct) labels.
        cfg: Configuration object containing experiment parameters.
    Returns:
        A dictionary containing various evaluation metrics and their confidence intervals.
    """
    assert len(np.unique(y_true)) > 2, "Only multi-class classification is supported."
    a_mask = preds != -1
    a_rate = np.sum(a_mask) / len(a_mask)

    def wmcc(y_true, y_pred): return mcc(y_true, y_pred) * \
        a_rate

    def _mcc(y_true, y_pred): return mcc(y_true, y_pred)

    def wami(y_true, y_pred): return ami(y_true, y_pred) * \
        a_rate

    def wari(y_true, y_pred): return ari(y_true, y_pred) * \
        a_rate

    preds_kwargs = dict(
        y_true=y_true[a_mask],
        y_pred=preds[a_mask],
        n_bootstraps=cfg.eval_params["n_bootstraps"]
    )

    # Get the values for each metric using the helper.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message="A single label was found in 'y_true' and 'y_pred'.*",
                                category=UserWarning
                                )
        mcc_val = safe_bootstrap(_mcc,  **preds_kwargs)
    ami_val = safe_bootstrap(ami,  **preds_kwargs)
    ari_val = safe_bootstrap(ari,  **preds_kwargs)
    if np.equal(a_mask.mean(), 1):
        wmcc_val = mcc_val
        wami_val = ami_val
        wari_val = ari_val
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="A single label was found in 'y_true' and 'y_pred'.*",
                                    category=UserWarning
                                    )
            wmcc_val = safe_bootstrap(wmcc, **preds_kwargs)
        wami_val = safe_bootstrap(wami, **preds_kwargs)
        wari_val = safe_bootstrap(wari,  **preds_kwargs)

    try:
        mAP_val = mAP(y_true[a_mask],
                      scores[a_mask])

    except:
        try:
            mAP_val = mAP(y_true[a_mask],
                          np.zeros_like(scores[a_mask]))
        except:
            try:
                mAP_val = mAP(y_true[a_mask],
                              np.zeros_like(scores[a_mask]))
            except:
                mAP_val = 0

    metric_with_ci = {
        "mcc": mcc_val,
        "ami": ami_val,
        "ari": ari_val,
        "wmcc": wmcc_val,
        "wami": wami_val,
        "wari": wari_val,
        "map": mAP_val,
        "wmap": mAP_val * a_rate,
        "acceptance_rate": a_rate,
        "n": y_true[a_mask].shape[0],
    }
    return metric_with_ci


def should_process_layer(layer_id: int, cfg: Dict) -> bool:
    """Determine if a given layer should be processed.
    Args:
        layer_id: The ID of the layer to check.
        cfg: Configuration dictionary containing experiment parameters.
    Returns:
        True if the layer should be processed, False otherwise.
    """
    layer_range = np.quantile(
        cfg.model['layers'], cfg.layer_range, method="closest_observation")
    return (layer_range[0] <= layer_id <= layer_range[1])


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tf:
        tf.write(data)
        tmp_name = tf.name
    Path(tmp_name).replace(target)


def _atomic_write_json(target: Path, obj: Any, indent: int = 2) -> None:
    def default(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        # add more conversions if needed
        raise TypeError(f"Unserializable type: {type(o)}")
    payload = json.dumps(obj, indent=indent, default=default).encode("utf-8")
    _atomic_write_bytes(target, payload)


def _atomic_joblib_dump(target: Path, obj: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tf:
        tmp_path = Path(tf.name)
    joblib.dump(obj, tmp_path, compress=3, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(target)


# Other misc function


# def return_layers(model, config):
#     """
#     Return the layers to save activations from (as a list)
#     """
#     return list(range(len(model.get_submodule(config["module"]).get_submodule(config["encoders"]))))


# def load_data(args, datasets):
#     """
#     Load the data from the config file
#     """
#     dh = DataHandler(args.experiment, datasets,
#                      args.agg, with_calibration=True, load_scores=True,
#                      )
#     dh.assemble(test_size=args.test_size, calibration_size=args.cal_size,
#                 seed=args.random_seed, exclusive_split=True)
#     return dh


# def save_checkpoint(scores, dataset, experiment, batch_index, suffix=""):
#     """Save checkpoint with scores and the current batch index."""
#     save_dir = f"outputs/checkpoints/{experiment}/{dataset}/{suffix}"
#     os.makedirs(save_dir, exist_ok=True)
#     torch.save({'scores': scores, 'batch_index': batch_index},
#                os.path.join(save_dir, "checkpoint.pt"))


# def load_checkpoint(dataset, experiment):
#     """Load checkpoint if it exists."""
#     checkpoint_path = f"outputs/checkpoints/{experiment}/{dataset}/checkpoint.pt"
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, weights_only=True)
#         return checkpoint['scores'], checkpoint['batch_index']
#     return None, 0


# def load_statements_from_multiple_data(dataset: List) -> List[str]:
#     """
#     Load the dataset from the mulitple files
#     """
#     data = list()
#     for dataset in dataset:
#         data.append(pl.read_csv(f"datasets/{dataset}.csv"))
#     return flatten(pl.concat(data).select("statement").to_numpy())


# def load_statements(dataset: List) -> List[str]:
#     """
#     Load the dataset from the file
#     """
#     return flatten(pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()), flatten(pl.read_csv(f"datasets/{dataset}.csv").select("correct").to_numpy())


# def prepare_hf_model(model_config, device):
#     """
#     Prepare the HuggingFace model and tokenizerz for the experiment
#     """
#     if model_config["dtype"] == "float16":
#         _dtype = torch.float16
#         model = AutoModelForCausalLM.from_pretrained(
#             pretrained_model_name_or_path=model_config["model"], token=model_config["token"], torch_dtype=_dtype,  attn_implementation="eager", device_map={"": device})
#     elif model_config["dtype"] == "float32":
#         _dtype = torch.float32
#         model = AutoModelForCausalLM.from_pretrained(
#             pretrained_model_name_or_path=model_config["model"], token=model_config["token"], torch_dtype="auto", device_map={"": device})
#     else:
#         raise ValueError("dtype must be either 'bfloat16' or 'float32'.")

#     # Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path=model_config["model"], token=model_config["token"])
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.padding_side = "left"
#     # Load model

#     return model, tokenizer


# def load_config(key: str) -> List[int]:
#     """
#     Load the configuration from the file and return the values for the key
#     """
#     with open("config.yaml", "r") as f:
#         return yaml.safe_load(f)[key]


# def flatten(xss) -> List:
#     """
#     Flatten a list of lists
#     """
#     return [x for xs in xss for x in xs]


# def translate_embedding(X, direction, target_coord: float, absolute: bool = True):
#     """
#     Translate the 'X' embedding in the direction  by 'new_coord' units (if abslute is True)
#     Args:
#     - X: The embedding to translate
#     - direction: The direction to translate the embedding
#     - new_coord: The new coordinate to move to
#     Returns:
#     - The translated embedding
#     """
#     # move in respect to the current coordinate
#     curr_coord = torch.einsum("bsh, h -> bs", X, direction)
#     delta = target_coord - curr_coord
#     proj = torch.einsum("h, bs -> bsh", direction, delta)
#     return X + proj


# def amplify_embedding(X, direction, factor: float = 1.0):
#     """
#     Amplify the 'X' embedding in the direction by 'factor' units
#     Args:
#     - X: The embedding to amplify
#     - direction: The direction to amplify the embedding
#     - factor: The factor to amplify the embedding
#     Returns:
#     - The amplified embedding
#     """
#     proj = torch.einsum("h, bsh -> bsh", direction, X)
#     proj = torch.sign(proj) * direction
#     return X + factor * proj


# def bootstrap_ci(metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05, random_state=None):
#     """
#     Calculate bootstrapped confidence intervals for a given scikit-learn metric.

#     Parameters:
#     - metric_func: A scikit-learn metric function (e.g., accuracy_score, precision_score).
#     - y_true: Array-like of shape (n_samples,), Ground truth (correct) labels.
#     - y_pred: Array-like of shape (n_samples,), Predicted labels, as returned by a classifier.
#     - n_bootstraps: Number of bootstrap samples to generate (default: 1000).
#     - alpha: Significance level for the confidence intervals (default: 0.05).
#     - random_state: Random seed for reproducibility (default: None).

#     Returns:
#     - ci_lower: Lower bound of the confidence interval.
#     - ci_upper: Upper bound of the confidence interval.
#     """

#     if random_state is not None:
#         np.random.seed(random_state)

#     bootstrapped_scores = []

#     for _ in range(n_bootstraps):
#         # Generate a bootstrap sample
#         indices = np.random.randint(0, len(y_true), len(y_true))
#         y_true_bootstrap = y_true[indices]
#         y_pred_bootstrap = y_pred[indices]

#         # Calculate the metric for the bootstrap sample
#         score = metric_func(y_true_bootstrap, y_pred_bootstrap)
#         bootstrapped_scores.append(score)

#     # Calculate the confidence interval
#     ci_lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
#     ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

#     return metric_func(y_true, y_pred), ci_lower, ci_upper


# def return_label(data):
#     """
#     Return labels from the dataframe
#     """
#     correct, real, fake, fictional, negated = data["correct"].values, data["real_object"].values, data[
#         "fake_object"].values, data["fictional_object"].values, data["negation"].values
#     combined = np.select(
#         [
#             (correct == 0) & (real == 1) & (fake == 0) & (fictional == 0),
#             (correct == 1) & (real == 1) & (fake == 0) & (fictional == 0),
#             (fake == 1) & (fictional == 0) & (real == 0),
#             (correct == 0) & (fake == 1) | (correct == 0) & (fictional == 1),
#             (correct == 1) & (fake == 1) | (correct == 1) & (fictional == 1),
#         ],
#         [0, 1, 4, 2, 3], default=4
#     )
#     return correct, real, fake, combined, negated, fictional


# def normalize(X):
#     if X.ndim == 1:
#         return X / np.linalg.norm(X)
#     return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


# def create_label_map(unique_labels, colors):
#     """
#     Create a label map for plots with unique labels and colors
#     """
#     return {label: color for label, color in zip(unique_labels, colors)}


# def get_device():
#     """
#     Get the device to use for computation
#     """
#     if torch.cuda.is_available():
#         # If CUDA is available, select the first CUDA device
#         device = torch.device("cuda:0")
#         print("Using CUDA device:", torch.cuda.get_device_name(0))
#     # Check for MPS availability on supported macOS devices (requires PyTorch 1.12 or newer)
#     elif torch.backends.mps.is_available():
#         # If MPS is available, use MPS device
#         device = torch.device("mps")
#         print("Using MPS (Metal Performance Shaders) device")
#     else:
#         # Fallback to CPU if neither CUDA nor MPS is available
#         device = torch.device("cpu")
#         print("Using CPU")
#     return device

# def extract_number(filename):
#     """
#     Extract the number from a filename
#     """
#     match = re.search(r'layer_(\d+)', filename)
#     return int(match.group(1)) if match else 0
