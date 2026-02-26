"""Utility functions for metrics, logging, and file operations.

This module provides various utility functions for calculating metrics,
bootstrap confidence intervals, and atomic file operations.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import energy_distance
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import average_precision_score as mAP
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import recall_score as recall

log = logging.getLogger(__name__)


def available_layers(output_dir: Path | str) -> list[int]:
    """Get list of available layer IDs from manifest files.

    Args:
        output_dir: Path to the output directory containing manifests.

    Returns:
        Sorted list of layer IDs found in manifest files.
    """
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifests"

    layers = set()
    for f in manifest_path.glob("manifest_*.json"):
        m = re.search(r"manifest_(\d+)", f.name)
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)


# Misc Functions


def bootstrap_ci(
    metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05, random_state=None
) -> tuple[float, float, float]:
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


def safe_bootstrap(metric, **kwargs) -> tuple[float, float, float]:
    """
    Safely compute bootstrap confidence intervals for a given metric.
    If an error occurs, return a default value of (0, -1e-6, 1e-6).
    Args:
        metric: Metric function to compute (e.g., accuracy_score, precision_score).
        **kwargs: Arguments to pass to the bootstrap_ci function.
    Returns:
        A tuple containing the mean, lower bound, and upper bound of the confidence interval.
    """
    try:
        return bootstrap_ci(metric, **kwargs)
    except Exception:
        return (0, -1e-6, 1e-6)


def log_metric_binary(
    preds: np.ndarray, scores: np.ndarray, y_true: np.ndarray, mask: np.ndarray, cfg
) -> dict[str, float]:
    """Calculate and return binary classification metrics.

    Args:
        preds: Predicted labels.
        scores: Prediction scores.
        y_true: Ground truth labels.
        mask: Boolean mask for valid samples.
        cfg: Configuration object containing evaluation parameters.

    Returns:
        Dictionary containing various binary classification metrics with
        confidence intervals.
    """
    # yhat = probs.round()
    if cfg.task != -1:
        is_binary = len(np.unique(y_true)) == 2
        if not is_binary:
            log.warning("Not a binary classification problem.")
        is_ok = (len(np.unique(preds)) > 0) & (len(np.unique(preds)) < 4)
        assert (
            is_ok
        ), "Only binary classification is supported (or binary with abstention class '-1')."

    a_mask = (preds != -1).flatten()
    preds = preds.flatten()
    scores = scores.flatten()
    a_rate = np.sum(a_mask[mask]) / len(a_mask[mask])

    def wmcc(y_true, y_pred):
        return mcc(y_true, y_pred) * a_rate

    def wami(y_true, y_pred):
        return ami(y_true, y_pred) * a_rate

    def wari(y_true, y_pred):
        return ari(y_true, y_pred) * a_rate

    full_mask = a_mask & mask

    binary_kwargs = dict(
        y_true=y_true[full_mask],
        y_pred=preds[full_mask],
        n_bootstraps=cfg.eval_params["n_bootstraps"],
    )

    # Get the values for each metric using the helper.
    mcc_val = safe_bootstrap(mcc, **binary_kwargs)
    ami_val = safe_bootstrap(ami, **binary_kwargs)
    ari_val = safe_bootstrap(ari, **binary_kwargs)
    recall_val = safe_bootstrap(recall, **binary_kwargs)
    if np.equal(a_mask.mean(), 1):
        wmcc_val = mcc_val
        wami_val = ami_val
        wari_val = ari_val
        wrecall_val = recall_val
    else:
        wmcc_val = safe_bootstrap(wmcc, **binary_kwargs)
        wami_val = safe_bootstrap(wami, **binary_kwargs)
        wari_val = safe_bootstrap(wari, **binary_kwargs)
        wrecall_val = safe_bootstrap(recall, **binary_kwargs)
    try:
        probs = scores[full_mask]
        x_min = probs.min()
        x_max = probs.max()

        # Apply min-max scaling
        probs_scaled = (probs - x_min) / (x_max - x_min)
        targets = y_true[full_mask]
        energy_val = energy_distance(
            probs_scaled[targets == 0], probs_scaled[targets == 1]
        )
    except Exception as e:
        log.warning(f"Error calculating energy distance: {e}. Setting to 1000.")
        energy_val = 1000

    try:
        mAP_val = mAP(y_true[full_mask], scores[full_mask])

    except:
        try:
            mAP_val = mAP(y_true[full_mask], np.zeros_like(scores[full_mask]))
        except:
            try:
                mAP_val = mAP(y_true[mask], np.zeros_like(scores[mask]))
            except:
                mAP_val = 0

    try:
        cm = confusion_matrix(y_true[mask], preds[mask], labels=[0, 1, 2, -1]).tolist()
    except:
        log.warning("Error calculating confusion matrix.")
        cm = None

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
        "cm": cm,
    }
    return metric_with_ci


def log_metric_multiclass(
    preds: np.ndarray,
    scores: np.ndarray,
    y_true: np.ndarray,
    cfg,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
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

    def wmcc(y_true, y_pred):
        return mcc(y_true, y_pred) * a_rate

    def _mcc(y_true, y_pred):
        return mcc(y_true, y_pred)

    def wami(y_true, y_pred):
        return ami(y_true, y_pred) * a_rate

    def wari(y_true, y_pred):
        return ari(y_true, y_pred) * a_rate

    preds_kwargs = dict(
        y_true=y_true[a_mask],
        y_pred=preds[a_mask],
        n_bootstraps=cfg.eval_params["n_bootstraps"],
    )

    # Get the values for each metric using the helper.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found in 'y_true' and 'y_pred'.*",
            category=UserWarning,
        )
        mcc_val = safe_bootstrap(_mcc, **preds_kwargs)
    ami_val = safe_bootstrap(ami, **preds_kwargs)
    ari_val = safe_bootstrap(ari, **preds_kwargs)
    if np.equal(a_mask.mean(), 1):
        wmcc_val = mcc_val
        wami_val = ami_val
        wari_val = ari_val
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found in 'y_true' and 'y_pred'.*",
                category=UserWarning,
            )
            wmcc_val = safe_bootstrap(wmcc, **preds_kwargs)
        wami_val = safe_bootstrap(wami, **preds_kwargs)
        wari_val = safe_bootstrap(wari, **preds_kwargs)

    try:
        mAP_val = mAP(y_true[a_mask], scores[a_mask])

    except:
        try:
            mAP_val = mAP(y_true[a_mask], np.zeros_like(scores[a_mask]))
        except:
            try:
                mAP_val = mAP(y_true[a_mask], np.zeros_like(scores[a_mask]))
            except:
                mAP_val = 0

    try:
        cm = confusion_matrix(
            y_true[mask].ravel(), preds[mask].ravel(), labels=[0, 1, 2, -1]
        ).tolist()
    except:
        log.warning("Error calculating confusion matrix.")
        cm = None
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
        "cm": cm,
    }
    return metric_with_ci


def should_process_layer(layer_id: int, cfg: dict) -> bool:
    """Determine if a given layer should be processed.
    Args:
        layer_id: The ID of the layer to check.
        cfg: Configuration dictionary containing experiment parameters.
    Returns:
        True if the layer should be processed, False otherwise.
    """
    layer_range = np.quantile(
        cfg.model["layers"], cfg.layer_range, method="closest_observation"
    )
    return layer_range[0] <= layer_id <= layer_range[1]


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomically write bytes to a file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tf:
        tf.write(data)
        tmp_name = tf.name
    Path(tmp_name).replace(target)


def _atomic_write_json(target: Path, obj: Any, indent: int = 2) -> None:
    """Atomically write object as JSON to a file."""

    def default(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        # add more conversions if needed
        raise TypeError(f"Unserializable type: {type(o)}")

    payload = json.dumps(obj, indent=indent, default=default).encode("utf-8")
    _atomic_write_bytes(target, payload)


def _atomic_joblib_dump(target: Path, obj: Any) -> None:
    """Atomically dump object using joblib."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tf:
        tmp_path = Path(tf.name)
    joblib.dump(obj, tmp_path, compress=3, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(target)

def safe_divide(
    numerator: float,
    denominator: float,
    min_denom: float = 1e-8,
    default: float | None = 0.0,
) -> float | 0.0:
    """Safe division with minimum denominator threshold.
    
    Args:
        numerator: The numerator.
        denominator: The denominator.
        min_denom: Minimum absolute value for denominator.
        default: Value to return if denominator is below threshold.
    
    Returns:
        numerator / denominator if |denominator| > min_denom, else default.
    """
    if abs(denominator) < min_denom:
        return default
    return numerator / denominator