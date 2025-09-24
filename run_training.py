# Code for ONE-vs-ALL probes, works both for SIL (MD+CP and SVM) and MIL (Sawmil) probes.

from scipy.stats import energy_distance
from sklearn.metrics import (
    recall_score as recall,
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
    adjusted_mutual_info_score as ami,
    adjusted_rand_score as ari,
)
import logging
import hydra
from utils_hydra import load_data, return_label, safe_bootstrap
from misc.task import Task
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
import time
from glob import glob
from misc.db import LogDataBase
import joblib
import sys
import re
import pprint
from pathlib import Path
from utils import (
    should_process_layer,
    _atomic_joblib_dump,
    _atomic_write_json,
)

from typing import Optional, Dict, List
from sklearn.base import TransformerMixin, BaseEstimator

from runners import SVMProbeRunner, MDProbeRunner, SawmilProbeRunner, SPCA_Runner

log = logging.getLogger("Training")

PROBES = {
    'svm': SVMProbeRunner,
    'mean_diff': MDProbeRunner,
    'sawmil': SawmilProbeRunner,
    'spca': SPCA_Runner
}

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    log.warning("Patched sklearn is running...")
except:
    pass

def validate_config(cfg: DictConfig):
    assert type(
        cfg.datapack['datasets']) == list or type(cfg.datapack['datasets']).__name__ == "ListConfig", f"Datasets must be a list. Not {type(cfg.datapack['datasets'])}"
    assert len(cfg.datapack['datasets']
               ) > 0, "At least one dataset must be selected."
    OmegaConf.set_struct(cfg, False)  # Allow overriding
    trial_name = cfg.trial_name
    if cfg.search:
        trial_name += "_search"
    trial_name += f'_task-{cfg.task}'
    cfg["trial_name"] = trial_name
    # if cfg["task"] == 2:
    #     cfg["probe"]["assume_known_positives"] = False
    cfg["output_dir"] = os.path.join(cfg.output_dir, trial_name)
    OmegaConf.set_struct(cfg, True)

    assert len(cfg.layer_range) == 2, "Layer range must be a list of two integers."
    assert cfg.task != -1, 'Cannot be multiclass task, choose a binary task.'


def log_stats(cfg):
    datasets_test = cfg.datapack["datasets_test"] if len(
        cfg.datapack["datasets_test"]) > 0 else cfg.datapack["datasets"]
    log.warning(
        f"Training {cfg.probe['name'].upper()}-based probe for {cfg.model['name']} activations [task: {cfg.task}]")
    log.warning(
        f"\t\tTrain datasets: {cfg.datapack['datasets']}")
    log.warning(f"\t\tTest datasets: {datasets_test})")
    log.warning(f"\t\tOutput directory: {cfg.output_dir}")


def log_metric(preds, scores, y_true, mask, cfg):
    """
    Log the metrics to the Weights and Biases dashboard with prefix and return as a dictionary without prefix.
    """
    # yhat = probs.round()
    is_binary = len(np.unique(y_true)) == 2
    assert is_binary, "Only binary classification is supported."
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


def save(concept_direction: np.ndarray,
         concept_bias: np.ndarray,
         metric_dict: Dict,
         layer_id: int,
         cfg: DictConfig,
         estimator: Optional[BaseEstimator] = None,
         conf_calibrator: Optional[BaseEstimator] = None,
         scaler: Optional[TransformerMixin] = None,
         transformer: Optional[TransformerMixin] = None,
         y_hat: np.ndarray = None, y_true: np.ndarray = None):
    '''
    Save the artifacts of the run.
    Args:
        concept_direction: The direction of the concept (coef_).
        concept_bias: The bias of the concept (intercept_).
        scaler: The scaler used for the features.
        transformer: The transformer used for the features.
        conf_calibrator: The conformal calibrator used for the predictions.
        metric_dict: The dictionary containing the metrics.
        cfg: The configuration object.
        layer_id: The ID of the layer.
        y_hat: The predicted labels.
        y_true: The true labels.
        estimator: The object of the classifier/regressor.

    '''
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Config (resolved)
    config_path = output_dir / "config.json"
    _atomic_write_json(config_path, OmegaConf.to_container(cfg, resolve=True))
    # 2. Metrics
    metrics_path = output_dir / f"metrics_{layer_id}.json"
    _atomic_write_json(metrics_path, metric_dict)
    # 3. Objects
    est_path = output_dir / f"estimator_{layer_id}.joblib" if estimator is not None else None
    if (estimator is not None):
        _atomic_joblib_dump(est_path, estimator)
    scl_path = output_dir / f"scaler_{layer_id}.joblib" if scaler is not None else None
    if (scaler is not None):
        _atomic_joblib_dump(scl_path, scaler)
    trn_path = output_dir / f"transformer_{layer_id}.joblib" if transformer is not None else None
    if (transformer is not None):
        _atomic_joblib_dump(trn_path, transformer)
    clb_path = output_dir / f"calibrator_{layer_id}.joblib" if conf_calibrator is not None else None
    if (conf_calibrator is not None):
        _atomic_joblib_dump(clb_path, conf_calibrator)

    # 4. Numpy arrays
    coef_path = output_dir / f"coef_{layer_id}.npy"
    bias_path = output_dir / f"bias_{layer_id}.npy"
    np.save(coef_path, concept_direction)
    np.save(bias_path, concept_bias)

    yh_path = output_dir / f"y_hat_{layer_id}.npy" if y_hat is not None else None
    if (y_hat is not None):
        np.save(yh_path, y_hat)
    else:
        log.warning("y_hat is None")
    yt_path = output_dir / f"y_true_{layer_id}.npy" if y_true is not None else None
    if (y_true is not None):
        np.save(yt_path, y_true)

        # 6) Manifest (quick glance + reproducibility bits)
    manifest = {
        "layer_id": layer_id,
        "paths": {
            "config": str(config_path),
            "metrics": str(metrics_path),
            "coef": str(coef_path),
            "bias": str(bias_path),
            "preds": str(yh_path) if (y_hat is not None or y_true is not None) else None,
            "estimator": str(est_path) if est_path else None,
            "scaler": str(scl_path) if scl_path else None,
            "transformer": str(trn_path) if trn_path else None,
            "calibrator": str(clb_path) if clb_path else None,
        },
        "shapes": {
            "coef": tuple(np.shape(concept_direction)),
            "bias": tuple(np.shape(concept_bias)),
            "y_hat": None if y_hat is None else tuple(np.shape(y_hat)),
            "y_true": None if y_true is None else tuple(np.shape(y_true)),
        },
        "dtypes": {
            "coef": str(getattr(concept_direction, "dtype", "")),
            "bias": str(getattr(concept_bias, "dtype", "")),
            "y_hat": None if y_hat is None else str(getattr(y_hat, "dtype", "")),
            "y_true": None if y_true is None else str(getattr(y_true, "dtype", "")),
        },
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "hydra": hydra.__version__,
            "sklearn": sys.modules['sklearn'].__version__,
            "scipy": sys.modules['scipy'].__version__,
            "polars": sys.modules['polars'].__version__ if 'polars' in sys.modules else "",
            "pandas": sys.modules['pandas'].__version__ if 'pandas' in sys.modules else "",
            "torch": sys.modules['torch'].__version__ if 'torch' in sys.modules else "",
        },
    }
    manifest_path = output_dir / "manifests"
    manifest_path.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path / f"manifest_{layer_id}.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest


def checkpointing(cfg, layer_range: List[int]) -> List[int]:
    """
    Check which layers are missing from the output directory and return them.
    If a layer is missing, also include the previous layer (to avoid missing values).
    """
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_coefs = glob(f"{cfg.output_dir}/coef_*")
    completed_layers = []
    for file in recorded_coefs:
        match = re.search(r'coef_(\d+)', file)
        if match:
            completed_layers.append(int(match.group(1)))
    model_layers = cfg.model["layers"]
    missing_layers = []
    for layer_id in model_layers:
        if layer_id < layer_range[0] or layer_id > layer_range[1]:
            continue
        if layer_id not in completed_layers:
            missing_layers.append(layer_id)

    if len(missing_layers) > 0:
        min_layer = min(missing_layers)
        if min_layer - 1 >= layer_range[0]:
            missing_layers.append(min_layer - 1)

    return sorted(missing_layers)


@hydra.main(version_base=None, config_path="configs", config_name="probe_sil")
def main(cfg: DictConfig):
    validate_config(cfg)
    log_stats(cfg)

    dh = load_data(cfg)
    dh_test = dh
    assert dh.datasets == dh_test.datasets, "The test and train dataset sources must be the same."
    layer_range = np.quantile(
        cfg.model['layers'], cfg.layer_range, method="closest_observation")
    log.warning(f"Layer range: {layer_range[0]} - {layer_range[1]}")

    # Checkpointing
    if cfg.start_from_checkpoint:
        missing_layers = checkpointing(cfg, layer_range)
        if len(missing_layers) == 0:
            log.warning(
                "All layers are already processed...")
            layers = []
            # raise Exception("All layers are already processed.")
        else:
            log.warning(
                f"Checkpointing: Processing the missing layers: {missing_layers}")
            layers = missing_layers
    else:
        layers = cfg["layers"]

    task = Task(cfg.task)
    db = LogDataBase(
        tab_name=f"{cfg.probe['name']}_fit", db_name="experiments")
    db.write(trial_id=f"{cfg.model.name}-{cfg.trial_name}",
             model=cfg.model.name,
             datapack=cfg.datapack.name,
             task=cfg.task,
             parameters=f"STARTED",
             progress=0,
             status=0)

    # PER LAYER
    if cfg.run_debugging:
        layers = [13]
    for layer_id in layers:
        if cfg.run_debugging == True and layer_id > 6 and should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id} || Debugging mode")
        elif cfg.run_debugging == False and should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id}")
        else:
            log.warning(f"Skipping layer {layer_id}")
            continue

        # LOAD THE train DATA
        X_tr = dh.train_bags(
            layer_id=layer_id, drop_zeros=True)["embeddings"]
        data_train = dh.get_train_df().reset_index(drop=True)
        _y_train, r_train, _ = return_label(data_train)

        # LOAD THE TEST DATA
        data_test = dh_test.get_test_df().reset_index(drop=True)
        _y_test, r_test, _ = return_label(data_test)

        # LOAD THE CALIBRATION DATA
        if dh.with_calibration:
            data_cal = dh.get_cal_df().reset_index(drop=True)
            _y_cal, r_cal, _ = return_label(data_cal)

        train_labels = task.return_labels(_y_train, r_train)
        y_train, mask = train_labels['targets'], train_labels['mask']
        test_labels = task.return_labels(_y_test, r_test)
        y_test, mask_test = test_labels['targets'], test_labels['mask']
        if dh.with_calibration:
            cal_labels = task.return_labels(_y_cal, r_cal)
            y_cal, mask_cal = cal_labels['targets'], cal_labels['mask']

        start_time = time.time()
        runner = PROBES[cfg.probe['name']](cfg)
        if cfg.search:
            result = runner.parameter_search(
                X=X_tr, y=y_train, mask=mask)
        else:
            # try:
            result = runner.single_training(
                X=X_tr, y=y_train, mask=mask)

        # CONFORMAL PREDICTION
        X_te = dh_test.test_bags(
            layer_id=layer_id, drop_zeros=True)["embeddings"]
        # LOAD THE CALIBRATION DATA
        X_cal = dh.cal_bags(
            layer_id=layer_id, drop_zeros=True)["embeddings"]
        # CONFORMAL PREDICTION
        calibrator = runner.conformal_training(X_cal, y_cal, mask_cal)

        yh_te = runner.predict_proba(X_te)
        yc_te = runner.conformal_prediction(X_te)
        preds = runner.predict(X_te)
        # Assemble Metrics
        metric_dict = {}
        metric_dict['default'] = log_metric(preds=preds,
                                            scores=yh_te,
                                            y_true=y_test,
                                            mask=mask_test,
                                            cfg=cfg)
        metric_dict['default']["coverage"] = 1.0
        metric_dict['default'] = runner.update_metric(metric_dict['default'])
        metric_dict['conformal'] = log_metric(preds=yc_te, scores=yh_te,
                                              y_true=y_test, mask=mask_test, cfg=cfg)
        metric_dict['conformal']["coverage"] = calibrator.coverage(
            scores=yh_te[mask_test], y=y_test[mask_test])
        try:
            metric_dict['conformal']["acceptance_rate"] = runner.conformal_acc_rate(
                X_te[mask_test])
        except:
            metric_dict['conformal']["acceptance_rate"] = calibrator.acceptance_rate(
                yh_te)

        if cfg.save_results:
            save(concept_direction=runner.direction,
                 concept_bias=runner.bias,
                 estimator=runner.estimator,
                 metric_dict=metric_dict,
                 scaler=runner.scaler,
                 transformer=runner.transformer,
                 conf_calibrator=runner.calibrator,
                 cfg=cfg,
                 layer_id=layer_id,
                 y_hat=yh_te,
                 y_true=y_test)
        else:
            log.warning(
                f"Conformal metric: {pprint.pformat(metric_dict['conformal'], indent=2)}")
            log.warning(
                f"Default metric: {pprint.pformat(metric_dict['default'], indent=2)}")
        end_time = time.time()

        # logging
        first_part = f"\t{cfg.probe.name.capitalize()} probe took"
        log.warning(
            f" L{layer_id} {first_part:<20} {(end_time - start_time):<4.2f} seconds | MCC (def): {metric_dict['default']['wmcc'][0]:>5.2f} | MCC (conf): {metric_dict['conformal']['wmcc'][0]:>5.2f}")
        log.warning(
            f"\t\tA-rate (def): {metric_dict['default']['acceptance_rate']:>5.2f} | A-rate (conf): {metric_dict['conformal']['acceptance_rate']:>5.2f}")
        log.warning(
            '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||')

        db_params = f"C:{cfg.probe['init_params']} | Layers: {layer_id}/{layers[-1]}"
        db_trial_id = f"{cfg.model.name}-{cfg.trial_name}"
        db.write(trial_id=db_trial_id,
                 model=cfg.model.name,
                 datapack=cfg.datapack.name,
                 task=cfg.task,
                 parameters=db_params,
                 progress=layer_id/layers[-1],
                 status=0)
    log.warning(f"Finished running the {cfg.probe.name} probe.")
    db_trial_id = f"{cfg.model.name}-{cfg.trial_name}"
    db.write(trial_id=db_trial_id,
             model=cfg.model.name,
             datapack=cfg.datapack.name,
             task=cfg.task,
             parameters='Finished',
             progress=1,
             status=1)


if __name__ == "__main__":
    main()
