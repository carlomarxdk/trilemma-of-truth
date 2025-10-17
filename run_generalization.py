from utils_hydra import load_data_with_test
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
from glob import glob
import json
import re
import pprint
from typing import Dict
from pathlib import Path
import sys
import joblib
from misc.probe_data import MulticlassProbeData as MPD
from misc.db import LogDataBase

from utils import (
    should_process_layer,
    log_metric_multiclass as log_metric,
    log_metric_binary,
    _atomic_write_json,
    available_layers,
)
from runners import SVMProbeRunner, MDProbeRunner, SawmilProbeRunner, SPCA_Runner, TTPD_Runner

PROBES = {
    'svm': SVMProbeRunner,
    'mean_diff': MDProbeRunner,
    'sawmil': SawmilProbeRunner,
    'spca': SPCA_Runner,
    'ttpd': TTPD_Runner,
}

log = logging.getLogger('Generalization')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def validate_config(cfg: DictConfig):
    assert type(
        cfg.datapack['datasets']) == list or type(cfg.datapack['datasets']).__name__ == "ListConfig", f"Datasets must be a list. Not {type(cfg.datapack['datasets'])}"
    assert type(
        cfg.datapack_test['datasets']) == list or type(cfg.datapack_test['datasets']).__name__ == "ListConfig", f"Datasets must be a list. Not {type(cfg.datapack_test['datasets'])}"
    assert len(cfg.datapack['datasets']
               ) > 0, "At least one dataset must be selected."
    assert len(cfg.datapack_test['datasets']
               ) > 0, "At least one test dataset must be selected."
    OmegaConf.set_struct(cfg, False)  # Allow overriding
    trial_name = cfg.trial_name
    if cfg.search:
        trial_name += "_search"
    trial_name += f'_task-{cfg.task}'
    cfg["trial_name"] = trial_name
    cfg["output_dir"] = os.path.join(cfg.output_dir, trial_name)
    OmegaConf.set_struct(cfg, True)

    assert len(cfg.layer_range) == 2, "Layer range must be a list of two integers."


def log_stats(cfg):
    log.warning(f"Running probe generalizability experiment.")
    log.warning(f"Datapack Test: {cfg.datapack_test['name']}")
    datasets_test = cfg.datapack["datasets_test"] if len(
        cfg.datapack["datasets_test"]) > 0 else cfg.datapack["datasets"]
    log.warning(
        f"Collection {cfg.probe['name']}-based metrics for {cfg.model['name']}")
    log.warning(
        f"\t\tTrain datasets: {cfg.datapack['datasets']}")
    log.warning(f"\t\tTest datasets: {datasets_test})")
    log.warning(f"\t\tOutput directory: {cfg.output_dir}")


def checkpointing(cfg, available_layers):
    output_dir = Path(cfg.output_dir) / f"g_{cfg.datapack_test['name']}"
    recorded_layers = glob(f"{output_dir}/y_hat_*")
    completed_layers = []
    for file in recorded_layers:
        match = re.search(r'y_hat_(\d+)', file)
        if match:
            completed_layers.append(int(match.group(1)))
    model_layers = set(available_layers)

    completed_layers = set(completed_layers)

    if len(completed_layers) == 0:
        missing_layers = list(model_layers)
    else:
        missing_layers = list(model_layers - completed_layers)

    return sorted(missing_layers)


def save(metric_dict: Dict,
         layer_id: int,
         cfg: DictConfig,
         y_hat: np.ndarray = None,
         y_true: np.ndarray = None):
    '''
    Save the artifacts of the run.
    Args:
        metric_dict: The dictionary containing the metrics.
        cfg: The configuration object.
        layer_id: The ID of the layer.
        y_hat: The predicted labels.
        y_true: The true labels.

    '''
    output_dir = Path(str(cfg.output_dir)) / f"g_{cfg.datapack_test['name']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics
    metrics_path = output_dir / f"metrics_{layer_id}.json"
    _atomic_write_json(metrics_path, metric_dict)
    # 2. Numpy arrays
    yh_path = output_dir / \
        f"y_hat_{layer_id}.npy" if y_hat is not None else None
    if (y_hat is not None):
        np.save(yh_path, y_hat)
    else:
        log.warning("y_hat is None")
    yt_path = output_dir / f"y_true.npy" if y_true is not None else None
    if (y_true is not None):
        np.save(yt_path, y_true)

        # 6) Manifest (quick glance + reproducibility bits)
    manifest = {
        "layer_id": layer_id,
        "datapack_base": cfg.datapack['name'],
        "datapack_test": cfg.datapack_test['name'],
        "probe": cfg.probe['name'],
        "model": cfg.model['name'],
        "task": cfg.task,
        "paths": {
            "metrics": str(metrics_path),
        },
        "shapes": {
            "y_hat": None if y_hat is None else tuple(np.shape(y_hat)),
            "y_true": None if y_true is None else tuple(np.shape(y_true)),
        },
        "dtypes": {
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

    log.warning(f"Saved artifacts to {output_dir}")
    return manifest


@hydra.main(version_base=None, config_path="configs", config_name="probe_sil")
def main(cfg: DictConfig):
    log.warning(f"Datapack Test: {cfg.datapack_test}")
    validate_config(cfg)
    log_stats(cfg)
    db = LogDataBase(
        tab_name=f"{cfg.probe['name']}_generalization", db_name="experiments")
    db.write(trial_id=f"{cfg.model.name}-{cfg.datapack.name}-{cfg.datapack_test.name}",
             model=cfg.model.name,
             datapack=cfg.datapack.name,
             task=-1,
             parameters=f"STARTED",
             progress=0,
             status=0)
    
    dh_test = load_data_with_test(cfg)
    labels = dh_test.get_test_labels()
    layer_range = np.quantile(
        cfg.model['layers'], cfg.layer_range, method="closest_observation")
    log.warning(f"Layer range: {layer_range[0]} - {layer_range[1]}")

    avail_layers = available_layers(cfg.output_dir)
    print(f"Available layers: {avail_layers}")

    # Checkpointing
    if cfg.start_from_checkpoint:
        missing_layers = checkpointing(cfg, available_layers=avail_layers)
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

    if cfg.run_debugging:
        layers = [13]
    # PER LAYER
    for layer_id in layers:
        if cfg.run_debugging == True and layer_id > 6 and should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id} || Debugging mode")
        elif cfg.run_debugging == False and should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id}")
        else:
            log.warning(f"Skipping layer {layer_id}")
            continue
        # LOAD THE TEST DATA
        runner = PROBES[cfg.probe['name']](cfg)
        runner.load(output_dir=cfg.output_dir, layer_id=layer_id)
        
        
        # CONFORMAL PREDICTION
        X_te = dh_test.test_bags(
            layer_id=layer_id, drop_zeros=True)["embeddings"]
        
        bag_yh_te = runner.bag_predict_proba(X_te)
        bag_yc_te = runner.bag_conformal_prediction(X_te)
        bag_preds = runner.bag_predict(X_te)
        
        metric_dict ={
            'bag': {},
            'instance': {}, #last instance in bag
            'instance_tf': {}, #only on true and false
        }

        # Metrics for the whole bag
        metric_dict['bag']['default'] = log_metric(preds=bag_preds,
                                            scores=bag_yh_te,
                                            y_true=labels,
                                            cfg=cfg)
        metric_dict['bag']['conformal'] = log_metric(y_true=labels,
                                              preds=bag_yc_te,
                                              scores=bag_yh_te,
                                              cfg=cfg)
        
        # Metrics for the last instance in the bag
        yh_te = runner.inst_predict_proba(X_te)
        yc_te = runner.inst_conformal_prediction(X_te)
        preds = runner.inst_predict(X_te)
        
        metric_dict['instance']['default'] = log_metric(preds=preds,
                                            scores=yh_te,
                                            y_true=labels,
                                            cfg=cfg)
        metric_dict['instance']['conformal'] = log_metric(y_true=labels,
                                              preds=yc_te,
                                              scores=yh_te,
                                              cfg=cfg)
        
        # Metrics for the last instance in the bag, only true and false (no neither-valued statements)
        mask_tf = (labels == 1) | (labels == 0)
        metric_dict['instance_tf']['default'] = log_metric_binary(preds=preds,
                                                                  scores=yh_te,
                                                                  y_true=labels,
                                                                  mask=mask_tf, cfg=cfg)
        metric_dict['instance_tf']['conformal'] = log_metric_binary(y_true=labels,
                                                                   preds=yc_te,
                                                                   scores=yh_te,
                                                                   mask=mask_tf, cfg=cfg)

        if cfg.save_results:
            _ = save(metric_dict=metric_dict,
                     layer_id=layer_id,
                     cfg=cfg,
                     y_hat=bag_yh_te,
                     y_true=labels)
        else:
            log.warning(
                f"Metric for layer {layer_id}: \n{pprint.pformat(metric_dict, indent=2)}")

        # Write to DATABASE
        db_params = f"WMCC: {metric_dict['bag']['conformal']['wmcc']}Layers: {layer_id}/{avail_layers[-1]}"
        db_trial_id = f"{cfg.model.name}-{cfg.datapack.name}-{cfg.datapack_test.name}"
        db.write(trial_id=db_trial_id,
                 model=cfg.model.name,
                 datapack=cfg.datapack.name,
                 task=cfg.task,
                 parameters=db_params,
                 progress=layer_id/avail_layers[-1],
                 status=0)

    db.write(trial_id=f"{cfg.model.name}-{cfg.datapack.name}-{cfg.datapack_test.name}",
             model=cfg.model.name,
             datapack=cfg.datapack.name,
             task=-1,
             parameters=f"Finished",
             progress=1,
             status=1)
    log.warning(
        'Finished!')


if __name__ == "__main__":
    main()
