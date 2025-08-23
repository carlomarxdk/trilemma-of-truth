from utils_hydra import load_data_with_test, return_label,  safe_bootstrap
from sklearn.metrics import (
    normalized_mutual_info_score as nmi,
    adjusted_mutual_info_score as ami,
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
    adjusted_rand_score as ari,
)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os
from glob import glob
import json
import re
import pprint
import pickle
from copy import deepcopy
from misc.probe_data import MulticlassProbeData as MPD
from misc.db import LogDataBase

from utils import should_process_layer
import warnings


log = logging.getLogger(__name__)


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
    output_dir = cfg.output_dir.split(
        "task")[0] + '-to-' + cfg.datapack_test["name"]
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


def log_metric(preds, scores, y_true, cfg):
    """
    Log the metrics to the Weights and Biases dashboard with prefix and return as a dictionary without prefix.
    """

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


@hydra.main(version_base=None, config_path="configs", config_name="probe_linear_mil")
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
    if cfg.probe['name'] == 'sawmil':
        probe_path = f'outputs/probes/{cfg.probe["name"]}/{cfg.model["name"]}/{cfg.datapack["name"]}_search_task--1'
        # reader = MCProbeData(model_name=cfg.model['name'],
        #                      datapack_name=cfg.datapack['name'],
        #                      probe_name=cfg.probe['name'])
        reader = MPD(probe_path)
    else:
        raise NotImplementedError()
    dh_test = load_data_with_test(cfg)
    data_test = dh_test.get_test_df().reset_index(drop=True)
    labels = dh_test.get_test_labels()
    layer_range = np.quantile(
        cfg.model['layers'], cfg.layer_range, method="closest_observation")
    log.warning(f"Layer range: {layer_range[0]} - {layer_range[1]}")

    available_layers = reader.available_layers()
    print(f"Available layers: {available_layers}")

    # Checkpointing
    if cfg.start_from_checkpoint:
        missing_layers = checkpointing(cfg, available_layers=available_layers)
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
        if cfg.probe['name'] == 'sawmil':
            X_te = dh_test.test_bags(
                layer_id=layer_id)["embeddings"]
            probs = reader.predict_proba(layer_id=layer_id, bags=X_te)
            preds = reader.predict(layer_id=layer_id, bags=X_te)

            cp = reader.calibrator(layer_id=layer_id)
            result = cp.evaluate(probs)
            yc_test = result["predictions"]
        else:
            raise NotImplementedError(
                f"Probe {cfg.probe['name']} not implemented for test data loading.")
        # Make predictions

        metric_dict = {}
        metric_dict['default'] = log_metric(y_true=labels,
                                            preds=preds,
                                            scores=probs,
                                            cfg=cfg)
        metric_dict['conformal'] = log_metric(y_true=labels,
                                              preds=yc_test,
                                              scores=probs,
                                              cfg=cfg)

        output_dir = cfg.output_dir.split(
            "task")[0] + '-to-' + cfg.datapack_test["name"]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if cfg.save_results:
            log.warning('Saving path: ' + output_dir)
            log.warning(
                f" (Saving) Metric for layer {layer_id}: \n{pprint.pformat(metric_dict, indent=2)}")
            with open(f"{output_dir}/metrics_{layer_id}.json", "w") as f:
                json.dump(metric_dict, f, cls=NpEncoder)
            np.save(f"{output_dir}/y_hat_{layer_id}.npy", probs)
            np.save(f"{output_dir}/y_true.npy", labels)
        else:
            log.warning(
                f"Metric for layer {layer_id}: \n{pprint.pformat(metric_dict, indent=2)}")

        # Write to DATABASE
        db_params = f"WMCC: {metric_dict['conformal']['wmcc']}Layers: {layer_id}/{reader.available_layers()[-1]}"
        db_trial_id = f"{cfg.model.name}-{cfg.datapack.name}-{cfg.datapack_test.name}"
        db.write(trial_id=db_trial_id,
                 model=cfg.model.name,
                 datapack=cfg.datapack.name,
                 task=cfg.task,
                 parameters=db_params,
                 progress=layer_id/reader.available_layers()[-1],
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
