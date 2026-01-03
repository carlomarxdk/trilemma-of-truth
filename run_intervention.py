"""Run causal interventions on language model representations.

This module performs difference-in-differences analyses to test whether probe
directions causally influence model outputs by intervening on hidden states
during forward passes.
"""

from __future__ import annotations

import logging
import os
import pprint
import re
import warnings
from collections.abc import Sequence
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import statsmodels.api as sm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings when loading pickled models
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

from misc.db import LogDataBase
from misc.probe_data import ExperimentData
from response.interventions_utils import (
    InstructInterventionDataProcessor,
    InterventionDataProcessor,
    compute_layer_scale,
    diff_of_diff_ols,
    intervention_success_rate,
    mean_logprobs,
    random_answer_ids,
    translate_concept,
)
from runners import (
    MDProbeRunner,
    MulticlassMILRunner,
    MulticlassSVMRunner,
    SawmilProbeRunner,
    SPCA_Runner,
    SVMProbeRunner,
    TTPD_Runner,
)
from utils import (
    _atomic_write_json,
    available_layers,
    should_process_layer,
)
from utils_hydra import (
    clear_device_cache,
    get_device,
    load_data,
    prepare_nnsight,
)

PROBES = {
    "svm": SVMProbeRunner,
    "mean_diff": MDProbeRunner,
    "sawmil": SawmilProbeRunner,
    "sawmil_mc": MulticlassMILRunner,
    "svm_mc": MulticlassSVMRunner,
    "spca": SPCA_Runner,
    "ttpd": TTPD_Runner,
}


log = logging.getLogger("Intervention")


def validate_config(cfg: DictConfig):
    """Validate and prepare configuration for intervention experiments.

    Args:
        cfg: Hydra configuration object.

    Raises:
        ValueError: If datasets are not a list, empty, or layer_range is invalid.

    """
    if not (
        isinstance(cfg.datapack["datasets"], list)
        or type(cfg.datapack["datasets"]).__name__ == "ListConfig"
    ):
        raise ValueError(
            f"Datasets must be a list. Not {type(cfg.datapack['datasets'])}"
        )
    if len(cfg.datapack["datasets"]) == 0:
        raise ValueError("At least one dataset must be selected.")
    OmegaConf.set_struct(cfg, False)  # Allow overriding
    trial_name = cfg.trial_name
    if cfg.search:
        trial_name += "_search"
    trial_name += f"_task-{cfg.task}"
    cfg["trial_name"] = trial_name
    cfg["output_dir"] = os.path.join(cfg.output_dir, trial_name)
    cfg["probe_dir"] = os.path.join(cfg.probe_dir, trial_name)
    if cfg.device is None:
        cfg["device"] = str(get_device())
    OmegaConf.set_struct(cfg, True)

    if len(cfg.layer_range) != 2:
        raise ValueError("Layer range must be a list of two integers.")


def log_stats(cfg: DictConfig):
    """Log experiment configuration and parameter information.

    Args:
        cfg: Hydra configuration object.

    """
    datasets_test = (
        cfg.datapack["datasets_test"]
        if len(cfg.datapack["datasets_test"]) > 0
        else cfg.datapack["datasets"]
    )
    log.warning(
        f"Interventions for {cfg.probe['name']}-based probe for {cfg.model['name']} [task: {cfg.task}]"
    )
    log.warning(f"\t\tTrain datasets: {cfg.datapack['datasets']}")
    log.warning(f"\t\tTest datasets: {datasets_test})")
    layer_range = np.quantile(
        cfg.model["layers"], cfg.layer_range, method="closest_observation"
    )
    log.warning(f"Layer range: {layer_range[0]} - {layer_range[1]}")
    log.warning(f"\t\tConfiguration: {cfg}")


def checkpointing(cfg: DictConfig, existing_layers: Sequence[int]) -> list[int]:
    """Checkpointing function to resume interrupted intervention runs.

    Args:
        cfg: Configuration
        existing_layers: List of all available model layers
    Returns:
        missing_layers: List of layers that have not been processed yet

    """
    output_dir = Path(cfg.output_dir)
    recorded_layers = glob(f"{output_dir}/layer_*")
    completed_layers = []
    for file in recorded_layers:
        match = re.search(r"layer_(\d+)", file)
        if match:
            completed_layers.append(int(match.group(1)))
    model_layers = set(existing_layers)

    completed_layers = set(completed_layers)

    if len(completed_layers) == 0:
        missing_layers = list(model_layers)
    else:
        missing_layers = list(model_layers - completed_layers)

    return sorted(missing_layers)


@hydra.main(version_base=None, config_path="configs", config_name="interventions")
def main(cfg: OmegaConf): # noqa: C901
    """Run intervention experiments on language model layers.

    Loads probes, performs causal interventions by translating hidden states
    along probe directions, and analyzes effects on output probabilities using
    difference-in-differences regression.

    Args:
        cfg: Hydra configuration object.

    """
    # PRERAMBLE
    validate_config(cfg)
    log_stats(cfg)
    db = LogDataBase(tab_name=f"{cfg.probe['name']}_inter", db_name="experiments")
    db.write(
        trial_id=f"{cfg.model.name}-{cfg.datapack.name}-{cfg.task}",
        model=cfg.model.name,
        datapack=cfg.datapack.name,
        task=cfg.task,
        parameters="STARTED",
        progress=0,
        status=0,
    )

    task = cfg.task
    if task == -1:
        raise NotImplementedError("Multiclass probe not implemented yet.")

    device = torch.device(cfg.device)
    model, tokenizer = prepare_nnsight(cfg)
    dh = load_data(cfg)
    runner = PROBES[cfg.probe["name"]](cfg)  # INSTANTIATE THE PROBE RUNNER

    if not cfg.model["instruct"]:
        log.warning("Using standard InterventionDataProcessor")
        idp = InterventionDataProcessor(
            datahandler=dh, datapack_name=cfg.datapack["name"], tokenizer=tokenizer
        )
    elif cfg.model["instruct"]:
        log.warning("Using InstructInterventionDataProcessor")
        idp = InstructInterventionDataProcessor(
            datahandler=dh,
            datapack_name=cfg.datapack["name"],
            tokenizer=tokenizer,
            user_role=cfg.model["user_role"],
            assist_role=cfg.model["assist_role"],
            system_role=cfg.model["system_role"],
        )
    else:
        raise NotImplementedError()

    avail_layers = available_layers(cfg.probe_dir)

    # Determine which layers to process
    if cfg.use_best_layers:
        # Use top N best performing layers based on probe metrics
        if cfg.probe['name'] != 'svm':
            best_layer_metric_keys =  cfg.best_layer_metric_keys
        else:
            best_layer_metric_keys = ["default", "mcc"]
        log.warning(
            f"Using top {cfg.num_best_layers} best performing layers based on {best_layer_metric_keys} metrics"
        )
        
        # Create ExperimentData object to access probe metrics
        experiment_data = ExperimentData(
            model_name=cfg.model["name"],
            probe_name=cfg.probe["name"],
            dataset_name=cfg.datapack["name"],
            task=cfg.task,
            with_search=cfg.search,
        )
        # Get top k best performing layers
        best_layers = experiment_data.top_k_layers(
            k=cfg.num_best_layers,
            keys=best_layer_metric_keys,
        )
        log.warning(f"Selected best layers: {best_layers}")
        
        # If checkpointing is enabled, only process layers not yet completed
        if cfg.start_from_checkpoint:
            missing_layers = checkpointing(cfg, existing_layers=best_layers)
            if len(missing_layers) == 0:
                log.warning("All best layers are already processed...")
                layers = []
            else:
                log.warning(
                    f"Checkpointing: Processing the missing best layers: {missing_layers}"
                )
                layers = missing_layers
        else:
            layers = best_layers
    else:
        # Use standard layer selection based on layer_range or checkpointing
        if cfg.start_from_checkpoint:
            missing_layers = checkpointing(cfg, existing_layers=avail_layers)
            if len(missing_layers) == 0:
                log.warning("All layers are already processed...")
                layers = []
            else:
                log.warning(
                    f"Checkpointing: Processing the missing layers: {missing_layers}"
                )
                layers = missing_layers
        else:
            layers = cfg["layers"]

    if cfg.run_debugging:
        layers = [13]

    dataset = idp.return_processed_test_df()
    start_token = cfg.counter_method["start_token"]
    c = cfg.counter_method["scaler"]

    for layer_id in layers:
        if should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id}")
        else:
            log.warning(f"Skipping layer {layer_id}")
            continue
        # Setup probe and direction
        _runner = runner.load(output_dir=cfg.probe_dir, layer_id=layer_id)
        layer = model.model.layers[layer_id]
        direction = torch.from_numpy(_runner.direction.astype(np.float32)).to(device)
        delta = compute_layer_scale(dh=dh, direction=direction, layer_id=layer_id) * c
        # Compute scale

        # Store results
        RES_orig, RES_neg, RES_pos = [], [], []
        RAND_orig, RAND_neg, RAND_pos = [], [], []

        np.random.seed(cfg.random_seed)
        for n, (i, row) in enumerate(dataset.iterrows()):
            statement = row["statement"]
            proba_orig, proba_neg, proba_pos = [], [], []
            proba_rorig, proba_rneg, proba_rpos = [], [], []

            seq_stats, seq_ans, seq_ans_ids, seq_init_ids = idp.get_answer_seq_ids(
                statement=statement, answer="New York"
            )
            rand_ans_ids = random_answer_ids(
                seq_ids=seq_ans_ids, vocab_size=tokenizer.vocab_size
            )

            for j, st in enumerate(seq_stats[:-1]):
                if j == 0:
                    if start_token > 0:
                        raise ValueError("Start token must be less than 0.")
                    _start_token = len(seq_init_ids) + start_token
                #############################
                ### ORIGINAL VALUES
                with model.trace() as tracer:  # noqa: SIM117
                    with tracer.invoke(st) as _:
                        logits = model.output.logits[0, -1].clone().cpu().save()
                    

                probs = torch.log_softmax(logits, dim=-1)
                output_orig = probs[seq_ans_ids[j]]
                output_rorig = probs[rand_ans_ids[j]]
                proba_orig.append(output_orig)
                proba_rorig.append(output_rorig)
                # log.debug(f"Output Score (orig): {output_orig}")

                #############################
                ### POSITIVE INTERVENTION
                with model.trace() as tracer: # noqa: SIM117
                    with tracer.invoke(st) as _:
                        h = layer.output[0].clone() if isinstance(layer.output, tuple) else layer.output.clone()
                        #log.debug(f"Layer output shape: {h.shape}")

                        h[:, _start_token, :] = translate_concept(
                            h[:, _start_token, :],
                            direction,
                            delta)
                        layer.output[0][:] = h

                        logits = model.output.logits[0, -1].clone().cpu().save() 

                probs = torch.log_softmax(logits, dim=-1)
                output_pos = probs[seq_ans_ids[j]]
                output_rpos = probs[rand_ans_ids[j]]

                # log.debug(f"Output Score (pos): {output_pos}")
                proba_pos.append(output_pos)
                proba_rpos.append(output_rpos)

                #############################
                ### NEGATIVE INTERVENTION
                with model.trace() as tracer:  # noqa: SIM117
                    with tracer.invoke(st) as _:
                        h = layer.output[0].clone() if isinstance(layer.output, tuple) else layer.output.clone()
                        #log.debug(f"Layer output shape: {h.shape}")
                        h[:, _start_token, :] = translate_concept(
                            h[:, _start_token, :],
                            direction,
                            -delta)
                        layer.output[0][:] = h
                        logits = model.output.logits[0, -1].clone().cpu().save()  

                probs = torch.log_softmax(logits, dim=-1)
                output_neg = probs[seq_ans_ids[j]]
                output_rneg = probs[rand_ans_ids[j]]

                # log.debug(f"Output Score (neg): {output_neg}")
                proba_neg.append(output_neg)
                proba_rneg.append(output_rneg)

            # log.debug(f"Statement {i} - Original probs: {proba_orig} - Pos probs: {proba_pos} - Neg probs: {proba_neg}")
            RES_orig.append(mean_logprobs(proba_orig))
            RES_pos.append(mean_logprobs(proba_pos))
            RES_neg.append(mean_logprobs(proba_neg))
            RAND_orig.append(mean_logprobs(proba_rorig))
            RAND_pos.append(mean_logprobs(proba_rpos))
            RAND_neg.append(mean_logprobs(proba_rneg))

            log.debug(
                f"Statement {i} | Pos: {RES_pos[-1]} |  Orig: {RES_orig[-1]}  | Neg: {RES_neg[-1]}"
            )

            if n >= cfg.limit_num_statements:
                break
            clear_device_cache(device)

        # PART II
        RES_neg, RES_orig, RES_pos, RAND_neg, RAND_orig, RAND_pos = map(
            np.array, [RES_neg, RES_orig, RES_pos, RAND_neg, RAND_orig, RAND_pos]
        )

        # Replace NaNs with fallback values
        def fill_nan(arr, fallback, current_layer_id):
            mask = np.isnan(arr)
            if mask.any():
                log.warning(f"Found {mask.sum()} NaNs in layer {current_layer_id}")
                arr = np.where(mask, fallback, arr)
            return arr

        RAND_orig = fill_nan(RAND_orig, RES_orig, layer_id)
        RES_neg = fill_nan(RES_neg, RES_orig, layer_id)
        RES_pos = fill_nan(RES_pos, RES_orig, layer_id)
        RAND_neg = fill_nan(RAND_neg, RAND_orig, layer_id)
        RAND_pos = fill_nan(RAND_pos, RAND_orig, layer_id)

        # Compute diffs
        diff_neg = RES_neg - RES_orig
        diff_pos = RES_pos - RES_orig
        diff_rand_neg = RAND_neg - RAND_orig
        diff_rand_pos = RAND_pos - RAND_orig

        result = diff_of_diff_ols(
            diff_pos=diff_pos,
            diff_neg=diff_neg,
            diff_rand_pos=diff_rand_pos,
            diff_rand_neg=diff_rand_neg,
            dataset=dataset,
        )
        
        success = intervention_success_rate(
            diff_pos=diff_pos,
            diff_neg=diff_neg,
            dataset=dataset,
        )

        # Full summary for the interaction
        log.debug(result.summary())
        log.debug(f"Success rate: {pprint.pformat(success)}")

        save(
            cfg=cfg,
            layer_id=layer_id,
            did_result=result,
            success_results=success,
            s_orig=RES_orig,
            s_neg=RES_neg,
            s_pos=RES_pos,
            r_orig=RAND_orig,
            r_neg=RAND_neg,
            r_pos=RAND_pos,
        )

        db_params = f"Rand OLS: {result.params['Intercept']} Layers: {layer_id/avail_layers[-1]}"
        db_trial_id = f"{cfg.model.name}-{cfg.datapack.name}-{cfg.task}"
        status = 1 if layer_id == avail_layers[-1] else 0
        db.write(
            trial_id=db_trial_id,
            model=cfg.model.name,
            datapack=cfg.datapack.name,
            task=cfg.task,
            parameters=db_params,
            progress=layer_id / avail_layers[-1],
            status=status,
        )

    db_trial_id = f"{cfg.model.name}-{cfg.datapack.name}-{cfg.task}"
    db.write(
        trial_id=db_trial_id,
        model=cfg.model.name,
        datapack=cfg.datapack.name,
        task=cfg.task,
        parameters="Finished",
        progress=1,
        status=1,
    )
    log.warning(
        f"Finished processing layers for {cfg.probe['name']}-based probe for {cfg.model['name']} [task: {cfg.task}]"
    )




def save(
    cfg: DictConfig,
    layer_id: int,
    did_result: sm.regression.linear_model.RegressionResultsWrapper,
    success_results: dict,
    s_orig: np.ndarray,
    s_neg: np.ndarray,
    s_pos: np.ndarray,
    r_orig: np.ndarray,
    r_neg: np.ndarray,
    r_pos: np.ndarray,
):
    """Save difference-in-differences analysis results to disk.

    Extracts coefficients, standard errors, confidence intervals, and p-values
    from the fitted DiD model, computes descriptive statistics, and saves
    everything to JSON. Also saves raw log-probability arrays as .npy files
    for downstream analysis.

    Args:
        cfg: Configuration object with 'output_dir', 'save_results', and 'task'
            attributes.
        layer_id: Index of the transformer layer being analyzed.
        did_result: Fitted OLS model from diff_of_diff_ols() containing DiD
            coefficients and statistics.
        s_orig: Log-probabilities of correct tokens under no intervention.
        s_neg: Log-probabilities of correct tokens under negative intervention.
        s_pos: Log-probabilities of correct tokens under positive intervention.
        r_orig: Log-probabilities of random tokens under no intervention.
        r_neg: Log-probabilities of random tokens under negative intervention.
        r_pos: Log-probabilities of random tokens under positive intervention.

    Side Effects:
        If cfg.save_results is True:
            - Creates cfg.output_dir if it doesn't exist
            - Writes layer_{layer_id}.json with DiD results and descriptives
            - Writes layer_{layer_id}_{sorig,sneg,spos,rorig,rneg,rpos}.npy files
        Always logs results via log.warning.

    Note:
        The JSON output contains two sections:
            - 'did': All coefficients from the 2×2 factorial model, with the
              interaction term ('is_correct_token:is_pos_translation') being
              the primary DiD estimator.
            - 'descriptives': Mean log-probabilities for each condition.

    """
    output = {
        "did": {
            # Intercept (random token, negative intervention baseline)
            "intercept_coef": did_result.params["Intercept"],
            "intercept_std": did_result.bse["Intercept"],
            "intercept_ci": did_result.conf_int().loc["Intercept"].values.tolist(),
            "intercept_pval": did_result.pvalues["Intercept"],
            "intercept_zval": did_result.tvalues["Intercept"],
            # Main effect: correct vs random token
            "token_coef": did_result.params["is_correct_token"],
            "token_std": did_result.bse["is_correct_token"],
            "token_ci": did_result.conf_int().loc["is_correct_token"].values.tolist(),
            "token_pval": did_result.pvalues["is_correct_token"],
            "token_zval": did_result.tvalues["is_correct_token"],
            # Main effect: positive vs negative translation
            "translation_coef": did_result.params["is_pos_translation"],
            "translation_std": did_result.bse["is_pos_translation"],
            "translation_ci": did_result.conf_int()
            .loc["is_pos_translation"]
            .values.tolist(),
            "translation_pval": did_result.pvalues["is_pos_translation"],
            "translation_zval": did_result.tvalues["is_pos_translation"],
            # Interaction (DiD estimator) — the key result
            "interaction_coef": did_result.params[
                "is_correct_token:is_pos_translation"
            ],
            "interaction_std": did_result.bse["is_correct_token:is_pos_translation"],
            "interaction_ci": did_result.conf_int()
            .loc["is_correct_token:is_pos_translation"]
            .values.tolist(),
            "interaction_pval": did_result.pvalues[
                "is_correct_token:is_pos_translation"
            ],
            "interaction_zval": did_result.tvalues[
                "is_correct_token:is_pos_translation"
            ],
            # Model stats
            "r_squared": did_result.rsquared,
            "df_resid": did_result.df_resid,
            "n_obs": int(did_result.nobs),
            "n_statements": int(did_result.nobs / 4),  # 4 obs per statement
            "signf": int(
                did_result.pvalues["is_correct_token:is_pos_translation"] < 0.05
            ),
        },
        "success_results": success_results,
        "descriptives": {
            "correct_orig_mean": float(np.mean(s_orig)),
            "correct_pos_mean": float(np.mean(s_pos)),
            "correct_neg_mean": float(np.mean(s_neg)),
            "random_orig_mean": float(np.mean(r_orig)),
            "random_pos_mean": float(np.mean(r_pos)),
            "random_neg_mean": float(np.mean(r_neg)),
        },
    }

    if os.path.exists(f"{cfg.output_dir}") is False:
        os.makedirs(f"{cfg.output_dir}")

    if cfg.save_results:
        log.warning(f"Saving results for layer {layer_id} for Task {cfg.task}")
        log.warning(pprint.pformat(output))

        score_path = Path(cfg.output_dir) / "scores" / f"layer_{layer_id}"
        score_path.mkdir(parents=True, exist_ok=True)

        np.save(score_path / "s_orig.npy", s_orig)
        np.save(score_path / "s_neg.npy", s_neg)
        np.save(score_path / "s_pos.npy", s_pos)
        np.save(score_path / "r_orig.npy", r_orig)
        np.save(score_path / "r_neg.npy", r_neg)
        np.save(score_path / "r_pos.npy", r_pos)

        summary_path = Path(cfg.output_dir) / f"layer_{layer_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(summary_path, output)
        log.info(f"Saved results to {summary_path}")
    else:
        log.warning(f"Results for layer {layer_id} for Task {cfg.task} (not saved).")
        log.warning(pprint.pformat(output))


if __name__ == "__main__":
    main()
