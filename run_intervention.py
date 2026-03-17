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
    separate_direction_ols,
    intervention_success_rate,
    mean_logprobs,
    random_answer_ids,
    translate_concept,
    check_ols_health,
    test_asymmetry
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
    safe_divide,
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


def _get_logits(output):
    """Extract logits tensor from varied HF/NNSight outputs."""

    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Cannot extract logits from output type {type(output)}")


def _get_hidden(out):
    """Extract hidden-state tensor from layer outputs (handles tuples)."""

    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"Cannot extract hidden state from output type {type(out)}")


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
        if cfg.probe['name'] == 'ttpd':
            direction = _runner.get_truth_direction()
        else:
            direction = _runner.direction.astype(np.float32)
        direction = torch.from_numpy(direction).to(device)
        # Compute scale
        delta, delta_stats = compute_layer_scale(dh=dh, direction=direction, layer_id=layer_id)
        delta = c * delta

        # Store results
        RES_orig, RES_neg, RES_pos = [], [], []
        RAND_orig, RAND_neg, RAND_pos = [], [], []

        np.random.seed(cfg.random_seed)
        # Track how many *correct* statements we've processed so we can stop
        # once we reach `cfg.limit_num_statements` correct examples.
        correct_count = 0
        limit_enabled = cfg.limit_num_statements > 0

        for n, (i, row) in enumerate(dataset.iterrows()):
            statement = row["statement"]
            if row['correct'] == 0:
                log.debug(f"Skipping statement {i} as it is marked incorrect in dataset.")
                RES_orig.append(-100)
                RES_pos.append(-100)
                RES_neg.append(-100)
                RAND_orig.append(-100)
                RAND_pos.append(-100)
                RAND_neg.append(-100)
                continue
            proba_orig, proba_neg, proba_pos = [], [], []
            proba_rorig, proba_rneg, proba_rpos = [], [], []

            seq_stats, seq_ans, seq_ans_ids, seq_init_ids = idp.get_answer_seq_ids(
                statement=statement, answer=row["answer"]
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
                try:
                    with model.trace() as tracer:  # noqa: SIM117
                        with tracer.invoke(st) as _:
                            logits = _get_logits(model.output)[0, -1].clone().cpu().save()
                        
                    probs = torch.log_softmax(logits, dim=-1)
                    output_orig = probs[seq_ans_ids[j]]
                    output_rorig = probs[rand_ans_ids[j]]
                    proba_orig.append(output_orig)
                    proba_rorig.append(output_rorig)
                except (KeyError, SystemError) as e: # catch errors with Gemma-2-9b model
                    log.error(
                        f"Error during original pass at layer {layer_id}, statement {i}, step {j}: {e}"
                    )
                    proba_orig.append(torch.tensor(float(-1e10)))
                    proba_rorig.append(torch.tensor(float(-1e10)))

                #############################
                ### POSITIVE INTERVENTION
                try:
                    with model.trace() as tracer: # noqa: SIM117
                        with tracer.invoke(st) as _:
                            h = _get_hidden(layer.output).clone()
                            #log.debug(f"Layer output shape: {h.shape}")

                            h[:, _start_token, :] = translate_concept(
                                h[:, _start_token, :],
                                direction,
                                delta)
                            layer.output[0][:] = h

                            logits = _get_logits(model.output)[0, -1].clone().cpu().save() 
                
                    probs = torch.log_softmax(logits, dim=-1)
                    output_pos = probs[seq_ans_ids[j]]
                    output_rpos = probs[rand_ans_ids[j]]

                    proba_pos.append(output_pos)
                    proba_rpos.append(output_rpos)
                except (KeyError, SystemError) as e: # catch errors with Gemma-2-9b model
                    log.error(
                        f"Error during positive intervention at layer {layer_id}, statement {i}, step {j}: {e}"
                    )
                    proba_pos.append(torch.tensor(float('nan')))
                    proba_rpos.append(torch.tensor(float('nan')))
                    

                #############################
                ### NEGATIVE INTERVENTION
                try:
                    with model.trace() as tracer:  # noqa: SIM117
                        with tracer.invoke(st) as _:
                            h = _get_hidden(layer.output).clone()
                            #log.debug(f"Layer output shape: {h.shape}")
                            h[:, _start_token, :] = translate_concept(
                                h[:, _start_token, :],
                                direction,
                                -delta)
                            layer.output[0][:] = h
                            logits = _get_logits(model.output)[0, -1].clone().cpu().save()  

                    probs = torch.log_softmax(logits, dim=-1)
                    output_neg = probs[seq_ans_ids[j]]
                    output_rneg = probs[rand_ans_ids[j]]

                    # log.debug(f"Output Score (neg): {output_neg}")
                    proba_neg.append(output_neg)
                    proba_rneg.append(output_rneg)
                except (KeyError, SystemError) as e: # catch errors with Gemma-2-9b model
                    log.error(
                        f"Error during negative intervention at layer {layer_id}, statement {i}, step {j}: {e}"
                    )
                    proba_neg.append(torch.tensor(float('nan')))
                    proba_rneg.append(torch.tensor(float('nan')))

                # Optional debug: warn if interventions did not move logits
                if cfg.run_debugging:
                    delta_pos = (output_pos - output_orig).abs().max().item()
                    delta_neg = (output_neg - output_orig).abs().max().item()
                    if delta_pos == 0.0 and delta_neg == 0.0:
                        log.warning(
                            "No logit change after interventions | "
                            f"layer {layer_id} statement {i} step {j}"
                        )

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

            # Only count statements that were marked correct (we skip incorrect
            # ones above). Stop when we've processed the configured number of
            # correct statements.
            correct_count += 1
            clear_device_cache(device)
            if limit_enabled and correct_count >= cfg.limit_num_statements:
                break

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
        
        asymmetry = test_asymmetry(
            diff_pos=diff_pos,
            diff_neg=diff_neg,
            diff_rand_pos=diff_rand_pos,
            diff_rand_neg=diff_rand_neg,
            dataset=dataset,
            additional_mask=None,
        )
        
        unidir_ols = separate_direction_ols(
            diff_pos=diff_pos,
            diff_neg=diff_neg,
            diff_rand_pos=diff_rand_pos,
            diff_rand_neg=diff_rand_neg,
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
            asymmetry_results=asymmetry,
            unidir_results=unidir_ols,
            delta_stats=delta_stats,
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
    delta_stats: dict,
    success_results: dict,
    asymmetry_results: dict,
    unidir_results: dict,
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
        delta_stats: Dictionary of descriptive statistics about the intervention scale.
        did_result: Fitted OLS model from diff_of_diff_ols() containing DiD
            coefficients and statistics.
        asymmetry_results: Results from asymmetry tests.
        success_results: Results from intervention success rate analysis.
        unidir_results: Results from separate direction OLS analyses.
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
    mask = ~np.isnan(s_orig) & (s_orig != -100)
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
            "token_signf": int(did_result.pvalues["is_correct_token"] < 0.05),
            # Main effect: positive vs negative translation
            "translation_coef": did_result.params["is_pos_translation"],
            "translation_std": did_result.bse["is_pos_translation"],
            "translation_ci": did_result.conf_int()
            .loc["is_pos_translation"]
            .values.tolist(),
            "translation_pval": did_result.pvalues["is_pos_translation"],
            "translation_signf": int(
                did_result.pvalues["is_pos_translation"] < 0.05
            ),
            "translation_zval": did_result.tvalues["is_pos_translation"],
            # Interaction (DiD estimator): the KEY result
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
            "interaction_signf": int(
                did_result.pvalues["is_correct_token:is_pos_translation"] < 0.05
            ),
            # Model stats
            "r_squared": did_result.rsquared,
            "df_resid": did_result.df_resid,
            "n_obs": int(did_result.nobs),
            "n_statements": int(did_result.nobs / 4),  # 4 obs per statement
            "residual_std": float(np.sqrt(did_result.scale)),
            ## Key metric: normalized interaction
            "norm_interaction": float(safe_divide(
                did_result.params["is_correct_token:is_pos_translation"],
                np.sqrt(did_result.scale))),
            "selectivity_ratio": float(safe_divide(
                abs(did_result.params["is_correct_token:is_pos_translation"]),
                abs(did_result.params["is_correct_token"]))),
            "condition_number": float(did_result.condition_number),
            "health": check_ols_health(did_result),
        },
        "success_results": success_results,
        "unidir_results": {
            "positive": {
                "intercept_coef": unidir_results["positive"].params["Intercept"],
                "intercept_std": unidir_results["positive"].bse["Intercept"],
                "intercept_ci": unidir_results["positive"].conf_int().loc["Intercept"].values.tolist(),
                "intercept_pval": unidir_results["positive"].pvalues["Intercept"],
                "intercept_zval": unidir_results["positive"].tvalues["Intercept"],
                "token_coef": unidir_results["positive"].params["is_correct_token"],
                "token_std": unidir_results["positive"].bse["is_correct_token"],
                "token_ci": unidir_results["positive"].conf_int().loc["is_correct_token"].values.tolist(),
                "token_pval": unidir_results["positive"].pvalues["is_correct_token"],
                "token_zval": unidir_results["positive"].tvalues["is_correct_token"],
                "token_signf": int(unidir_results["positive"].pvalues["is_correct_token"] < 0.05),
                "r_squared": unidir_results["positive"].rsquared,
                "df_resid": unidir_results["positive"].df_resid,
                "n_obs": int(unidir_results["positive"].nobs),
                "condition_number": float(unidir_results["positive"].condition_number),
            },
            "negative": {
                "intercept_coef": unidir_results["negative"].params["Intercept"],
                "intercept_std": unidir_results["negative"].bse["Intercept"],
                "intercept_ci": unidir_results["negative"].conf_int().loc["Intercept"].values.tolist(),
                "intercept_pval": unidir_results["negative"].pvalues["Intercept"],
                "intercept_zval": unidir_results["negative"].tvalues["Intercept"],
                "token_coef": unidir_results["negative"].params["is_correct_token"],
                "token_std": unidir_results["negative"].bse["is_correct_token"],
                "token_ci": unidir_results["negative"].conf_int().loc["is_correct_token"].values.tolist(),
                "token_pval": unidir_results["negative"].pvalues["is_correct_token"],
                "token_zval": unidir_results["negative"].tvalues["is_correct_token"],
                "token_signf": int(unidir_results["negative"].pvalues["is_correct_token"] < 0.05),
                "r_squared": unidir_results["negative"].rsquared,
                "df_resid": unidir_results["negative"].df_resid,
                "n_obs": int(unidir_results["negative"].nobs),
                "condition_number": float(unidir_results["negative"].condition_number),
            },
            "asymmetry_ratio": safe_divide(
                unidir_results["positive"].params["is_correct_token"],
                unidir_results["negative"].params["is_correct_token"],
            ),
            "signf": int(
                unidir_results["positive"].pvalues["is_correct_token"] < 0.05
                and unidir_results["negative"].pvalues["is_correct_token"] < 0.05
            ),  # both directions significant
        },
        "descriptives": {
            "correct_orig_mean": float(np.mean(s_orig[mask])),
            "correct_pos_mean": float(np.mean(s_pos[mask])),
            "correct_neg_mean": float(np.mean(s_neg[mask])),
            "random_orig_mean": float(np.mean(r_orig[mask])),
            "random_pos_mean": float(np.mean(r_pos[mask])),
            "random_neg_mean": float(np.mean(r_neg[mask])),
        },
        "delta_stats": delta_stats,
        "asymmetry": asymmetry_results,
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
        np.save(score_path / "mask.npy", mask)

        summary_path = Path(cfg.output_dir) / f"layer_{layer_id}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(summary_path, output)
        log.info(f"Saved results to {summary_path}")
    else:
        log.warning(f"Results for layer {layer_id} for Task {cfg.task} (not saved).")
        log.warning(pprint.pformat(output))


if __name__ == "__main__":
    main()
