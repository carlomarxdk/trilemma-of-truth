from __future__ import annotations

import json
import logging
import os
import pprint
from glob import glob

import hydra
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import torch
from omegaconf import DictConfig, OmegaConf

from misc.db import LogDataBase
from misc.probe_data import ProbeData
from response.interventions_utils import (
    InstructInterventionDataProcessor,
    InterventionDataProcessor,
    to_log_proba,
    translate_concept,
)
from utils import should_process_layer
from utils_hydra import (
    NpEncoder,
    clear_device_cache,
    get_device,
    load_data,
    prepare_nnsight,
)

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig):
    assert (
        type(cfg.datapack["datasets"]) == list
        or type(cfg.datapack["datasets"]).__name__ == "ListConfig"
    ), f"Datasets must be a list. Not {type(cfg.datapack['datasets'])}"
    assert len(cfg.datapack["datasets"]) > 0, "At least one dataset must be selected."
    OmegaConf.set_struct(cfg, False)  # Allow overriding
    trial_name = cfg.trial_name
    if cfg.probe["sparsify_data"] > 0:
        trial_name += f"_sparse-{cfg.probe['sparsify_data']}"
    if cfg.search:
        trial_name += "_search"
    trial_name += f"_task-{cfg.task}"
    cfg["trial_name"] = trial_name
    cfg["output_dir"] = os.path.join(cfg.output_dir, trial_name)
    if cfg.device == None:
        cfg["device"] = str(get_device())
    OmegaConf.set_struct(cfg, True)

    assert len(cfg.layer_range) == 2, "Layer range must be a list of two integers."


def log_stats(cfg):
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


def checkpointing(cfg, available_layers):
    output_dir = cfg.output_dir
    completed_layers = set()

    # Check each layer if it has files for all three required signatures.
    for layer in available_layers:
        # Construct the file patterns for the current layer
        layer_pattern = os.path.join(output_dir, f"layer_{layer}.*")
        spos_pattern = os.path.join(output_dir, f"layer_{layer}_spos.*")
        rpos_pattern = os.path.join(output_dir, f"layer_{layer}_rpos.*")
        # Check that files exist for each pattern
        if glob(layer_pattern) and glob(spos_pattern) and glob(rpos_pattern):
            completed_layers.add(layer)

    # Determine missing layers as those in available_layers not in completed_layers
    missing_layers = sorted(set(available_layers) - completed_layers)
    return list(missing_layers)


@hydra.main(version_base=None, config_path="configs", config_name="intervention_linear")
def main(cfg: OmegaConf):
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
    # PER LAYER
    probe_path = f"outputs/probes/{cfg.probe['name']}/{cfg.model['name']}/{cfg.datapack['name']}_search_task-{cfg.task}"
    # reader = ProbeData(output_dir=cfg.output_dir, task=cfg['task'], model_name=cfg.model["name"],
    #                       datapack=cfg.datapack['name'], trial_name=cfg.trial_name, probe_name=cfg.probe["name"])
    reader = ProbeData(probe_path)
    device = torch.device(cfg.device)

    model, tokenizer = prepare_nnsight(cfg)
    dh = load_data(cfg)
    # dh_test = load_data(cfg) if cfg.datapack['datasets_test'] else dh

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

    if cfg.start_from_checkpoint:
        log.warning("Checkpointing...")
        layers = checkpointing(cfg, reader.available_layers())

    else:
        layers = reader.available_layers()

    dataset = idp.return_processed_test_df()
    start_token = cfg.counter_method["start_token"]
    absolute = cfg.counter_method["absolute"]
    assert absolute, 'This experiment supports only "absolute" method of translation.'
    c = cfg.counter_method["target_coord"]

    print(layers)
    for layer_id in layers:
        if should_process_layer(layer_id, cfg):
            log.warning(f"Processing layer {layer_id}")
        else:
            log.warning(f"Skipping layer {layer_id}")
            continue
        layer = model.model.layers[layer_id]
        direction = reader.direction(layer_id, as_tensor=True).half().to(device)
        # Store results
        RES_pos = []
        RAND_pos = []
        RES_neg = []
        RAND_neg = []
        RES_orig = []
        RAND_orig = []

        n = 0
        for i, row in dataset.iterrows():
            statement = row["statement"]
            answer = row["answer"]
            proba_orig, proba_neg, proba_pos = [], [], []
            proba_rorig, proba_rneg, proba_rpos = [], [], []

            seq_stats, seq_ans, seq_ans_ids, seq_init_ids = idp.get_answer_seq_ids(
                statement=statement, answer=answer
            )
            rand_ans_ids = np.random.choice(
                tokenizer.vocab_size, len(seq_ans_ids), replace=False
            )

            for j, st in enumerate(seq_stats[:-1]):
                if j == 0:
                    assert start_token <= 0, "Start token must be less than 0."
                    _start_token = len(seq_init_ids) + start_token
                with model.trace() as tracer, tracer.invoke(st) as _:
                    _output_orig = model.output["logits"][0, -1].cpu().save()

                output_orig = torch.softmax(_output_orig, dim=0)[seq_ans_ids[j]]
                output_rorig = torch.softmax(_output_orig, dim=0)[
                    rand_ans_ids[j]
                ].unsqueeze(0)
                proba_orig.append(output_orig)
                proba_rorig.append(output_rorig)
                with model.trace() as tracer, tracer.invoke(st) as _:
                    output = layer.output
                    _output = output[0].clone()
                    _output[0, _start_token:] = translate_concept(
                        _output, direction, c, absolute=absolute
                    )[0, _start_token:]
                    # layer.output = (_output,)
                    if "stablelm" in cfg["model"]["name"]:
                        layer.output = (_output, output[1])
                    else:
                        layer.output = (_output,)
                    _output_pos = model.lm_head.output[0, -1].cpu().save()
                output_pos = torch.softmax(_output_pos, dim=0)[seq_ans_ids[j]]
                outpit_rpos = torch.softmax(_output_pos, dim=0)[rand_ans_ids[j]]
                # print(output_pos)
                proba_pos.append(output_pos)
                proba_rpos.append(outpit_rpos)
                with model.trace() as tracer, tracer.invoke(st) as _:
                    output = layer.output
                    _output = output[0].clone()
                    _output[0, _start_token:] = translate_concept(
                        _output, direction, -c, absolute=absolute
                    )[0, _start_token:]
                    # layer.output = (_output,)
                    if "stablelm" in cfg["model"]["name"]:
                        layer.output = (_output, output[1])
                    else:
                        layer.output = (_output,)
                    _output_neg = model.lm_head.output[0, -1].cpu().save()
                output_neg = torch.softmax(_output_neg, dim=0)[seq_ans_ids[j]]
                output_rneg = torch.softmax(_output_neg, dim=0)[rand_ans_ids[j]]
                proba_rneg.append(output_rneg)
                proba_neg.append(output_neg)
                # print(output_neg)

            RES_orig.append(to_log_proba(proba_orig))
            RES_pos.append(to_log_proba(proba_pos))
            RES_neg.append(to_log_proba(proba_neg))
            RAND_orig.append(to_log_proba(proba_rorig))
            RAND_pos.append(to_log_proba(proba_rpos))
            RAND_neg.append(to_log_proba(proba_rneg))

            if n > cfg.limit_num_statements:
                break
            n += 1
            clear_device_cache(device)

        # PART II
        RES_neg = np.array(RES_neg)
        RES_orig = np.array(RES_orig)
        RES_pos = np.array(RES_pos)
        RAND_neg = np.array(RAND_neg)
        RAND_orig = np.array(RAND_orig)
        RAND_pos = np.array(RAND_pos)

        if np.isnan(RES_neg).any() or np.isnan(RES_pos).any():
            log.warning(f"Found NaNs in the results for layer {layer_id}")

            for i in range(RES_neg.shape[0]):
                if np.isnan(RES_neg[i]):
                    RES_neg[i] = RES_orig[i]
                if np.isnan(RES_pos[i]):
                    RES_pos[i] = RES_orig[i]

        if (
            np.isnan(RAND_neg).any()
            or np.isnan(RAND_pos).any()
            or np.isnan(RAND_orig).any()
        ):
            log.warning(f"Found NaNs in the results for layer {layer_id}")

            for i in range(RAND_neg.shape[0]):
                if np.isnan(RAND_neg[i]):
                    RAND_neg[i] = RAND_orig[i]
                if np.isnan(RAND_pos[i]):
                    RAND_pos[i] = RAND_orig[i]
                if np.isnan(RAND_orig[i]):
                    RAND_orig[i] = RES_orig[i]

        diff_neg = RES_neg - RES_orig
        diff_pos = RES_pos - RES_orig

        diff_rand_neg = RAND_neg - RAND_orig
        diff_rand_pos = RAND_pos - RAND_orig

        ols_res = diff_ols(diff_pos, diff_neg, dataset)
        ols_res_rand = diff_ols_rand(
            diff_pos, diff_neg, diff_rand_pos, diff_rand_neg, dataset
        )
        ttest_res = diff_ttest(diff_pos, diff_neg, dataset, task=cfg.task)

        save(
            cfg=cfg,
            layer_id=layer_id,
            ols=ols_res,
            rand_ols=ols_res_rand,
            ttest=ttest_res,
            s_orig=RES_orig,
            s_neg=RES_neg,
            s_pos=RES_pos,
            r_orig=RAND_orig,
            r_neg=RAND_neg,
            r_pos=RAND_pos,
        )

        db_params = f"Rand OLS: {ols_res_rand.params['Intercept']} Layers: {layer_id/reader.available_layers()[-1]}"
        db_trial_id = f"{cfg.model.name}-{cfg.datapack.name}-{cfg.task}"
        status = 1 if layer_id == reader.available_layers()[-1] else 0
        db.write(
            trial_id=db_trial_id,
            model=cfg.model.name,
            datapack=cfg.datapack.name,
            task=cfg.task,
            parameters=db_params,
            progress=layer_id / reader.available_layers()[-1],
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


def diff_ols(diff_pos, diff_neg, dataset):
    """Return the OLS regression for real statements only.
    Hypothesis: Diff_pos > Diff_neg"""
    N = diff_pos.shape[0]
    y = dataset["correct"].values[:N]
    r = dataset["real_object"].values[:N]
    mask = r == 1
    diff = diff_pos - diff_neg
    df = pd.DataFrame({"diff": diff[mask], "group": y[mask]})
    model = smf.ols("diff ~ group", data=df)
    return model.fit(cov_type="HC3")


def diff_ols_rand(diff_pos, diff_neg, diff_rand_pos, diff_rand_neg, dataset):
    """Return the OLS regression for real and true statements only.'
    'Hypothesis: Diff_true > Diff_random"""
    N = diff_pos.shape[0]
    y = dataset["correct"].values[:N]
    r = dataset["real_object"].values[:N]
    mask = (r == 1) & (y == 1)
    diff = diff_pos - diff_neg
    df1 = pd.DataFrame({"diff": diff[mask], "group": 1})
    diff_rand = diff_rand_pos - diff_rand_neg
    df2 = pd.DataFrame({"diff": diff_rand[mask], "group": 0})
    df = pd.concat([df1, df2], ignore_index=True)

    model = smf.ols("diff ~ group", data=df)
    return model.fit(cov_type="HC3")


def diff_ttest(diff_pos, diff_neg, dataset, task):
    """Return the t-test stats for real statements only."""
    N = diff_pos.shape[0]
    y = dataset["correct"].values[:N]
    r = dataset["real_object"].values[:N]
    mask = (r == 1) & (y == 1)
    if task in [0, 4]:
        result = stats.ttest_rel(diff_pos[mask], diff_neg[mask], alternative="greater")
    elif task in [1, 5]:
        result = stats.ttest_rel(diff_pos[mask], diff_neg[mask], alternative="less")
    return result


def save(
    cfg, layer_id, ols, rand_ols, ttest, s_orig, s_neg, s_pos, r_orig, r_neg, r_pos
):
    output = {
        "ols": {
            "intercept_coef": ols.params["Intercept"],
            "intercept_std": ols.bse["Intercept"],
            "intercept_ci": ols.conf_int().loc["Intercept"].values,
            "intercept_pval": ols.pvalues["Intercept"],
            "intercept_zval": ols.tvalues["Intercept"],
            "group_coef": ols.params["group"],
            "group_std": ols.bse["group"],
            "group_ci": ols.conf_int().loc["group"].values,
            "group_pval": ols.pvalues["group"],
            "group_zval": ols.tvalues["group"],
            "df": ols.df_resid,
            "n_inst": ols.nobs,
        },
        "rand_ols": {
            "intercept_coef": rand_ols.params["Intercept"],
            "intercept_std": rand_ols.bse["Intercept"],
            "intercept_ci": rand_ols.conf_int().loc["Intercept"].values,
            "intercept_pval": rand_ols.pvalues["Intercept"],
            "intercept_zval": rand_ols.tvalues["Intercept"],
            "group_coef": rand_ols.params["group"],
            "group_std": rand_ols.bse["group"],
            "group_ci": rand_ols.conf_int().loc["group"].values,
            "group_pval": rand_ols.pvalues["group"],
            "group_zval": rand_ols.tvalues["group"],
            "df": rand_ols.df_resid,
            "n_inst": rand_ols.nobs,
            "signf": int(rand_ols.pvalues["group"] < 0.05),
        },
        "ttest": {
            "statistic": ttest.statistic,
            "pvalue": ttest.pvalue,
            "df": ttest.df,
            "signf": int(ttest.pvalue < 0.05),
        },
    }

    if os.path.exists(f"{cfg.output_dir}") is False:
        os.makedirs(f"{cfg.output_dir}")

    if cfg.save_results:
        log.warning(f"Saving results for layer {layer_id} for Task {cfg.task}")
        log.warning(pprint.pformat(output))
        with open(f"{cfg.output_dir}/layer_{layer_id}.json", "w") as f:
            json.dump(output, f, cls=NpEncoder)

        np.save(f"{cfg.output_dir}/layer_{layer_id}_sorig.npy", s_orig)
        np.save(f"{cfg.output_dir}/layer_{layer_id}_sneg.npy", s_neg)
        np.save(f"{cfg.output_dir}/layer_{layer_id}_spos.npy", s_pos)
        np.save(f"{cfg.output_dir}/layer_{layer_id}_rorig.npy", r_orig)
        np.save(f"{cfg.output_dir}/layer_{layer_id}_rneg.npy", r_neg)
        np.save(f"{cfg.output_dir}/layer_{layer_id}_rpos.npy", r_pos)
    else:
        log.warning(f"Results for layer {layer_id} for Task {cfg.task} (not saved).")
        log.warning(pprint.pformat(output))


if __name__ == "__main__":
    main()
