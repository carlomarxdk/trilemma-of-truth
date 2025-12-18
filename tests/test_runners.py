from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runners.runner_md import MDProbeRunner
from runners.runner_sawmil import SawmilProbeRunner
from runners.runner_spca import SPCA_Runner
from runners.runner_svm import SVMProbeRunner
from runners.runner_ttpd import TTPD_Runner
from utils_jupyter import load_hydra_config_with_params


@pytest.mark.parametrize(
    "RunnerClass,bagged,task, probe",
    [
        (SVMProbeRunner, False, 0, "svm"),
        (MDProbeRunner, False, 3, "mean_diff"),
        (SawmilProbeRunner, True, 0, "sawmil"),
        (SPCA_Runner, False, 3, "spca"),
        (TTPD_Runner, False, 3, "ttpd"),
    ],
)
def test_runner_predictions_have_output_and_consistency(
    RunnerClass, bagged, task, probe
):
    """Test that each runner produces output and (optionally) consistent predict/inst_predict results."""
    MODEL_NAME = "llama-3-8b"
    DATASET_NAME = "cities_loc"

    output_dir = (
        Path("outputs")
        / "probes"
        / probe
        / MODEL_NAME
        / f"{ DATASET_NAME }_search_task-{str(task)}"
    )

    # Skip test if no trained outputs are available
    if not output_dir.exists() or not any(output_dir.glob("*")):
        pytest.skip(f"Skipping {RunnerClass.__name__}: no files found in {output_dir}")

    cfg = load_hydra_config_with_params(
        model=MODEL_NAME,
        probe=probe,
        datapack=DATASET_NAME,
        task=task,
        config_name="probe_training",
    )

    layer_id = 10
    runner = RunnerClass(cfg=cfg).load(output_dir=output_dir, layer_id=layer_id)

    # Make dummy bag
    bag = [np.random.randn(30, 4096) for _ in range(5)]

    # Run predictions
    bag_preds = runner.bag_predict(bag)
    preds = runner.predict(bag)
    inst_preds = runner.inst_predict(bag)

    # 1. Check that all predictions have non-empty output
    assert (
        bag_preds is not None and len(bag_preds) > 0
    ), f"{RunnerClass.__name__}: bag_predict returned empty"
    assert (
        preds is not None and len(preds) > 0
    ), f"{RunnerClass.__name__}: predict returned empty"
    assert (
        inst_preds is not None and len(inst_preds) > 0
    ), f"{RunnerClass.__name__}: inst_predict returned empty"

    # 2. Optionally check equality between predict and inst_predict
    if bagged:
        np.testing.assert_allclose(
            preds, bag_preds, err_msg=f"{RunnerClass.__name__}: predict != bag_predict"
        )
    else:
        np.testing.assert_allclose(
            preds,
            inst_preds,
            err_msg=f"{RunnerClass.__name__}: predict != inst_predict",
        )
