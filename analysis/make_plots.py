"""Generate summary figures from probe metrics.

This CLI ports core logic from make_plots.ipynb into modular functions.

Args:
    None

Returns:
    None

Example:
    Run from the repo root:
    $ python -m analysis.make_plots
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd

from analysis.plots.constants import (
    DATAPACKS,
    MODEL_NAMES,
    PROBES,
    SAVE_DIR,
    TASKS,
)
from misc.probe_data import ExperimentData

from .plots.causal import (
    plot_causal_success_average,
    plot_causal_success_by_dataset,
    plot_causal_success_interaction_2d,
    plot_causal_interaction_average,
    plot_causal_interaction_by_dataset,
)
from .plots.performance import (
    plot_generalization_by_condition,
    plot_generalization_by_dataset,
    plot_performance_by_condition,
    plot_performance_by_dataset,
)

WITH_SEARCH = True
LOGGER = logging.getLogger(__name__)

def collect_performance_df() -> pd.DataFrame:
    """Collect per-run metrics across models, datapacks, and probes.

    For non-zero-shot probes, selects the best layer and reads metrics for
    conditions. For zero-shot, reads metrics.json directly.

    Returns:
        pd.DataFrame: Columns ['model','condition','datapack','probe','subexperiment',
        'CI_low','mcc','CI_high'].
    """
    keys = ["conformal", "mcc"]
    result: list[tuple] = []
    for model, _ in MODEL_NAMES.items():
        for datapack in DATAPACKS:
            for probe in PROBES:
                if probe == "zero_shot":
                    metrics_path = (
                        Path("outputs/probes/zero_shot/")
                        / model
                        / datapack
                        / "metrics.json"
                    )
                    with metrics_path.open("r", encoding="utf-8") as handle:
                        values = json.load(handle)[keys[0]][keys[1]]
                    for cond in ["bag", "instance", "instance_tf"]:
                        result.append(
                            (
                                model,
                                cond,
                                datapack,
                                probe,
                                "zero_shot",
                                values[1],
                                values[0],
                                values[2],
                            )
                        )
                    continue

                eD = ExperimentData(
                    model_name=model,
                    dataset_name=datapack,
                    task=TASKS[probe],
                    probe_name=probe,
                    with_search=WITH_SEARCH,
                )
                for cond in ["bag", "instance", "instance_tf"]:
                    try:
                        if probe == "svm":
                            key_set = [cond] + ["default", "mcc"]
                        # we compare sAwMIL to base SVM without any modifications
                        else:
                            key_set = [cond] + keys
                        subexperiment = "g_" + datapack
                        lid = eD.best_layer(
                            keys=key_set, path=eD.base_path / subexperiment
                        )
                        values = eD.read_metrics(
                            layer_id=lid,
                            keys=key_set,
                            path=eD.base_path / subexperiment,
                        )
                        result.append(
                            (
                                model,
                                cond,
                                datapack,
                                probe,
                                subexperiment,
                                values[1],
                                values[0],
                                values[2],
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Skipping metrics for model=%s datapack=%s probe=%s "
                            "cond=%s subexp=%s: %s",
                            model,
                            datapack,
                            probe,
                            cond,
                            subexperiment,
                            exc,
                        )
                        continue
    df = pd.DataFrame(
        result,
        columns=[
            "model",
            "condition",
            "datapack",
            "probe",
            "subexperiment",
            "CI_low",
            "mcc",
            "CI_high",
        ],
    )
    return df


def collect_intervention_success(root: Path | None = None) -> pd.DataFrame:
    """Collect best-layer intervention success metrics.

    Traverses ``outputs/interv`` (or ``root`` when provided) to identify the
    highest-success layer per probe/model/dataset combination. Ties in
    ``success_rate`` are broken by selecting the smallest layer index.

    Args:
        root: Optional override for the intervention output directory.

    Returns:
        DataFrame with one row per best layer containing columns ``probe``,
        ``model``, ``datapack``, ``best_layer``,
        ``success_rate``

    Raises:
        FileNotFoundError: If the intervention directory is missing.
    """

    base_dir = Path(root) if root is not None else Path("outputs/interv")
    if not base_dir.exists():
        raise FileNotFoundError(f"Intervention output directory missing: {base_dir}")

    records: list[dict] = []
    for probe_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        probe = probe_dir.name
        for model_dir in sorted(m for m in probe_dir.iterdir() if m.is_dir()):
            model = model_dir.name
            for dataset_dir in sorted(d for d in model_dir.iterdir() if d.is_dir()):
                best_layer: int | None = None
                best_success: float = float("-inf")
                best_payload: dict | None = None

                for layer_path in sorted(dataset_dir.glob("layer_*.json")):
                    try:
                        layer_idx = int(layer_path.stem.split("_")[1])
                    except (IndexError, ValueError):
                        continue

                    with layer_path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)

                    success_payload = payload.get("success_results") or {}
                    success_rate = success_payload.get("success_rate")
                    if success_rate is None:
                        continue
                    
                    interaction_coeff = payload.get("did", {}).get("interaction_coef")

                    is_better = success_rate > best_success
                    is_tie = math.isclose(success_rate, best_success)
                    if (
                        best_layer is None
                        or is_better
                        or (is_tie and layer_idx < best_layer)
                    ):
                        best_layer = layer_idx
                        best_success = success_rate
                        best_interaction_coeff = abs(interaction_coeff)
                        best_payload = success_payload

                if best_layer is None or best_payload is None:
                    continue

                dataset = dataset_dir.name.split("_search_task")[0]
                records.append(
                    {
                        "probe": probe,
                        "model": model,
                        "datapack": dataset,
                        "best_layer": best_layer,
                        "success_rate": best_success,
                        "interaction_coef": best_interaction_coeff,
                    }
                )

    df = pd.DataFrame(records)

    if df.empty:
        return df

    df.sort_values(
        ["probe", "model", "datapack", "success_rate"], inplace=True, ascending=False
    )
    df.reset_index(drop=True, inplace=True)
    return df

def collect_generalization_df() -> pd.DataFrame:
    """Collect cross-dataset generalization metrics.

    Mirrors the notebook logic: iterates over models, source datapacks, and probes;
    evaluates across the configured generalization subexperiments, and records
    per-condition MCC scores with confidence bounds.

    Returns:
        pd.DataFrame: Columns ['model','condition','datapack','probe','subexperiment',
        'CI_low','mcc','CI_high'].
    """

    keys = ["conformal", "mcc"]
    generalization_targets = [
        "g_city_locations",
        "g_word_definitions",
        "g_med_indications",
    ]
    relevant_probes = [probe for probe in PROBES if probe != "zero_shot"]

    result: list[tuple] = []
    for model, _ in MODEL_NAMES.items():
        for datapack in DATAPACKS:
            for probe in relevant_probes:
                exp_data = ExperimentData(
                    model_name=model,
                    dataset_name=datapack,
                    task=TASKS[probe],
                    probe_name=probe,
                    with_search=WITH_SEARCH,
                )
                for subexperiment in generalization_targets:
                    for cond in ["bag", "instance", "instance_tf"]:
                        if probe == "svm":
                            key_set = [cond, "default", "mcc"]
                        else:
                            key_set = [cond] + keys
                        layer_id = exp_data.best_layer(
                            keys=key_set,
                            path=exp_data.base_path / subexperiment,
                        )
                        values = exp_data.read_metrics(
                            layer_id=layer_id,
                            keys=key_set,
                            path=exp_data.base_path / subexperiment,
                        )
                        result.append(
                            (
                                model,
                                cond,
                                datapack,
                                probe,
                                subexperiment,
                                values[1],
                                values[0],
                                values[2],
                            )
                        )

    return pd.DataFrame(
        result,
        columns=[
            "model",
            "condition",
            "datapack",
            "probe",
            "subexperiment",
            "CI_low",
            "mcc",
            "CI_high",
        ],
    )


def main() -> None:
    """Generate the performance-by-dataset figure.

    Aggregates metrics and saves a grouped bar plot under outputs/figures.
    """

    #### Intervention plots
    print("Collecting intervention data...")
    df_success = collect_intervention_success()

    print("Generating intervention-success-by-dataset plot...")
    output_path = plot_causal_success_by_dataset(
        df_success=df_success, save_dir=SAVE_DIR
    )
    print(f"\tSaved causal-success-by-dataset plot to {output_path}")

    print("Generating intervention-success-average plot...")
    output_path = plot_causal_success_average(df_success=df_success, save_dir=SAVE_DIR)
    print(f"\tSaved causal-success-average plot to {output_path}")
    
    output_path = plot_causal_success_interaction_2d(
        df_success=df_success, save_dir=SAVE_DIR
    )
    print(f"\tSaved causal-success-interaction-2d plot to {output_path}")
    
    output_path = plot_causal_interaction_average(
        df_success=df_success, save_dir=SAVE_DIR
    )
    print(f"Saved causal-interaction-average plot to {output_path}")
    
    output_path = plot_causal_interaction_by_dataset(
        df_success=df_success, save_dir=SAVE_DIR
    )
    print(f"Saved causal-interaction-by-dataset plot to {output_path}")

    #### Performance plots
    print("Collecting performance data...")
    df = collect_performance_df()
    print(f"\tCollected {len(df)} rows across {df['datapack'].nunique()} datapacks")

    print("Generating performance-by-dataset plot...")
    output_path = plot_performance_by_dataset(df_results=df, save_dir=SAVE_DIR)
    print(f"\tSaved performance-by-dataset plot to {output_path}")

    print("Generating performance-by-condition plot...")
    output_path = plot_performance_by_condition(df_results=df, save_dir=SAVE_DIR)
    print(f"\tSaved performance-by-condition plot to {output_path}")

    #### Generalization plots
    print("Collecting generalization data...")
    df_generalization = collect_generalization_df()

    print("Generating generalization-by-dataset plot...")
    output_path = plot_generalization_by_dataset(
        df_results=df_generalization, save_dir=SAVE_DIR
    )
    print(f"\tSaved generalization-by-dataset plot to {output_path}")

    print("Generating generalization-by-condition plot...")
    output_path = plot_generalization_by_condition(
        df_results=df_generalization, save_dir=SAVE_DIR
    )

    print(f"\tSaved generalization-by-condition plot to {output_path}")


if __name__ == "__main__":
    main()
