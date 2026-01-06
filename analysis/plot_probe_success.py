"""Aggregate intervention success metrics and plot per-probe performance.

This script gathers intervention outputs stored under ``outputs/interv``, selects the
best-performing layer per probe/model/dataset combination, and produces a bar plot
of average success rates per probe with individual experiment points overlaid.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from matplotlib.ticker import PercentFormatter

# Model, datapack, and probe configuration
MODELS = [
    'qwen-2.5-14b',
    'qwen-2.5-7b',
    'mistral-7B-v0.3',
    'gemma-2-9b',
    'gemma-7b',
    'llama-3-8b',
    'llama-3.2-3b',
    '_qwen-2.5-14b',
    '_qwen-2.5-7b',
    '_mistral-7B-v0.3',
    '_gemma-2-9b',
    '_gemma-7b',
    '_llama-3.1-8b',
    '_llama-3-8b-med',
    '_llama-3.1-8b-bio',
    '_llama-3.2-3b',
]

DATAPACKS = ['city_locations', 'med_indications', 'word_definitions']
PROBES = ['zero_shot', 'mean_diff', 'ttpd', 'spca', 'svm', 'sawmil']

TASKS = {
    'mean_diff': 3,
    'ttpd': 3,
    'spca': 3,
    'sawmil': -1,
    'svm': -1,
}

MODEL_NAMES = {
    'llama-3-8b': 'Llama-3-8B',
    'llama-3.2-3b': 'Llama-3.2-3B',
    'mistral-7B-v0.3': 'Mistral-7B-v0.3',
    'qwen-2.5-7b': 'Qwen-2.5-7B',
    'qwen-2.5-14b': 'Qwen-2.5-14B',
    'gemma-7b': 'Gemma-7B',
    'gemma-2-9b': 'Gemma-2-9B',
    '_llama-3.1-8b': 'Llama-3.1-8B-Instruct',
    '_llama-3.2-3b': 'Llama-3.2-3B-Instruct',
    '_mistral-7B-v0.3': 'Mistral-7B-Instruct-v0.3',
    '_qwen-2.5-7b': 'Qwen-2.5-7B-Instruct',
    '_qwen-2.5-14b': 'Qwen-2.5-14B-Instruct',
    '_gemma-7b': 'Gemma-7B-it',
    '_gemma-2-9b': 'Gemma-2-9B-it',
    '_llama-3.1-8b-bio': 'Bio-Medical-Llama-3-8B',
    '_llama-3-8b-med': 'Llama3-Med42-8B',
}

MODEL_SHORTNAMES = {
    'llama-3-8b': 'Llama-3-8B',
    'llama-3.1-8b': 'Llama-3.1-8B',
    'llama-3.2-3b': 'Llama-3.2-3B',
    'mistral-7B-v0.3': 'Mistral-7B-v0.3',
    'qwen-2.5-7b': 'Qwen-2.5-7B',
    'qwen-2.5-14b': 'Qwen-2.5-14B',
    'gemma-7b': 'Gemma-7B',
    'gemma-2-9b': 'Gemma-2-9B',
    '_llama-3.1-8b': 'Llama-3.1-8B',
    '_llama-3.2-3b': 'Llama-3.2-3B',
    '_mistral-7B-v0.3': 'Mistral-7B-v0.3',
    '_qwen-2.5-7b': 'Qwen-2.5-7B',
    '_qwen-2.5-14b': 'Qwen-2.5-14B',
    '_gemma-7b': 'Gemma-7B',
    '_gemma-2-9b': 'Gemma-2-9B',
    '_llama-3.1-8b-bio': 'Bio-Medical-Llama',
    '_llama-3-8b-med': 'Llama3-Med42-8B',
}

MODEL_TYPES = {
    'llama-3-8b': 'default',
    'llama-3.1-8b': 'default',
    'llama-3.2-3b': 'default',
    'mistral-7B-v0.3': 'default',
    'qwen-2.5-7b': 'default',
    'qwen-2.5-14b': 'default',
    'gemma-7b': 'default',
    'gemma-2-9b': 'default',
    '_llama-3.1-8b': 'chat',
    '_llama-3.2-3b': 'chat',
    '_mistral-7B-v0.3': 'chat',
    '_qwen-2.5-7b': 'chat',
    '_qwen-2.5-14b': 'chat',
    '_gemma-7b': 'chat',
    '_gemma-2-9b': 'chat',
    '_llama-3.1-8b-bio': 'chat',
    '_llama-3-8b-med': 'chat',
}

DATASET_NAMES = {
    'city_locations': 'City Locations',
    'med_indications': 'Medical Indications',
    'word_definitions': 'Word Definitions',
}

CONDITION_NAMES = {
    'bag': 'Bag-Level',
    'instance': 'Instance-Level',
    'instance_tf': 'Instance-Level (TF Only)',
}

PROBE_ORDER = ['zero_shot', 'mean_diff', 'ttpd', 'spca', 'svm', 'sawmil']
DATAPACK_ORDER = ['city_locations', 'med_indications', 'word_definitions']

RESULTS_ROOT = Path("outputs/interv")
FIGURE_PATH = Path("outputs/figures/probe_success_best_layer.png")
DETAIL_CSV_PATH = Path("outputs/figures/probe_success_by_model_dataset.csv")
SUMMARY_CSV_PATH = Path("outputs/figures/probe_success_best_layer_summary.csv")
FOREST_FIGURE_PATH = Path("outputs/figures/probe_interaction_forest.png")
INTER_BAR_FIGURE_PATH = Path("outputs/figures/probe_interaction_bar.png")


def _strip_task_suffix(dataset_name: str) -> str:
    """Drop trailing task identifiers to recover the base dataset name."""

    split_token = "_search_task"
    if split_token in dataset_name:
        return dataset_name.split(split_token, maxsplit=1)[0]
    return dataset_name


def _load_layer_success(layer_path: Path) -> tuple[int | None, float | None, dict, dict]:
    """Load a layer JSON and return success and DiD metadata."""

    try:
        layer_id = int(layer_path.stem.split("_")[1])
    except (IndexError, ValueError):
        return None, None, {}, {}

    with layer_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    success_payload = payload.get("success_results", {})
    did_payload = payload.get("did", {})
    return layer_id, success_payload.get("success_rate"), success_payload, did_payload


def collect_best_layer_success(root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Collect best-layer success per probe, model, and dataset.

    Args:
        root: Root directory containing probe outputs (defaults to outputs/interv).

    Returns:
        DataFrame with one row per (probe, model, dataset) summarizing the best
        layer and associated success metrics.

    Raises:
        FileNotFoundError: If the expected output directory does not exist.
    """

    if not root.exists():
        raise FileNotFoundError(f"Intervention output directory missing: {root}")

    records: list[dict] = []
    for probe_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        probe = probe_dir.name
        for model_dir in sorted(m for m in probe_dir.iterdir() if m.is_dir()):
            model = model_dir.name
            for dataset_dir in sorted(d for d in model_dir.iterdir() if d.is_dir()):
                layer_files = sorted(dataset_dir.glob("layer_*.json"))
                if not layer_files:
                    continue

                best_layer_id: int | None = None
                best_success = -math.inf
                best_payload: dict | None = None
                best_did: dict | None = None

                for layer_file in layer_files:
                    layer_id, success_rate, success_payload, did_payload = _load_layer_success(
                        layer_file
                    )
                    if layer_id is None or success_rate is None:
                        continue
                    is_better = success_rate > best_success
                    is_tie = math.isclose(success_rate, best_success)
                    choose_layer = is_better or (
                        is_tie and best_layer_id is not None and layer_id < best_layer_id
                    )
                    if choose_layer:
                        best_layer_id = layer_id
                        best_success = success_rate
                        best_payload = success_payload
                        best_did = did_payload

                if best_layer_id is None or best_payload is None or best_did is None:
                    continue

                dataset_name = dataset_dir.name
                record = {
                    "probe": probe,
                    "model": model,
                    "dataset": dataset_name,
                    "dataset_base": _strip_task_suffix(dataset_name),
                    "best_layer": best_layer_id,
                    "success_rate": best_success,
                    "dominant_direction": best_payload.get("dominant_direction"),
                    "n_success": best_payload.get("n_success"),
                    "n_total": best_payload.get("n_total"),
                    "opposition_rate": best_payload.get("opposition_rate"),
                    "zero_effect_rate": best_payload.get("zero_effect_rate"),
                    "p_value": best_payload.get("p_value"),
                    "interaction_coef": best_did.get("interaction_coef"),
                    "interaction_std": best_did.get("interaction_std"),
                    "interaction_pval": best_did.get("interaction_pval"),
                }
                records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df.sort_values(
        ["probe", "model", "dataset", "success_rate"], inplace=True, ascending=False
    )
    df.reset_index(drop=True, inplace=True)
    return df


def summarize_by_probe(best_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-probe mean success rates.

    Args:
        best_df: DataFrame returned by collect_best_layer_success.

    Returns:
        Summary DataFrame with mean and standard deviation of success_rate per
        probe, sorted according to PROBE_ORDER.
    """

    summary = (
        best_df.groupby("probe")["success_rate"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_success",
                "std": "std_success",
                "count": "n_runs",
            }
        )
    )
    summary["std_success"] = summary["std_success"].fillna(0.0)
    # Sort by PROBE_ORDER
    summary["probe"] = pd.Categorical(summary["probe"], categories=PROBE_ORDER, ordered=True)
    summary = summary.sort_values("probe")
    return summary


def summarize_interaction(best_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-probe interaction coefficient summary (best layers).

    Drops rows without `interaction_coef`, takes absolute values, and reports
    mean, std, count, and a simple 95% CI using normal approximation.
    Sorted according to PROBE_ORDER.
    """

    df = best_df.dropna(subset=["interaction_coef"])
    if df.empty:
        return pd.DataFrame(columns=[
            "probe",
            "mean_abs_interaction",
            "std_abs_interaction",
            "n_runs",
            "ci_low",
            "ci_high",
        ])

    df = df.assign(interaction_abs=df["interaction_coef"].abs())
    summary = (
        df.groupby("probe")["interaction_abs"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_abs_interaction",
                "std": "std_abs_interaction",
                "count": "n_runs",
            }
        )
    )
    summary["std_abs_interaction"] = summary["std_abs_interaction"].fillna(0.0)
    # normal approx CI
    se = summary["std_abs_interaction"] / summary["n_runs"].pow(0.5).replace(0, pd.NA)
    summary["ci_low"] = summary["mean_abs_interaction"] - 1.96 * se
    summary["ci_high"] = summary["mean_abs_interaction"] + 1.96 * se
    # Sort by PROBE_ORDER
    summary["probe"] = pd.Categorical(summary["probe"], categories=PROBE_ORDER, ordered=True)
    summary = summary.sort_values("probe")
    return summary


def plot_probe_success(
    best_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path = FIGURE_PATH,
) -> Path:
    """Plot average success rate for each probe with per-run points.

    Args:
        best_df: Best-layer DataFrame with per-run success rates.
        summary_df: Aggregated per-probe summary statistics.
        output_path: Path where the PNG figure should be saved.

    Returns:
        Path to the saved PNG file.
    """

    order = summary_df["probe"].tolist()
    sns.set_theme(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use glasgowS palette from cmcrameri like in make_plots.ipynb
    palette = [cm.glasgowS(i) for i in range(2, 2 + len(order) * 2, 2)]

    sns.barplot(
        data=best_df,
        x="probe",
        y="success_rate",
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
        hue="probe",
        palette=palette,
        legend=False,
        width=0.65,
        edgecolor="black",
        linewidth=1,
    )
    sns.stripplot(
        data=best_df,
        x="probe",
        y="success_rate",
        order=order,
        ax=ax,
        color="black",
        size=6,
        alpha=0.7,
        jitter=0.18,
    )

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylabel("Success rate (best layer per model/dataset)")
    ax.set_xlabel("Probe")
    ax.set_ylim(0, 1.05)

    for idx, probe in enumerate(order):
        n_runs = int(summary_df.loc[summary_df["probe"] == probe, "n_runs"].iloc[0])
        mean_success = summary_df.loc[
            summary_df["probe"] == probe, "mean_success"
        ].iloc[0]
        ax.text(
            idx,
            min(mean_success + 0.05, 1.02),
            f"n={n_runs}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    sns.despine()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_interaction_forest(
    summary_df: pd.DataFrame, output_path: Path = FOREST_FIGURE_PATH
) -> Path:
    """Forest plot of interaction coefficients per probe.

    Args:
        summary_df: Output of summarize_interaction.
        output_path: Path where the PNG figure should be saved.
    """

    if summary_df.empty:
        raise ValueError("No interaction coefficients available to plot.")

    summary_df = summary_df.dropna(subset=["mean_abs_interaction", "ci_low", "ci_high"])
    if summary_df.empty:
        raise ValueError("No valid interaction coefficients with confidence intervals to plot.")

    order = summary_df["probe"].tolist()
    y_pos = range(len(order))

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(order) + 1))

    ax.errorbar(
        x=summary_df["mean_abs_interaction"],
        y=list(y_pos),
        xerr=[
            summary_df["mean_abs_interaction"] - summary_df["ci_low"],
            summary_df["ci_high"] - summary_df["mean_abs_interaction"],
        ],
        fmt="o",
        color="black",
        ecolor="gray",
        elinewidth=1.5,
        capsize=4,
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(order)
    ax.set_xlabel("|DiD interaction coefficient| (best layer per model/dataset)")
    ax.set_ylabel("Probe")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for y, (_, row) in zip(y_pos, summary_df.iterrows()):
        ax.text(
            row["ci_high"] + 0.01,
            y,
            f"n={int(row['n_runs'])}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_interaction_bar(
    best_df: pd.DataFrame, output_path: Path = INTER_BAR_FIGURE_PATH
) -> Path:
    """Bar plot of |interaction| by probe and dataset (best layers).

    Shows mean absolute interaction coefficient per probe, grouped by dataset
    (dataset_base), with 95% CI error bars across models.
    Probes ordered by PROBE_ORDER; datasets by DATAPACK_ORDER.
    """

    df = best_df.dropna(subset=["interaction_coef"])
    if df.empty:
        raise ValueError("No interaction coefficients available to plot.")

    df = df.assign(interaction_abs=df["interaction_coef"].abs())
    # Filter PROBE_ORDER to only include probes present in the data
    order = [p for p in PROBE_ORDER if p in df["probe"].unique()]

    plt.close("all")
    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, :])

    palette_map = {
        "city_locations": cm.lipariS(2),
        "med_indications": cm.lipariS(4),
        "word_definitions": cm.lipariS(3),
    }

    sns.barplot(
        data=df,
        x="probe",
        y="interaction_abs",
        hue="dataset_base",
        hue_order=DATAPACK_ORDER,
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax_top_left,
        width=0.7,
        palette=palette_map,
        edgecolor="black",
        linewidth=1,
        capsize=0.08,
    )
    ax_top_left.set_ylabel("|DiD interaction coefficient|")
    ax_top_left.set_xlabel("Probe")
    ax_top_left.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_top_left.legend(title="Dataset", loc="upper right")

    sns.barplot(
        data=best_df,
        x="probe",
        y="success_rate",
        hue="dataset_base",
        hue_order=DATAPACK_ORDER,
        order=order,
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax_top_right,
        width=0.7,
        palette=palette_map,
        edgecolor="black",
        linewidth=1,
        capsize=0.08,
    )
    ax_top_right.set_ylabel("Success rate")
    ax_top_right.set_xlabel("Probe")
    ax_top_right.set_ylim(0, 1.05)
    ax_top_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_top_right.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_top_right.legend_.remove()

    # Scatter plot: absolute interaction coefficient vs success rate
    scatter_df = best_df.dropna(subset=["interaction_coef", "success_rate", "zero_effect_rate"])
    scatter_df = scatter_df.assign(interaction_abs=scatter_df["interaction_coef"].abs())
    scatter = ax_bottom.scatter(
        scatter_df["interaction_abs"],
        scatter_df["success_rate"],
        c=scatter_df["zero_effect_rate"],
        s=80,
        alpha=0.7,
        cmap="viridis",
        edgecolors="black",
        linewidth=0.5,
    )
    ax_bottom.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="ω = 0.5")
    ax_bottom.set_xlabel("|DiD interaction coefficient|")
    ax_bottom.set_ylabel("Success rate (ω)")
    ax_bottom.grid(True, linestyle=":", alpha=0.5)
    ax_bottom.legend(loc="upper left")
    
    # Add colorbar for zero-effect rate
    cbar = fig.colorbar(scatter, ax=ax_bottom)
    cbar.set_label("Zero-effect rate", rotation=270, labelpad=20)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _export_tables(best_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Persist detailed and summary CSV tables alongside the figure."""

    DETAIL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(DETAIL_CSV_PATH, index=False)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)


def find_degenerate_did(root: Path = RESULTS_ROOT, min_obs: int = 4) -> pd.DataFrame:
    """Identify intervention JSONs with degenerate or NaN DiD statistics.

    Flags files where the `did` block contains NaN/Inf values, too few
    observations, or non-positive residual degrees of freedom. Useful for
    excluding bad runs before plotting.

    Args:
        root: Root directory containing probe outputs.
        min_obs: Minimum number of observations required to keep a run.

    Returns:
        DataFrame with one row per flagged file and a list of reasons.
    """

    records: list[dict] = []
    for path in root.glob("**/layer_*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - defensive
            records.append(
                {
                    "probe": path.parts[-4],
                    "model": path.parts[-3],
                    "dataset": path.parts[-2],
                    "layer": None,
                    "path": str(path),
                    "reasons": [f"json_error:{exc}"],
                }
            )
            continue

        did = data.get("did", {})
        reasons: list[str] = []

        def _flag_numeric(key: str) -> None:
            val = did.get(key)
            if isinstance(val, (int, float)):
                if math.isnan(val) or math.isinf(val):
                    reasons.append(f"{key}_nan_or_inf")

        for k in (
            "intercept_pval",
            "intercept_zval",
            "token_pval",
            "token_zval",
            "translation_pval",
            "translation_zval",
            "interaction_pval",
            "interaction_zval",
            "r_squared",
        ):
            _flag_numeric(k)

        n_obs = did.get("n_obs")
        if isinstance(n_obs, (int, float)) and n_obs < min_obs:
            reasons.append("too_few_obs")

        df_resid = did.get("df_resid")
        if isinstance(df_resid, (int, float)) and df_resid <= 0:
            reasons.append("df_resid_le_zero")

        if not isinstance(did, dict) or not did:
            reasons.append("missing_did")

        if reasons:
            try:
                layer = int(path.stem.split("_")[1])
            except Exception:
                layer = None
            records.append(
                {
                    "probe": path.parts[-4],
                    "model": path.parts[-3],
                    "dataset": path.parts[-2],
                    "layer": layer,
                    "path": str(path),
                    "reasons": reasons,
                }
            )

    if not records:
        return pd.DataFrame(columns=["probe", "model", "dataset", "layer", "path", "reasons"])

    df = pd.DataFrame(records)
    df.sort_values(["probe", "model", "dataset", "layer"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    """Generate probe success summary figure and CSV exports."""

    best_df = collect_best_layer_success()
    if best_df.empty:
        raise ValueError("No layer JSON files found under outputs/interv.")

    # Filter out zero_shot and spca for plotting
    plot_df = best_df[~best_df["probe"].isin(["zero_shot", "spca"])].copy()

    summary_df = summarize_by_probe(plot_df)
    interaction_df = summarize_interaction(plot_df)
    _export_tables(best_df, summary_df)
    plot_probe_success(plot_df, summary_df, FIGURE_PATH)
    plot_interaction_forest(interaction_df, FOREST_FIGURE_PATH)
    plot_interaction_bar(plot_df, INTER_BAR_FIGURE_PATH)


if __name__ == "__main__":
    main()
