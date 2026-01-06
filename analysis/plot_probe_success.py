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
from matplotlib.ticker import PercentFormatter

RESULTS_ROOT = Path("outputs/interv")
FIGURE_PATH = Path("outputs/figures/probe_success_best_layer.png")
DETAIL_CSV_PATH = Path("outputs/figures/probe_success_by_model_dataset.csv")
SUMMARY_CSV_PATH = Path("outputs/figures/probe_success_best_layer_summary.csv")


def _strip_task_suffix(dataset_name: str) -> str:
    """Drop trailing task identifiers to recover the base dataset name."""

    split_token = "_search_task"
    if split_token in dataset_name:
        return dataset_name.split(split_token, maxsplit=1)[0]
    return dataset_name


def _load_layer_success(layer_path: Path) -> tuple[int | None, float | None, dict]:
    """Load a layer JSON and return the parsed success metadata."""

    try:
        layer_id = int(layer_path.stem.split("_")[1])
    except (IndexError, ValueError):
        return None, None, {}

    with layer_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    success_payload = payload.get("success_results", {})
    return layer_id, success_payload.get("success_rate"), success_payload


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

                for layer_file in layer_files:
                    layer_id, success_rate, success_payload = _load_layer_success(
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

                if best_layer_id is None or best_payload is None:
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
        probe, sorted from highest to lowest mean.
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
        .sort_values("mean_success", ascending=False)
    )
    summary["std_success"] = summary["std_success"].fillna(0.0)
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
    palette = sns.color_palette("colorblind", n_colors=len(order))

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


def _export_tables(best_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Persist detailed and summary CSV tables alongside the figure."""

    DETAIL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(DETAIL_CSV_PATH, index=False)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)


def main() -> None:
    """Generate probe success summary figure and CSV exports."""

    best_df = collect_best_layer_success()
    if best_df.empty:
        raise ValueError("No layer JSON files found under outputs/interv.")

    summary_df = summarize_by_probe(best_df)
    _export_tables(best_df, summary_df)
    plot_probe_success(best_df, summary_df, FIGURE_PATH)


if __name__ == "__main__":
    main()
