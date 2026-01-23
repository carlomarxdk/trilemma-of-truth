"""Causal success plots showing intervention successes by dataset and probe.

The bar chart mirrors the styling used for performance figures but replaces MCC
metrics with raw success counts aggregated from intervention runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .constants import (
    CONDITION_COLOR,
    DATAPACK_ORDER,
    DATASET_COLOR,
    DATASET_NAMES,
    PROBE_NAMES,
    PROBE_ORDER,
    SAVE_DIR,
    SAVEFIG_OPTS,
    _setup_style,
)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success counts per probe/datapack pair."""

    summary = (
        df.groupby(["probe", "datapack"])["success_rate"]
        .agg(
            mean_success="mean",
            sem_success="sem",
            n_runs="size",
            total_success="sum",
        )
        .reset_index()
    )
    summary["sem_success"] = summary["sem_success"].fillna(0.0)
    summary["mean_success"] = summary["mean_success"].fillna(0.0)

    summary["probe"] = pd.Categorical(
        summary["probe"], categories=PROBE_ORDER, ordered=True
    )
    summary["datapack"] = pd.Categorical(
        summary["datapack"], categories=DATAPACK_ORDER, ordered=True
    )
    summary = summary.sort_values(["probe", "datapack"]).reset_index(drop=True)

    return summary


def _summarize_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate interaction coefficients per probe/datapack pair."""

    summary = (
        df.groupby(["probe", "datapack"])["interaction_coef"]
        .agg(
            mean_interaction="mean",
            sem_interaction="sem",
            n_runs="size",
        )
        .reset_index()
    )
    summary["sem_interaction"] = summary["sem_interaction"].fillna(0.0)
    summary["mean_interaction"] = summary["mean_interaction"].fillna(0.0)

    summary["probe"] = pd.Categorical(
        summary["probe"], categories=PROBE_ORDER, ordered=True
    )
    summary["datapack"] = pd.Categorical(
        summary["datapack"], categories=DATAPACK_ORDER, ordered=True
    )
    summary = summary.sort_values(["probe", "datapack"]).reset_index(drop=True)

    return summary


def _summarize_probe_mean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate average success rate per probe and retain per-run rows."""

    if "probe" not in df.columns or "success_rate" not in df.columns:
        raise ValueError("Input must include 'probe' and 'success_rate'.")

    working = df.copy()
    working["success_rate"] = pd.to_numeric(working["success_rate"], errors="coerce")
    working = working.dropna(subset=["probe", "success_rate"])
    if working.empty:
        raise ValueError("No valid success-rate rows to summarize.")

    summary = (
        working.groupby("probe")["success_rate"]
        .agg(mean_success="mean", sem_success="sem", n_runs="size")
        .reset_index()
    )
    summary["mean_success"] = summary["mean_success"].fillna(0.0)
    summary["sem_success"] = summary["sem_success"].fillna(0.0)

    summary["probe"] = pd.Categorical(
        summary["probe"], categories=PROBE_ORDER, ordered=True
    )
    summary = summary.sort_values("probe").reset_index(drop=True)

    working["probe"] = pd.Categorical(
        working["probe"], categories=PROBE_ORDER, ordered=True
    )
    working = working.sort_values("probe").reset_index(drop=True)
    return summary, working


def _summarize_probe_interaction(
    df_success: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate log-transformed interaction coefficient per probe.

    Applies log transformation to interaction_coef, then computes mean and SEM.

    Args:
        df_success: DataFrame containing 'probe' and 'interaction_coef' columns.

    Returns:
        Tuple of (summary, working) where summary has mean/sem per probe,
        and working has per-run log-transformed values.

    Raises:
        ValueError: If input lacks required columns or no valid rows remain.
    """
    if (
        "probe" not in df_success.columns
        or "interaction_coef" not in df_success.columns
    ):
        raise ValueError("Input must include 'probe' and 'interaction_coef'.")

    working = df_success.copy()
    working["interaction_coef"] = pd.to_numeric(
        working["interaction_coef"], errors="coerce"
    )
    working = working.dropna(subset=["probe", "interaction_coef"])

    if working.empty:
        raise ValueError("No valid interaction coefficient rows to summarize.")

    summary = (
        working.groupby("probe")["interaction_coef"]
        .agg(
            mean_interaction="mean",
            sem_interaction="sem",
            n_runs="size",
        )
        .reset_index()
    )
    summary["mean_interaction"] = summary["mean_interaction"].fillna(0.0)
    summary["sem_interaction"] = summary["sem_interaction"].fillna(0.0)

    summary["probe"] = pd.Categorical(
        summary["probe"], categories=PROBE_ORDER, ordered=True
    )
    summary = summary.sort_values("probe").reset_index(drop=True)

    working["probe"] = pd.Categorical(
        working["probe"], categories=PROBE_ORDER, ordered=True
    )
    working = working.sort_values("probe").reset_index(drop=True)
    return summary, working


def plot_causal_success_by_dataset(
    df_success: pd.DataFrame, save_dir: Path | None = None
) -> Path:
    """Plot intervention success counts grouped by probe and dataset.

    Aggregates ``success_rate`` across intervention runs and renders grouped bars
    showing the total number of successful interventions per probe/datapack pair.

    Args:
        df_success: DataFrame containing ``probe``,``success_rate`` and ``datapack``.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.

    Raises:
        ValueError: If the input lacks required columns or no rows remain after
            filtering invalid values.
    """

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    summary = _summarize(df_success)

    probe_order = [p for p in PROBE_ORDER if p in summary["probe"].unique()]
    datapack_order = [d for d in DATAPACK_ORDER if d in summary["datapack"].unique()]
    if not probe_order or not datapack_order:
        raise ValueError("Insufficient probe or datapack coverage to render plot.")

    n_probes = len(probe_order)
    n_datapacks = len(datapack_order)
    bar_width = 0.2
    bar_gap = 0.05
    step = bar_width + bar_gap

    x_centers = np.arange(n_probes)
    offsets = (np.arange(n_datapacks) - (n_datapacks - 1) / 2) * step

    fig, ax = plt.subplots(figsize=(5.35, 3))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []

    for probe_idx, probe in enumerate(probe_order):
        for dp_idx, datapack in enumerate(datapack_order):
            subset = summary[
                (summary["probe"] == probe) & (summary["datapack"] == datapack)
            ]
            if subset.empty:
                continue

            pos = x_centers[probe_idx] + offsets[dp_idx]
            bar = ax.bar(
                pos,
                subset["mean_success"],
                bar_width,
                yerr=subset["sem_success"],
                capsize=3,
                color=DATASET_COLOR[datapack],
                linewidth=1.3,
                edgecolor="black",
            )
            label = DATASET_NAMES.get(datapack, datapack.replace("_", " ").title())
            if label not in legend_labels:
                legend_handles.append(bar[0])
                legend_labels.append(label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", length=4, width=1.2)

    ax.set_xticks(x_centers)
    labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Mean success rate  ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("Intervention successes by dataset and probe")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    margin = 0.2
    left_margin = x_centers[0] + offsets.min() - bar_width / 2 - margin
    right_margin = x_centers[-1] + offsets.max() + bar_width / 2 + margin
    ax.set_xlim(left_margin, right_margin)
    ax.hlines(
        y=0,
        xmin=left_margin,
        xmax=right_margin,
        colors="gray",
        linestyles="dotted",
        zorder=0,
    )
    ax.legend(
        legend_handles,
        legend_labels,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        handlelength=1.2,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_success_by_dataset.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Intervention successes grouped by dataset and probe",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_causal_interaction_by_dataset(
    df_success: pd.DataFrame, save_dir: Path | None = None
) -> Path:
    """Plot interaction coefficients grouped by probe and dataset.

    Aggregates ``interaction_coef`` across intervention runs and renders grouped bars
    showing the mean interaction coefficient per probe/datapack pair.

    Args:
        df_success: DataFrame containing ``probe``, ``interaction_coef`` and ``datapack``.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.

    Raises:
        ValueError: If the input lacks required columns or no rows remain after
            filtering invalid values.
    """

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    summary = _summarize_interaction(df_success)

    probe_order = [p for p in PROBE_ORDER if p in summary["probe"].unique()]
    datapack_order = [d for d in DATAPACK_ORDER if d in summary["datapack"].unique()]
    if not probe_order or not datapack_order:
        raise ValueError("Insufficient probe or datapack coverage to render plot.")

    n_probes = len(probe_order)
    n_datapacks = len(datapack_order)
    bar_width = 0.2
    bar_gap = 0.05
    step = bar_width + bar_gap

    x_centers = np.arange(n_probes)
    offsets = (np.arange(n_datapacks) - (n_datapacks - 1) / 2) * step

    fig, ax = plt.subplots(figsize=(5.35, 3))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []

    for probe_idx, probe in enumerate(probe_order):
        for dp_idx, datapack in enumerate(datapack_order):
            subset = summary[
                (summary["probe"] == probe) & (summary["datapack"] == datapack)
            ]
            if subset.empty:
                continue

            pos = x_centers[probe_idx] + offsets[dp_idx]
            bar = ax.bar(
                pos,
                subset["mean_interaction"],
                bar_width,
                yerr=subset["sem_interaction"],
                capsize=3,
                color=DATASET_COLOR[datapack],
                linewidth=1.3,
                edgecolor="black",
            )
            label = DATASET_NAMES.get(datapack, datapack.replace("_", " ").title())
            if label not in legend_labels:
                legend_handles.append(bar[0])
                legend_labels.append(label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", length=4, width=1.2)

    ax.set_xticks(x_centers)
    labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Mean interaction coefficient ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("Interaction coefficients by dataset and probe")
    ax.set_yscale("log")

    margin = 0.2
    left_margin = x_centers[0] + offsets.min() - bar_width / 2 - margin
    right_margin = x_centers[-1] + offsets.max() + bar_width / 2 + margin
    ax.set_xlim(left_margin, right_margin)

    ax.legend(
        legend_handles,
        legend_labels,
        title="Dataset",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        handlelength=1.2,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_interaction_by_dataset.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Interaction coefficients grouped by dataset and probe",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_causal_success_average(
    df_success: pd.DataFrame, save_dir: Path | None = None
) -> Path:
    """Plot average success rate across all datasets for each probe."""

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    summary, _ = _summarize_probe_mean(df_success)
    summary = summary[summary["probe"] != "svm"].reset_index(
        drop=True
    )  # Exclude SVM probe
    probe_order = [
        p for p in PROBE_ORDER if p in summary["probe"].unique() and p != "svm"
    ]
    if not probe_order:
        raise ValueError("No probes available to plot causal success averages.")

    # Reorder summary to match probe_order
    summary = summary.set_index("probe").loc[probe_order].reset_index()

    x_pos = np.arange(len(probe_order))
    bar_width = 0.5
    bag_color = CONDITION_COLOR.get("bag", "#4c6a91")

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(
        x_pos,
        summary["mean_success"],
        bar_width,
        yerr=summary["sem_success"],
        capsize=3,
        color=bag_color,
        linewidth=1.3,
        edgecolor="black",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", length=4, width=1.2)

    ax.set_xticks(x_pos)
    labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Mean Success Rate ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("Average Intervention Success")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    margin = 0.5
    ax.set_xlim(x_pos[0] - margin, x_pos[-1] + margin)
    ax.hlines(
        y=0,
        xmin=x_pos[0] - margin,
        xmax=x_pos[-1] + margin,
        colors="gray",
        linestyles="dotted",
        zorder=0,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_success_average.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Average intervention success by probe",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_causal_interaction_average(
    df_success: pd.DataFrame, save_dir: Path | None = None
) -> Path:
    """Plot average success rate across all datasets for each probe."""

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    # _summarize_probe_mean should be changed to _summarize_probe_interaction
    summary, working = _summarize_probe_interaction(df_success)
    probe_order = [p for p in PROBE_ORDER if p in summary["probe"].unique()]
    if not probe_order:
        raise ValueError("No probes available to plot causal success averages.")

    x_pos = np.arange(len(probe_order))
    bar_width = 0.5
    bag_color = CONDITION_COLOR.get("bag", "#4c6a91")

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(
        x_pos,
        summary["mean_interaction"],
        bar_width,
        yerr=summary["sem_interaction"],
        capsize=3,
        color=bag_color,
        linewidth=1.3,
        edgecolor="black",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", length=4, width=1.2)

    ax.set_xticks(x_pos)
    labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Mean Interaction Coefficient (log) ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("Average Interaction Coefficient (log)")

    margin = 0.5
    ax.set_xlim(x_pos[0] - margin, x_pos[-1] + margin)
    ax.hlines(
        y=0,
        xmin=x_pos[0] - margin,
        xmax=x_pos[-1] + margin,
        colors="gray",
        linestyles="dotted",
        zorder=0,
    )
    ax.set_yscale("log")

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_interaction_average.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Average intervention interaction coefficient by probe",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def violin_causal_success_average(
    df_success: pd.DataFrame, save_dir: Path | None = None
) -> Path:
    """Violin plot of per-run success rates grouped by probe."""

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    summary, working = _summarize_probe_mean(df_success)
    probe_order = [p for p in PROBE_ORDER if p in summary["probe"].unique()]
    if not probe_order:
        raise ValueError("No probes available to plot causal success violins.")

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(4.75, 3.1))

    sns.violinplot(
        data=working,
        x="probe",
        y="success_rate",
        order=probe_order,
        color=CONDITION_COLOR.get("bag", "#4c6a91"),
        saturation=0.9,
        cut=0,
        inner=None,
        linewidth=0,
        ax=ax,
    )

    sns.stripplot(
        data=working,
        x="probe",
        y="success_rate",
        order=probe_order,
        dodge=False,
        color="black",
        size=2.4,
        alpha=0.65,
        ax=ax,
    )

    ax.plot(
        np.arange(len(probe_order)),
        summary.set_index("probe").loc[probe_order, "mean_success"],
        color="black",
        marker="o",
        markersize=3.5,
        linewidth=1,
        label="Mean",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", length=4, width=1.1)

    ax.set_xticks(np.arange(len(probe_order)))
    labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("Success rate distribution")
    ax.set_xlabel("Probe")
    ax.set_title("Intervention success distribution by probe")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    margin = 0.4
    ax.set_xlim(-margin, len(probe_order) - 1 + margin)
    ax.hlines(
        y=0,
        xmin=-margin,
        xmax=len(probe_order) - 1 + margin,
        colors="gray",
        linestyles="dotted",
        zorder=0,
    )

    ax.legend(loc="upper left", fontsize="small", frameon=False)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_success_average_violin.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Intervention success distribution by probe",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_causal_success_interaction_2d(
    df_success: pd.DataFrame, save_dir: Path | None = None
) -> Path:
    """Plot interaction coefficient vs success rate colored by probe, marked by dataset.

    Args:
        df_success: DataFrame containing columns for interaction_coef,
            success_rate, probe, and datapack.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.
    """
    from .constants import PROBE_COLOR

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    # Define markers for each dataset
    dataset_markers = {
        "city_locations": "X",
        "word_definitions": "o",
        "med_indications": "s",  # square marker
    }

    fig, ax = plt.subplots(figsize=(4, 3))
    probe_order = [p for p in PROBE_ORDER if p in df_success["probe"].unique()]
    datapack_order = [d for d in DATAPACK_ORDER if d in df_success["datapack"].unique()]

    # Plot by probe and dataset
    for probe in probe_order:
        for datapack in datapack_order:
            subset = df_success[
                (df_success["probe"] == probe) & (df_success["datapack"] == datapack)
            ]
            if subset.empty:
                continue

            ax.scatter(
                subset["interaction_coef"],
                subset["success_rate"],
                label=f"{PROBE_NAMES.get(probe, probe)}",
                color=PROBE_COLOR.get(probe, "#808080"),
                marker=dataset_markers.get(datapack, "o"),
                alpha=0.8,
                edgecolors="w",
                s=50,
                linewidth=0.5,
            )

    ax.set_xlabel("Abs Interaction Coefficient (log-scale)")
    ax.set_ylabel("Success Rate")
    ax.set_xscale("log")
    ax.set_title("Causal Success vs Absolute Interaction Coefficient")

    # Create legends: one for probes (colors), one for datasets (markers)
    from matplotlib.lines import Line2D

    # Probe legend (colors)
    probe_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PROBE_COLOR.get(p, "#808080"),
            markersize=6,
            label=PROBE_NAMES.get(p, p),
        )
        for p in probe_order
    ]

    # Dataset legend (markers)
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=dataset_markers.get(d, "o"),
            color="w",
            markerfacecolor="gray",
            markersize=6,
            label=DATASET_NAMES.get(d, d),
        )
        for d in datapack_order
    ]

    first_legend = ax.legend(
        handles=probe_handles, title="Probe", loc="upper left", frameon=True
    )
    ax.add_artist(first_legend)
    ax.legend(handles=dataset_handles, title="Dataset", loc="lower right", frameon=True)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_success_interaction_2d.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Causal Success vs Absolute Interaction Coefficient",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path
