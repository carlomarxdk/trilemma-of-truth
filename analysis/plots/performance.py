"""Performance plots: grouped bars by dataset and condition."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from .constants import (
    CONDITION_COLOR,
    CONDITION_NAMES,
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
    """Compute mean and standard error of MCC per datapack/probe pair."""

    out = (
        df.groupby(["datapack", "probe"])
        .agg(mean=("mcc", "mean"), sem=("mcc", "sem"))
        .reset_index()
    )
    out["probe"] = pd.Categorical(out["probe"], categories=PROBE_ORDER, ordered=True)
    out["datapack"] = pd.Categorical(
        out["datapack"], categories=DATAPACK_ORDER, ordered=True
    )
    return out.sort_values(["probe", "datapack"])


def plot_performance_by_dataset(
    df_results: pd.DataFrame, save_dir: Path | None = None
) -> Path:  # noqa: C901
    """Plot MCC grouped by probe, dataset, and condition.

    Only the ``bag`` and ``instance`` conditions are visualized; all other
    condition values are ignored.

    Args:
        df_results: Raw results with columns ``datapack``, ``probe``, ``condition``, and
            ``mcc``.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.

    Raises:
        ValueError: If required columns are missing or no valid bag/instance pairs exist.
    """

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    required_cols = {"datapack", "probe", "condition", "mcc"}

    # Check for required columns and raise error if missing
    missing_cols = required_cols - set(df_results.columns)
    if missing_cols:
        raise ValueError(
            f"df_results is missing required columns: {sorted(missing_cols)}"
        )

    valid_conditions = {"bag", "instance"}
    present = df_results[df_results["condition"].isin(valid_conditions)].copy()

    counts = (
        present.groupby(["datapack", "probe", "condition"]).size().unstack(fill_value=0)
    )
    valid_pairs = counts[(counts.get("bag", 0) > 0) & (counts.get("instance", 0) > 0)]
    if valid_pairs.empty:
        warnings.warn(
            "No datapack/probe pairs have both bag and instance conditions.",
            stacklevel=2,
        )
        raise ValueError("Cannot plot performance without paired bag/instance results.")

    missing_pairs = counts.index.difference(valid_pairs.index)
    if len(missing_pairs) > 0:
        warnings.warn(
            f"Skipping {len(missing_pairs)} datapack/probe pairs missing bag or instance.",
            stacklevel=2,
        )

    # Retain only valid datapack/probe pairs with both conditions
    present = present.merge(
        valid_pairs.reset_index()[["datapack", "probe"]],
        on=["datapack", "probe"],
        how="inner",
    )

    df_bag = _summarize(present[present["condition"] == "bag"])
    df_inst = _summarize(present[present["condition"] == "instance"])

    # Layout: 6 bars per probe (instance1-3 then bag1-3) with uniform intra-bar gaps.
    n_probes = len(PROBE_ORDER)
    n_datapacks = len(DATAPACK_ORDER)
    bars_per_probe = n_datapacks * 2
    bar_width = 0.12
    bar_gap = 0.045  # uniform gap between any two neighboring bars
    step = bar_width + bar_gap
    group_gap = 0.3
    group_width = bars_per_probe * step - bar_gap  # no trailing gap after the last bar
    group_starts = np.arange(n_probes) * (group_width + group_gap)
    xtick_centers = group_starts + (group_width - bar_gap) / 2

    fig, ax = plt.subplots(figsize=(6.65, 3))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []
    seen_labels: set[str] = set()

    for probe_idx, probe in enumerate(PROBE_ORDER):
        start = group_starts[probe_idx]
        # Instances first for this probe (three bars), then bags (three bars).
        for dp_idx, datapack in enumerate(DATAPACK_ORDER):
            inst_subset = df_inst[
                (df_inst["datapack"] == datapack) & (df_inst["probe"] == probe)
            ]
            if inst_subset.empty:
                continue

            inst_pos = start + dp_idx * step
            inst_face = to_rgba(DATASET_COLOR[datapack], alpha=0.45)
            inst_bar = ax.bar(
                inst_pos,
                inst_subset["mean"],
                bar_width,
                yerr=inst_subset["sem"],
                capsize=3,
                color=inst_face,
                hatch="////",
                linewidth=1,
                edgecolor="black",
            )
            if (label := f"{DATASET_NAMES[datapack]} (instance)") not in seen_labels:
                legend_handles.append(inst_bar[0])
                legend_labels.append(label)
                seen_labels.add(label)

        for dp_idx, datapack in enumerate(DATAPACK_ORDER):
            bag_subset = df_bag[
                (df_bag["datapack"] == datapack) & (df_bag["probe"] == probe)
            ]
            if bag_subset.empty:
                continue

            bag_pos = start + (n_datapacks + dp_idx) * step
            bag_bar = ax.bar(
                bag_pos,
                bag_subset["mean"],
                bar_width,
                yerr=bag_subset["sem"],
                capsize=3,
                color=DATASET_COLOR[datapack],
                linewidth=1,
                edgecolor="black",
                alpha=1.0,
            )
            if (label := f"{DATASET_NAMES[datapack]} (bag)") not in seen_labels:
                legend_handles.append(bag_bar[0])
                legend_labels.append(label)
                seen_labels.add(label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis="both", length=4, width=2)

    ax.set_xticks(xtick_centers)
    probe_labels = [PROBE_NAMES.get(p, p) for p in PROBE_ORDER]
    ax.set_xticklabels(probe_labels, rotation=0, ha="center")
    ax.set_ylabel("Mean MCC ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("MCC by Probe and Dataset")
    ax.set_ylim(-0.05, 1.00)
    left_margin = group_starts[0] - bar_width - 0.1
    right_margin = group_starts[-1] + group_width + bar_width
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
        title="Dataset (Condition)",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        fontsize="small",
        title_fontsize="small",
        handlelength=1.2,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "performance_by_dataset.pdf"
    fig.tight_layout()
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "MCC by dataset and probe (bag vs instance)",
            "Author": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_performance_by_condition(
    df_results: pd.DataFrame, save_dir: Path | None = None
) -> Path:  # noqa: C901
    """Plot MCC aggregated by condition (bag vs instance) across probes.

    Args:
        df_results: Results with columns ``probe``, ``condition``, and ``mcc``.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.

    Raises:
        ValueError: If required columns are missing or no valid conditions are found.
    """

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    required_cols = {"probe", "condition", "mcc"}
    missing_cols = required_cols - set(df_results.columns)
    if missing_cols:
        raise ValueError(
            f"df_results is missing required columns: {sorted(missing_cols)}"
        )

    valid_conditions = {"bag", "instance"}
    present = df_results[df_results["condition"].isin(valid_conditions)].copy()
    if present.empty:
        raise ValueError("No rows with recognized conditions to plot.")

    df_agg = (
        present.groupby(["probe", "condition"])
        .agg(mean=("mcc", "mean"), sem=("mcc", "sem"))
        .reset_index()
    )
    
    print(df_agg)

    probe_order = [p for p in PROBE_ORDER if p in df_agg["probe"].unique()]
    if not probe_order:
        raise ValueError("No probes with valid condition data to plot.")

    condition_order = [
        cond for cond in ("instance", "bag") if cond in df_agg["condition"].unique()
    ]
    if not condition_order:
        raise ValueError("No recognized conditions present after filtering.")

    n_probes = len(probe_order)
    bar_width = 0.24
    bar_gap = 0.08
    half_gap = bar_gap / 2
    group_gap = 0.24
    group_width = 2 * (bar_width + half_gap)
    xtick_centers = np.arange(n_probes) * (group_width + group_gap)

    palette = {cond: CONDITION_COLOR[cond] for cond in condition_order}

    fig, ax = plt.subplots(figsize=(4, 3))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []
    seen: set[str] = set()

    for probe_idx, probe in enumerate(probe_order):
        center = xtick_centers[probe_idx]
        for cond_idx, cond in enumerate(condition_order):
            subset = df_agg[(df_agg["probe"] == probe) & (df_agg["condition"] == cond)]
            if subset.empty:
                continue

            sign = -1 if cond_idx == 0 else 1
            pos = center + sign * (bar_width / 2 + half_gap)
            bar = ax.bar(
                pos,
                subset["mean"],
                bar_width,
                yerr=subset["sem"],
                capsize=3,
                color=palette[cond],
                linewidth=1.3,
                edgecolor="black",
            )
            label = CONDITION_NAMES.get(cond, cond.replace("_", " ").title())
            if label not in seen:
                legend_handles.append(bar[0])
                legend_labels.append(label)
                seen.add(label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", length=4, width=1.5)

    ax.set_xticks(xtick_centers)
    probe_labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(probe_labels, rotation=0, ha="center")
    ax.set_ylabel("Mean MCC ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("MCC by Condition and Probe")
    ax.set_ylim(-0.05, 1.00)
    span = bar_width + half_gap
    left_margin = xtick_centers[0] - span - 0.2
    right_margin = xtick_centers[-1] + span + 0.1
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
        title="Condition",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        handlelength=1.2,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "performance_by_condition.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "MCC by condition across probes",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_generalization_by_dataset(
    df_results: pd.DataFrame, save_dir: Path | None = None
) -> Path:  # noqa: C901
    """Plot cross-dataset MCC by target dataset, probe, and condition.

    Expects a ``subexperiment`` column denoting the target dataset. Rows where the
    source and target datasets match are dropped. Only ``bag`` and ``instance``
    conditions are visualized.

    Args:
        df_results: Results with columns ``datapack`` (source), ``subexperiment``
            (target), ``probe``, ``condition``, and ``mcc``.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.

    Raises:
        ValueError: If required columns are missing or no valid bag/instance rows
            remain after filtering.
    """

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    required_cols = {"datapack", "subexperiment", "probe", "condition", "mcc"}
    missing_cols = required_cols - set(df_results.columns)
    if missing_cols:
        raise ValueError(
            f"df_results is missing required columns: {sorted(missing_cols)}"
        )

    valid_conditions = {"bag", "instance"}
    present = df_results[df_results["condition"].isin(valid_conditions)].copy()

    present = present.rename(columns={"datapack": "source_datapack"})
    present["target_datapack"] = present["subexperiment"].str.replace(
        r"^g_", "", regex=True
    )
    target_map = {
        "cities_loc": "city_locations",
        "defs": "word_definitions",
        "med_indications": "med_indications",
    }
    present["target_datapack"] = (
        present["target_datapack"].map(target_map).fillna(present["target_datapack"])
    )
    present = present[present["source_datapack"] != present["target_datapack"]]
    present = present[present["target_datapack"].isin(DATAPACK_ORDER)]
    if present.empty:
        raise ValueError("No cross-dataset rows with bag/instance conditions to plot.")

    present = present[
        [
            "model",
            "condition",
            "target_datapack",
            "probe",
            "mcc",
            "CI_low",
            "CI_high",
        ]
    ].rename(columns={"target_datapack": "datapack"})

    df_bag = _summarize(present[present["condition"] == "bag"])
    df_inst = _summarize(present[present["condition"] == "instance"])

    probe_order = [
        p
        for p in PROBE_ORDER
        if p in pd.concat([df_bag["probe"], df_inst["probe"]]).unique()
    ]
    if not probe_order:
        raise ValueError("No probes with valid cross-dataset data to plot.")

    n_probes = len(probe_order)
    n_datapacks = len(DATAPACK_ORDER)
    bars_per_probe = n_datapacks * 2
    bar_width = 0.12
    bar_gap = 0.05
    step = bar_width + bar_gap
    group_gap = 0.3
    group_width = bars_per_probe * step - bar_gap
    group_starts = np.arange(n_probes) * (group_width + group_gap)
    xtick_centers = group_starts + (group_width - bar_gap) / 2

    fig, ax = plt.subplots(figsize=(8, 3))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []
    seen_labels: set[str] = set()

    for probe_idx, probe in enumerate(probe_order):
        start = group_starts[probe_idx]
        # Instances first, then bags for each target dataset.
        for dp_idx, datapack in enumerate(DATAPACK_ORDER):
            inst_subset = df_inst[
                (df_inst["datapack"] == datapack) & (df_inst["probe"] == probe)
            ]
            if inst_subset.empty:
                continue

            inst_pos = start + dp_idx * step
            inst_face = to_rgba(DATASET_COLOR[datapack], alpha=0.45)
            inst_bar = ax.bar(
                inst_pos,
                inst_subset["mean"],
                bar_width,
                yerr=inst_subset["sem"],
                capsize=3,
                color=inst_face,
                hatch="////",
                linewidth=1.3,
                edgecolor="black",
            )
            if (label := f"{DATASET_NAMES[datapack]} (instance)") not in seen_labels:
                legend_handles.append(inst_bar[0])
                legend_labels.append(label)
                seen_labels.add(label)

        for dp_idx, datapack in enumerate(DATAPACK_ORDER):
            bag_subset = df_bag[
                (df_bag["datapack"] == datapack) & (df_bag["probe"] == probe)
            ]
            if bag_subset.empty:
                continue

            bag_pos = start + (n_datapacks + dp_idx) * step
            bag_bar = ax.bar(
                bag_pos,
                bag_subset["mean"],
                bar_width,
                yerr=bag_subset["sem"],
                capsize=3,
                color=DATASET_COLOR[datapack],
                linewidth=1.3,
                edgecolor="black",
                alpha=0.95,
            )
            if (label := f"{DATASET_NAMES[datapack]} (bag)") not in seen_labels:
                legend_handles.append(bag_bar[0])
                legend_labels.append(label)
                seen_labels.add(label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis="both", length=4, width=2)

    ax.set_xticks(xtick_centers)
    probe_labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(probe_labels, rotation=0, ha="center")
    ax.set_ylabel("Mean MCC ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("Cross-Dataset MCC by Target Dataset and Probe")
    ax.set_ylim(-0.05, 1.00)
    left_margin = group_starts[0] - bar_width - 0.1
    right_margin = group_starts[-1] + group_width + bar_width
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
        title="Target Dataset (Condition)",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        fontsize="small",
        title_fontsize="small",
        handlelength=1.2,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "generalization_by_dataset.pdf"
    fig.tight_layout()
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Cross-dataset MCC by target dataset and probe (bag vs instance)",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path


def plot_generalization_by_condition(
    df_results: pd.DataFrame, save_dir: Path | None = None
) -> Path:  # noqa: C901
    """Plot cross-dataset MCC aggregated by condition across probes.

    Args:
        df_results: Results with columns ``datapack``, ``subexperiment``, ``probe``,
            ``condition``, and ``mcc``.
        save_dir: Output directory; defaults to constants.SAVE_DIR when None.

    Returns:
        Path to the saved PDF figure.

    Raises:
        ValueError: If required columns are missing or no valid rows remain.
    """

    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    required_cols = {"datapack", "subexperiment", "probe", "condition", "mcc"}
    missing_cols = required_cols - set(df_results.columns)
    if missing_cols:
        raise ValueError(
            f"df_results is missing required columns: {sorted(missing_cols)}"
        )

    valid_conditions = {"bag", "instance"}
    present = df_results[df_results["condition"].isin(valid_conditions)].copy()

    present = present.rename(columns={"datapack": "source_datapack"})
    present["target_datapack"] = present["subexperiment"].str.replace(
        r"^g_", "", regex=True
    )
    target_map = {
        "cities_loc": "city_locations",
        "defs": "word_definitions",
        "med_indications": "med_indications",
    }
    present["target_datapack"] = (
        present["target_datapack"].map(target_map).fillna(present["target_datapack"])
    )
    present = present[present["source_datapack"] != present["target_datapack"]]
    present = present[present["target_datapack"].isin(DATAPACK_ORDER)]
    if present.empty:
        raise ValueError("No cross-dataset rows with bag/instance conditions to plot.")

    df_agg = (
        present.groupby(["probe", "condition"])
        .agg(mean=("mcc", "mean"), sem=("mcc", "sem"))
        .reset_index()
    )

    probe_order = [p for p in PROBE_ORDER if p in df_agg["probe"].unique()]
    if not probe_order:
        raise ValueError("No probes with valid cross-dataset data to plot.")

    condition_order = [
        cond for cond in ("instance", "bag") if cond in df_agg["condition"].unique()
    ]
    if not condition_order:
        raise ValueError("No recognized conditions present after filtering.")

    n_probes = len(probe_order)
    bar_width = 0.24
    bar_gap = 0.08
    half_gap = bar_gap / 2
    group_gap = 0.24
    group_width = 2 * (bar_width + half_gap)
    xtick_centers = np.arange(n_probes) * (group_width + group_gap)

    palette = {cond: CONDITION_COLOR[cond] for cond in condition_order}

    fig, ax = plt.subplots(figsize=(4, 3))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []
    seen: set[str] = set()

    for probe_idx, probe in enumerate(probe_order):
        center = xtick_centers[probe_idx]
        for cond_idx, cond in enumerate(condition_order):
            subset = df_agg[(df_agg["probe"] == probe) & (df_agg["condition"] == cond)]
            if subset.empty:
                continue

            sign = -1 if cond_idx == 0 else 1
            pos = center + sign * (bar_width / 2 + half_gap)
            bar = ax.bar(
                pos,
                subset["mean"],
                bar_width,
                yerr=subset["sem"],
                capsize=3,
                color=palette[cond],
                linewidth=1.3,
                edgecolor="black",
            )
            label = CONDITION_NAMES.get(cond, cond.replace("_", " ").title())
            if label not in seen:
                legend_handles.append(bar[0])
                legend_labels.append(label)
                seen.add(label)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", length=4, width=1.5)

    ax.set_xticks(xtick_centers)
    probe_labels = [PROBE_NAMES.get(p, p) for p in probe_order]
    ax.set_xticklabels(probe_labels, rotation=0, ha="center")
    ax.set_ylabel("Mean MCC ± SEM")
    ax.set_xlabel("Probe")
    ax.set_title("Cross-Dataset MCC by Condition and Probe")
    ax.set_ylim(-0.05, 1.00)
    span = bar_width + half_gap
    left_margin = xtick_centers[0] - span - 0.2
    right_margin = xtick_centers[-1] + span + 0.1
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
        title="Condition",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        handlelength=1.2,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "generalization_by_condition.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Cross-dataset MCC by condition across probes",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path
