"""Causal success plots showing intervention successes by dataset and probe.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from statsmodels.stats.proportion import proportion_confint

def random_effects_meta(coefs: np.ndarray, ses: np.ndarray) -> tuple[float, float, float]:
    w = 1.0 / ses ** 2
    theta_fixed = (coefs * w).sum() / w.sum()
    Q = (w * (coefs - theta_fixed) ** 2).sum()
    k = len(coefs)
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0, (Q - (k - 1)) / c)
    w_re = 1.0 / (ses ** 2 + tau2)
    theta_re = (coefs * w_re).sum() / w_re.sum()
    se_re = np.sqrt(1.0 / w_re.sum())
    return theta_re, se_re, tau2


from .causal_utils import (
    apply_fdr_correction_by_probe,
    bootstrap_ci
)

from .constants import (
    DOSE_COLOR,
    PROBE_NAMES,
    PROBE_ORDER,
    SAVE_DIR,
    SAVEFIG_OPTS,
    _setup_style,
)

def _summarize(dfs: dict[int, pd.DataFrame], dose: list[int] | int) -> dict[int, pd.DataFrame]:
    dose = [dose] if isinstance(dose, int) else dose
    dfs_adj = {}
    for d in dose:
        df = dfs[d]
        df = apply_fdr_correction_by_probe(df, pval_col='interaction_pval', alpha=0.05)
        dfs_adj[d] = df

    return dfs_adj

def plot_causal_success_by_probe(dfs: dict[int, pd.DataFrame], dose: list[int] | int, save_dir: str | Path | None = None) -> Path:
    
    dose = [dose] if isinstance(dose, int) else dose

    # 1. Identify successful interventions using adjusted p-values
    sr = []
    dfs = _summarize(dfs, dose)
    for d in dose:
        
        df = dfs[d]

        for probe in ['mean_diff', 'ttpd', 'svm', 'sawmil']:
            sub = df[df['probe'] == probe]
            n = len(sub)
            k = (sub['significant']).sum()
            ci_low, ci_high = proportion_confint(k, n, alpha=0.05, method='beta')
            sr.append({
                'dose': d,
                'probe': probe,
                'n': n,
                'k': k,
                'sr': k / n if n > 0 else 0,
                'ci_low': ci_low,
                'ci_high': ci_high
            })
    df_sr = pd.DataFrame(sr)
    print(df_sr)
    
    # 2. Plot success rates with confidence intervals
    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    required_cols = {"dose", "probe", "sr", "ci_low", "ci_high"}
    missing_cols = required_cols - set(df_sr.columns)
    if missing_cols:
        raise ValueError(
            f"df_results is missing required columns: {sorted(missing_cols)}"
        )

    valid_conditions = set(dose)
    df_agg = df_sr[df_sr["dose"].isin(valid_conditions)].copy()

    

    probe_order = [p for p in PROBE_ORDER if p in df_agg["probe"].unique()]
    if not probe_order:
        raise ValueError("No probes with valid condition data to plot.")

    dose_order = [
        d for d in dose if d in df_agg["dose"].unique()
    ]
    if not dose_order:
        raise ValueError("No recognized doses present after filtering.")

    n_probes = len(probe_order)
    bar_width = 0.24
    bar_gap = 0.10
    half_gap = bar_gap / 2
    group_gap = 0.30
    group_width = 2 * (bar_width + half_gap)
    xtick_centers = np.arange(n_probes) * (group_width + group_gap)

    palette = {dose: DOSE_COLOR[dose] for dose in dose_order}

    fig, ax = plt.subplots(figsize=(3, 2.5))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []
    seen: set[str] = set()

    for probe_idx, probe in enumerate(probe_order):
        center = xtick_centers[probe_idx]
        for dose_idx, dose in enumerate(dose_order):
            subset = df_agg[(df_agg["probe"] == probe) & (df_agg["dose"] == dose)]
            if subset.empty:
                continue

            sign = -1 if dose_idx == 0 else 1
            pos = center + sign * (bar_width / 2 + half_gap)
            sr_val = subset["sr"].iloc[0]
            ci_low_val = subset["ci_low"].iloc[0]
            ci_high_val = subset["ci_high"].iloc[0]
            bar = ax.bar(
                pos,
                sr_val,
                bar_width,
                yerr=[[sr_val - ci_low_val], [ci_high_val - sr_val]],
                capsize=3,
                color=palette[dose],
                linewidth=1.3,
                edgecolor="black",
            )
            label = ["±{d}".format(d=d) for d in dose_order][dose_idx]
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
    ax.set_ylabel("Intervention Success Rate")
    ax.set_xlabel("Probe")
    ax.set_title("Success by Intervention Dose and Probe")
    ax.set_ylim(0.00, 1.00)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    span = bar_width + half_gap
    left_margin = xtick_centers[0] - span - 0.2
    right_margin = xtick_centers[-1] + span + 0.1
    ax.set_xlim(left_margin, right_margin)
    ax.legend(
        legend_handles,
        legend_labels,
        title="Dose",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        handlelength=1.2,
    )
    
    ax.hlines(
        y=0.5,
        xmin=left_margin,
        xmax=right_margin,
        colors="gray",
        linestyles="dotted",
        zorder=0,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_success_by_dose.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Success by Intervention Dose across Probes",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path

def plot_selectivity_ration_by_probe(dfs: dict[int, pd.DataFrame], dose: list[int] | int, save_dir: str | Path | None = None) -> Path:
    dose = [dose] if isinstance(dose, int) else dose
    dfs = _summarize(dfs, dose)
    
    results = []
    for d in dose:
        df = dfs[d]
        # df['selectivity_ratio'] = df.apply(lambda row: row['selectivity_ratio'] \
        #             if (row['successful']) else 0, axis=1)
        
        df = df.query('health_status == True ')
        df['norm_interaction_std'] = df['interaction_std'] / df['residual_std']
        df['norm_interaction'] = np.abs(df['norm_interaction'])

        print('dose', d)
        for probe in ['mean_diff', 'ttpd', 'svm', 'sawmil']:
            sub = df[df['probe'] == probe]
            theta, se, tau2 = random_effects_meta(
            sub['norm_interaction'].values,
            sub['norm_interaction_std'].values
        )
            print(f"  {probe:<10}: θ={theta:.6f} CI=({theta-1.96*se:.6f}, {theta+1.96*se:.6f})")

        print('=========')
        
        df_median = df.groupby(['probe']).agg({'norm_interaction': 'mean'}).reset_index()
        
        ci_by_probe = df.groupby('probe').apply(
            lambda x: pd.Series(bootstrap_ci(x['norm_interaction'], stat=np.mean, n_boot=5000), index=['ci_low', 'ci_high'])
        ).reset_index()
        
        
        for probe in ['mean_diff', 'ttpd', 'svm', 'sawmil']:
            sub = df_median[df_median['probe'] == probe]
            if not sub.empty:
                mu = float(sub['norm_interaction'].iloc[0])
                ci_low = float(ci_by_probe[(ci_by_probe['probe'] == probe)]['ci_low'].iloc[0])
                ci_high = float(ci_by_probe[(ci_by_probe['probe'] == probe)]['ci_high'].iloc[0])
                results.append({
                    'dose': d,
                    'probe': probe,
                    'mu': mu,
                    'ci_low': ci_low,
                    'ci_high': ci_high
                })
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # 2. Plot success rates with confidence intervals
    _setup_style()
    save_dir = Path(save_dir) if save_dir is not None else SAVE_DIR

    required_cols = {"dose", "probe", "mu", "ci_low", "ci_high"}
    missing_cols = required_cols - set(df_results.columns)
    if missing_cols:
        raise ValueError(
            f"df_results is missing required columns: {sorted(missing_cols)}"
        )

    valid_conditions = set(dose)
    df_agg = df_results[df_results["dose"].isin(valid_conditions)].copy()

    probe_order = [p for p in PROBE_ORDER if p in df_agg["probe"].unique()]
    if not probe_order:
        raise ValueError("No probes with valid condition data to plot.")

    dose_order = [
        d for d in dose if d in df_agg["dose"].unique()
    ]
    if not dose_order:
        raise ValueError("No recognized doses present after filtering.")

    n_probes = len(probe_order)
    bar_width = 0.24
    bar_gap = 0.10
    half_gap = bar_gap / 2
    group_gap = 0.30
    group_width = 2 * (bar_width + half_gap)
    xtick_centers = np.arange(n_probes) * (group_width + group_gap)

    palette = {dose: DOSE_COLOR[dose] for dose in dose_order}

    fig, ax = plt.subplots(figsize=(3, 2.5))
    legend_handles: list[plt.Artist] = []
    legend_labels: list[str] = []
    seen: set[str] = set()

    for probe_idx, probe in enumerate(probe_order):
        center = xtick_centers[probe_idx]
        for dose_idx, dose in enumerate(dose_order):
            subset = df_agg[(df_agg["probe"] == probe) & (df_agg["dose"] == dose)]
            if subset.empty:
                continue

            sign = -1 if dose_idx == 0 else 1
            pos = center + sign * (bar_width / 2 + half_gap)
            mu_val = subset["mu"].iloc[0]
            ci_low_val = subset["ci_low"].iloc[0]
            ci_high_val = subset["ci_high"].iloc[0]
            bar = ax.bar(
                pos,
                mu_val,
                bar_width,
                yerr=[[np.clip(mu_val - ci_low_val, 0, None)], [np.clip(ci_high_val - mu_val, 0, None)]],
                capsize=3,
                color=palette[dose],
                linewidth=1.3,
                edgecolor="black",
            )
            label = ["±{d}".format(d=d) for d in dose_order][dose_idx]
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
    ax.set_ylabel("Normalized Interaction Effect (Median)")
    ax.set_xlabel("Probe")
    ax.set_title("Normalized Interaction Effect by Dose and Probe")
    span = bar_width + half_gap
    left_margin = xtick_centers[0] - span - 0.2
    right_margin = xtick_centers[-1] + span + 0.1
    ax.set_xlim(left_margin, right_margin)
    ax.legend(
        legend_handles,
        legend_labels,
        title="Dose",
        loc="upper left",
        bbox_to_anchor=(0.02, 1.02),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.6,
        borderaxespad=0.4,
        handlelength=1.2,
    )


    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "causal_selectivity_ratio_by_dose.pdf"
    fig.tight_layout(pad=0.02)
    fig.savefig(
        out_path,
        **SAVEFIG_OPTS,
        metadata={
            "Title": "Selectivity Ratio by Intervention Dose across Probes",
            "Creator": "Germans Savcisens",
        },
    )
    plt.close(fig)
    return out_path    
    
    
    


