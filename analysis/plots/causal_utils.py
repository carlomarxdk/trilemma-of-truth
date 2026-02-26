
"""Pool intervention results across all probes, models, and datasets into a single dataset.

This script aggregates all intervention results from outputs/interv-{dose}/ into a comprehensive
dataframe for analysis and exploration.
"""
from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from statsmodels.stats.multitest import multipletests

log = logging.getLogger(__name__)


def get_all_intervention_results(dose: int) -> Path:
    """Get the path to the interventions directory."""
    return Path(f"outputs/interv-{dose}")


def get_intervention_structure(dose: int) -> list[tuple[str, str, str, int]]:
    """Discover all probe/model/dataset/task combinations.
    
    Returns:
        List of tuples (probe, model, dataset, task).
    """
    interv_dir = get_all_intervention_results(dose)
    combinations = []
    
    # Iterate: probes -> models -> datasets
    for probe_dir in interv_dir.iterdir():
        if not probe_dir.is_dir():
            continue
        probe = probe_dir.name
        
        for model_dir in probe_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            for dataset_dir in model_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                dataset_name = dataset_dir.name
                
                # Extract task from dataset directory name
                # Format: dataset_name_task-X or dataset_name_search_task-X
                if "_task-" in dataset_name:
                    parts = dataset_name.split("_task-")
                    task = int(parts[-1])
                    combinations.append((probe, model, dataset_name, task))
    
    return combinations


def load_layer_results(layer_dir: Path, layer_id: int) -> dict | None:
    """Load intervention results for a specific layer.
    
    Args:
        layer_dir: Path to the dataset directory containing layer_*.json files.
        layer_id: The layer ID to load.
    
    Returns:
        Dictionary with layer results, or None if not found.
    """
    layer_file = layer_dir / f"layer_{layer_id}.json"
    if not layer_file.exists():
        return None
    
    try:
        with open(layer_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Error loading {layer_file}: {e}")
        return None


def get_available_layers(layer_dir: Path) -> list[int]:
    """Get list of layer IDs with intervention results."""
    layer_files = list(layer_dir.glob("layer_*.json"))
    layers = []
    for f in layer_files:
        layer_num = f.stem.replace("layer_", "")
        if layer_num.isdigit():
            layers.append(int(layer_num))
    return sorted(layers)


def create_pooled_dataset(dose:int=1) -> pd.DataFrame:
    """Create a pooled dataset of all intervention results.
    
    Returns:
        DataFrame with one row per (probe, model, dataset, layer).
    """
    rows = []
    combinations = get_intervention_structure(dose)
    
    log.info(f"Found {len(combinations)} probe/model/dataset combinations")
    
    for probe, model, dataset_name, task in combinations:
        dataset_dir = Path(f"outputs/interv-{dose}/{probe}/{model}/{dataset_name}")
        
        if not dataset_dir.exists():
            continue
        
        log.info(f"Processing: {probe} / {model} / {dataset_name}")
        
        available_layers = get_available_layers(dataset_dir)
        
        for layer_id in available_layers:
            results = load_layer_results(dataset_dir, layer_id)
            
            if results is None:
                continue
            
            did = results.get('did', {})
            success = results.get('success_results', {})
            descriptives = results.get('descriptives', {})
            unidir = results.get('unidir_results', {})
            delta_stats = results.get('delta_stats', {})
            asymmetry = results.get('asymmetry', {})
            
            # Extract unidirectional results
            unidir_pos = unidir.get('positive', {})
            unidir_neg = unidir.get('negative', {})
            
            # Extract asymmetry statistics
            correct_asym = asymmetry.get('correct_asymmetry', {})
            random_asym = asymmetry.get('random_asymmetry', {})
            diff_asym = asymmetry.get('differential_asymmetry', {})
            
            row = {
                # Identifiers
                'probe': probe,
                'model': model,
                'dataset': dataset_name,
                'layer': layer_id,
                'task': task,
                
                # DiD results
                'interaction_coef': did.get('interaction_coef'),
                'interaction_std': did.get('interaction_std'),
                'interaction_pval': did.get('interaction_pval'),
                'interaction_zval': did.get('interaction_zval'),
                'interaction_ci_lower': did.get('interaction_ci', [None, None])[0],
                'interaction_ci_upper': did.get('interaction_ci', [None, None])[1],
                'significant': bool(did.get('interaction_signf', 0)),
                'r_squared': did.get('r_squared'),
                'n_statements': did.get('n_statements'),
                'n_obs': did.get('n_obs'),
                'df_resid': did.get('df_resid'),
                'residual_std': did.get('residual_std'),
                'condition_number': did.get('condition_number'),
                
                # Main effects
                'token_coef': did.get('token_coef'),
                'token_std': did.get('token_std'),
                'token_pval': did.get('token_pval'),
                'translation_coef': did.get('translation_coef'),
                'translation_std': did.get('translation_std'),
                'translation_pval': did.get('translation_pval'),
                'intercept_coef': did.get('intercept_coef'),
                'intercept_std': did.get('intercept_std'),
                'intercept_pval': did.get('intercept_pval'),
                
                # New normalized metrics from DiD
                'norm_interaction': did.get('norm_interaction'),
                'selectivity_ratio': did.get('selectivity_ratio'),
                
                # Success metrics
                'success_rate': success.get('success_rate'),
                'n_success': success.get('n_success'),
                'n_total': success.get('n_total'),
                'binomial_pval': success.get('p_value'),
                'rate_dominant': success.get('rate_dom'),
                'rate_opposite': success.get('rate_opp'),
                
                # Descriptive statistics
                'correct_orig_mean': descriptives.get('correct_orig_mean'),
                'correct_pos_mean': descriptives.get('correct_pos_mean'),
                'correct_neg_mean': descriptives.get('correct_neg_mean'),
                'random_orig_mean': descriptives.get('random_orig_mean'),
                'random_pos_mean': descriptives.get('random_pos_mean'),
                'random_neg_mean': descriptives.get('random_neg_mean'),
                
                # Unidirectional results - Positive direction
                'unidir_pos_intercept_coef': unidir_pos.get('intercept_coef'),
                'unidir_pos_intercept_std': unidir_pos.get('intercept_std'),
                'unidir_pos_intercept_pval': unidir_pos.get('intercept_pval'),
                'unidir_pos_token_coef': unidir_pos.get('token_coef'),
                'unidir_pos_token_std': unidir_pos.get('token_std'),
                'unidir_pos_token_pval': unidir_pos.get('token_pval'),
                'unidir_pos_r_squared': unidir_pos.get('r_squared'),
                'unidir_pos_n_obs': unidir_pos.get('n_obs'),
                
                # Unidirectional results - Negative direction
                'unidir_neg_intercept_coef': unidir_neg.get('intercept_coef'),
                'unidir_neg_intercept_std': unidir_neg.get('intercept_std'),
                'unidir_neg_intercept_pval': unidir_neg.get('intercept_pval'),
                'unidir_neg_token_coef': unidir_neg.get('token_coef'),
                'unidir_neg_token_std': unidir_neg.get('token_std'),
                'unidir_neg_token_pval': unidir_neg.get('token_pval'),
                'unidir_neg_r_squared': unidir_neg.get('r_squared'),
                'unidir_neg_n_obs': unidir_neg.get('n_obs'),
                
                # Delta statistics
                'delta_mean': delta_stats.get('mean'),
                'delta_std': delta_stats.get('std'),
                'delta_min': delta_stats.get('min'),
                'delta_max': delta_stats.get('max'),
                
                # Asymmetry - Correct accuracy
                'asym_correct_mean_pos': correct_asym.get('mean_pos'),
                'asym_correct_mean_neg': correct_asym.get('mean_neg'),
                'asym_correct_ratio': correct_asym.get('ratio'),
                'asym_correct_t_stat': correct_asym.get('t_stat'),
                'asym_correct_p_value': correct_asym.get('p_value'),
                
                # Asymmetry - Random accuracy
                'asym_random_mean_pos': random_asym.get('mean_pos'),
                'asym_random_mean_neg': random_asym.get('mean_neg'),
                'asym_random_ratio': random_asym.get('ratio'),
                'asym_random_t_stat': random_asym.get('t_stat'),
                'asym_random_p_value': random_asym.get('p_value'),
                
                # Asymmetry - Differential
                'asym_diff_t_stat': diff_asym.get('t_stat'),
                'asym_diff_p_value': diff_asym.get('p_value'),
                
                # health
                'health_status': results.get('did', {}).get('health', {}).get('is_healthy', None),
            }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Post-processing: extract raw dataset name, normalize metrics
    df['raw_dataset'] = df['dataset'].copy()
    df['dataset'] = df['dataset'].str.replace(r'(_search.*|_\d+)$', '', regex=True)
    df['selectivity_ratio'] = np.abs(df['selectivity_ratio'])
    df['interaction_coef'] = np.abs(df['interaction_coef'])
    
    log.info(f"Created pooled dataset with {len(df)} rows")
    
    return df


def get_median_row_per_experiment(df, performance_col='interaction_coef'):
    """Select row with median performance for each experiment.
    
    For each (probe, model, dataset) group, picks the layer whose 
    performance value is closest to the group's median.
    
    Args:
        df: DataFrame with experiment results.
        performance_col: Column to use for median calculation.
    
    Returns:
        DataFrame with one row per experiment.
    """
    result_rows = []
    
    for (probe, model, dataset), group in df.groupby(['probe', 'model', 'dataset']):
        # Calculate group median
        median_val = group[performance_col].median()
        
        # Find row closest to median
        group['distance_to_median'] = (group[performance_col] - median_val).abs()
        median_row = group.loc[group['distance_to_median'].idxmin()].copy()
        
        result_rows.append(median_row)
    
    return pd.DataFrame(result_rows).drop('distance_to_median', axis=1)


def apply_fdr_correction_by_probe(df: pd.DataFrame, pval_col: str = 'interaction_pval',
                                   alpha: float = 0.05) -> pd.DataFrame:
    """Apply FDR correction within each probe group.
    
    Applies Benjamini-Hochberg FDR correction separately for each probe,
    treating each probe as an independent family of tests.
    
    Args:
        df: DataFrame containing p-values and probe identifiers.
        pval_col: Column name containing p-values to adjust.
        alpha: Significance level for FDR correction.
    
    Returns:
        DataFrame with added 'pval_adjusted' column.
    
    Example:
        df = apply_fdr_correction_by_probe(df, pval_col='interaction_pval')
    """
    df = df.copy()
    df['pval_adjusted'] = np.nan
    
    for probe in df['probe'].unique():
        mask = df['probe'] == probe
        reject, pvals_corrected, _, _ = multipletests(
            df.loc[mask, pval_col],
            alpha=alpha,
            method='fdr_bh'
        )
        df.loc[mask, 'pval_adjusted'] = pvals_corrected
        df.loc[mask, 'significant'] = reject
        df.loc[mask, 'successful'] = reject & (df.loc[mask, 'health_status'] == 1)
    
    return df



def bootstrap_ci(x, stat=np.median, n_boot=5000, alpha=0.05):
    boots = np.random.choice(x, (n_boot, len(x)), replace=True)
    stats = np.apply_along_axis(stat, 1, boots)
    return np.quantile(stats, [alpha / 2, 1 - alpha / 2])