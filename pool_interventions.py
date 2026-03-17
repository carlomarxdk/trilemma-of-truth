"""Pool intervention results across all probes, models, and datasets into a single dataset.

This script aggregates all intervention results from outputs/interv/ into a comprehensive
dataframe for analysis and exploration.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PoolInterventions")


def get_all_intervention_results(dose) -> Path:
    """Get the path to the interventions directory."""
    return Path(f"outputs/interv-{dose}")


def get_intervention_structure(dose) -> List[Tuple[str, str, str, int]]:
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


def load_layer_results(layer_dir: Path, layer_id: int) -> Dict:
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


def get_available_layers(layer_dir: Path) -> List[int]:
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
    log.info(f"Created pooled dataset with {len(df)} rows")
    
    return df


def main():
    """Main entry point."""
    log.info("Pooling intervention results...")
    
    # Create pooled dataset
    df = create_pooled_dataset()
    
    
    # Save to CSV
    output_file = Path("outputs/interv_pooled_results.csv")
    df.to_csv(output_file, index=False)
    log.info(f"\nSaved pooled results to: {output_file}")
    
    # Also save as parquet for efficient reloading
    output_parquet = Path("outputs/interv_pooled_results.parquet")
    df.to_parquet(output_parquet, index=False)
    log.info(f"Saved pooled results to: {output_parquet}")
    
    # Save summary statistics by probe, model, dataset
    summary = df.groupby(['probe', 'model', 'dataset']).agg({
        'interaction_coef': ['mean', 'std', 'min', 'max'],
        'interaction_pval': ['mean', 'min', 'max'],
        'success_rate': ['mean', 'std'],
        'significant': ['sum', 'count'],
        'layer': 'count'
    }).round(6)
    
    summary_file = Path("outputs/interv_summary_by_config.csv")
    summary.to_csv(summary_file)
    log.info(f"Saved summary by configuration to: {summary_file}")
    
    return df


if __name__ == "__main__":

    df = main()
    print("\n" + "="*80)
    print("Pooled intervention results ready for analysis!")
    print("="*80)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
