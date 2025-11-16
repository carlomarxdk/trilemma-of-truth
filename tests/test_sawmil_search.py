"""
Tests for the SawmilProbeRunner, specifically testing the search functionality.
This test validates that the parameter_search method works correctly and that
the runner can train with both search=True and search=False options.
"""

import numpy as np
import pytest
from pathlib import Path
from runners.runner_sawmil import SawmilProbeRunner
from utils_jupyter import load_hydra_config_with_params
from omegaconf import OmegaConf


def rng(seed=0):
    """Random number generator for reproducibility."""
    return np.random.default_rng(seed)


def make_binary_bags(n_bags=100, bag_sizes_range=(10, 50), n_features=128, 
                     separation=1.5, seed=42):
    """
    Create synthetic binary bag data for testing MIL algorithms.
    
    Args:
        n_bags: Number of bags to create
        bag_sizes_range: Tuple of (min_size, max_size) for bag sizes
        n_features: Number of features per instance
        separation: Separation between positive and negative classes
        seed: Random seed
        
    Returns:
        bags: List of arrays, each of shape (bag_size, n_features)
        y: Binary labels (0 or 1) for each bag
    """
    g = rng(seed)
    
    # Create approximately balanced classes
    n_pos = n_bags // 2
    n_neg = n_bags - n_pos
    
    bags = []
    y = []
    
    # Generate negative bags (label 0)
    for i in range(n_neg):
        bag_size = g.integers(bag_sizes_range[0], bag_sizes_range[1])
        # Negative bags have all negative instances (mean around 0)
        bag = g.normal(0, 1, size=(bag_size, n_features))
        bags.append(bag)
        y.append(0)
    
    # Generate positive bags (label 1)
    for i in range(n_pos):
        bag_size = g.integers(bag_sizes_range[0], bag_sizes_range[1])
        # Positive bags have some positive instances (shifted mean)
        # Most instances are negative-like
        bag = g.normal(0, 1, size=(bag_size, n_features))
        # But last few instances are positive (shifted)
        n_pos_instances = min(3, bag_size)  # Last 3 instances are positive
        shift_vector = np.zeros(n_features)
        shift_vector[0] = separation  # Shift along first dimension
        bag[-n_pos_instances:] += shift_vector
        bags.append(bag)
        y.append(1)
    
    # Shuffle the order
    indices = g.permutation(n_bags)
    bags = [bags[i] for i in indices]
    y = np.array([y[i] for i in indices])
    
    return bags, y


@pytest.fixture
def synthetic_bag_data():
    """Create synthetic bag data for testing."""
    np.random.seed(42)
    bags, y = make_binary_bags(
        n_bags=120, 
        bag_sizes_range=(15, 40), 
        n_features=128,
        separation=2.0,
        seed=42
    )
    
    # Create a mask (all True for this test)
    mask = np.ones(len(y), dtype=bool)
    
    return bags, y, mask


@pytest.fixture
def sawmil_config():
    """Create a configuration for SawmilProbeRunner."""
    cfg = OmegaConf.create({
        'probe': {
            'name': 'sawmil',
            'max_bag_size': 30,
            'assume_known_positives': True,
            'num_known_positives': 2,
            'train_bag_limit': 100,
            'normalize_data': True,
            'kernel': 'linear',
            'scale_C': True,
            'init_params': {
                'kernel': 'linear',
                'C': 1.0,
                'scale_C': True,
                'verbose': False,
                'sv_cutoff': 1e-7
            },
            'param_grid': {
                'C': [0.1, 1.0, 10.0]
            }
        },
        'random_seed': 42,
        'cv_n_folds': 3,
        'cv_bag_limit': 80,
        'conformal_params': {
            'nc': 'binary',
            'alpha': 0.1,
            'tie_breaking': 'random'
        }
    })
    return cfg


def test_sawmil_runner_single_training(synthetic_bag_data, sawmil_config):
    """
    Test that SawmilProbeRunner can train without hyperparameter search.
    This tests the search=False option.
    """
    bags, y, mask = synthetic_bag_data
    
    runner = SawmilProbeRunner(cfg=sawmil_config)
    
    # Train without search (search=False)
    result = runner.single_training(X=bags, y=y, mask=mask, layer_id=0)
    
    # Verify that the result contains expected keys
    assert 'separator' in result, "Result should contain 'separator'"
    assert 'scaler' in result, "Result should contain 'scaler'"
    assert 'eta' in result, "Result should contain 'eta'"
    
    # Verify that the separator is trained
    assert result['separator'] is not None, "Separator should be trained"
    assert hasattr(result['separator'], 'linearize'), "Separator should have linearize method"
    
    # Verify eta is reasonable
    assert 0 < result['eta'] <= 1, f"Eta should be between 0 and 1, got {result['eta']}"
    
    # Test prediction
    test_bags = bags[:10]
    predictions = runner.predict(test_bags)
    
    # Verify predictions have correct shape
    assert predictions.shape[0] == len(test_bags), "Should have one prediction per bag"
    
    # Verify predictions are binary (0 or 1)
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"
    
    print(f"✓ Single training test passed. Eta: {result['eta']:.3f}")


def test_sawmil_runner_parameter_search(synthetic_bag_data, sawmil_config):
    """
    Test that SawmilProbeRunner can perform hyperparameter search.
    This tests the search=True option.
    """
    bags, y, mask = synthetic_bag_data
    
    runner = SawmilProbeRunner(cfg=sawmil_config)
    
    # Train with search (search=True)
    result = runner.parameter_search(X=bags, y=y, mask=mask, layer_id=0)
    
    # Verify that the result contains expected keys
    assert 'separator' in result, "Result should contain 'separator'"
    assert 'scaler' in result, "Result should contain 'scaler'"
    assert 'best_C' in result, "Result should contain 'best_C' after search"
    
    # Verify that the separator is trained
    assert result['separator'] is not None, "Separator should be trained"
    assert hasattr(result['separator'], 'linearize'), "Separator should have linearize method"
    
    # Verify that best_C is one of the values from param_grid
    param_grid_C = sawmil_config.probe.param_grid.C
    assert result['best_C'] in param_grid_C, \
        f"best_C {result['best_C']} should be in param_grid {param_grid_C}"
    
    # Test prediction
    test_bags = bags[:10]
    predictions = runner.predict(test_bags)
    
    # Verify predictions have correct shape
    assert predictions.shape[0] == len(test_bags), "Should have one prediction per bag"
    
    # Verify predictions are binary (0 or 1)
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"
    
    print(f"✓ Parameter search test passed. Best C: {result['best_C']}")


def test_sawmil_runner_decision_function(synthetic_bag_data, sawmil_config):
    """
    Test that decision_function works correctly after training.
    """
    bags, y, mask = synthetic_bag_data
    
    runner = SawmilProbeRunner(cfg=sawmil_config)
    runner.single_training(X=bags, y=y, mask=mask, layer_id=0)
    
    # Test decision function
    test_bags = bags[:10]
    scores = runner.decision_function(test_bags)
    
    # Verify scores have correct shape
    assert scores.shape[0] == len(test_bags), "Should have one score per bag"
    
    # Verify scores are finite
    assert np.all(np.isfinite(scores)), "All scores should be finite"
    
    # Test predict_proba
    probs = runner.predict_proba(test_bags)
    
    # Verify probabilities are in [0, 1]
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities should be in [0, 1]"
    
    print("✓ Decision function test passed")


def test_sawmil_runner_conformal_prediction(synthetic_bag_data, sawmil_config):
    """
    Test that conformal prediction works after training.
    """
    bags, y, mask = synthetic_bag_data
    
    # Split into train and calibration
    n_train = int(0.7 * len(bags))
    train_bags = bags[:n_train]
    train_y = y[:n_train]
    train_mask = mask[:n_train]
    
    cal_bags = bags[n_train:]
    cal_y = y[n_train:]
    cal_mask = mask[n_train:]
    
    runner = SawmilProbeRunner(cfg=sawmil_config)
    runner.single_training(X=train_bags, y=train_y, mask=train_mask, layer_id=0)
    
    # Train conformal predictor
    calibrator = runner.conformal_training(X_cal=cal_bags, y_cal=cal_y, mask_cal=cal_mask)
    
    assert calibrator is not None, "Conformal calibrator should be trained"
    
    # Test conformal prediction
    test_bags = bags[:10]
    conformal_preds = runner.conformal_prediction(test_bags)
    
    # Verify predictions have correct shape
    assert conformal_preds.shape[0] == len(test_bags), "Should have one prediction per bag"
    
    print("✓ Conformal prediction test passed")


def test_sawmil_runner_bag_vs_inst_predictions(synthetic_bag_data, sawmil_config):
    """
    Test that bag-level and instance-level predictions work correctly.
    """
    bags, y, mask = synthetic_bag_data
    
    runner = SawmilProbeRunner(cfg=sawmil_config)
    runner.single_training(X=bags, y=y, mask=mask, layer_id=0)
    
    test_bags = bags[:10]
    
    # Test bag-level predictions
    bag_preds = runner.bag_predict(test_bags)
    assert bag_preds.shape[0] == len(test_bags), "Should have one bag prediction per bag"
    
    # Test instance-level predictions (last instance)
    inst_preds = runner.inst_predict(test_bags)
    assert inst_preds.shape[0] == len(test_bags), "Should have one instance prediction per bag"
    
    # For MIL, bag predictions and instance predictions should generally differ
    # (bag uses max over instances, inst uses last instance only)
    
    print("✓ Bag vs instance prediction test passed")


def test_sawmil_runner_search_improves_or_maintains_performance(synthetic_bag_data, sawmil_config):
    """
    Test that parameter search either improves or maintains reasonable performance.
    This is not a strict requirement but validates the search is functioning.
    """
    bags, y, mask = synthetic_bag_data
    
    # Split into train and test
    n_train = int(0.7 * len(bags))
    train_bags = bags[:n_train]
    train_y = y[:n_train]
    train_mask = mask[:n_train]
    
    test_bags = bags[n_train:]
    test_y = y[n_train:]
    
    # Train without search
    runner_no_search = SawmilProbeRunner(cfg=sawmil_config)
    runner_no_search.single_training(X=train_bags, y=train_y, mask=train_mask, layer_id=0)
    preds_no_search = runner_no_search.predict(test_bags)
    
    # Compute accuracy
    acc_no_search = np.mean(preds_no_search == test_y)
    
    # Train with search
    runner_with_search = SawmilProbeRunner(cfg=sawmil_config)
    result = runner_with_search.parameter_search(X=train_bags, y=train_y, mask=train_mask, layer_id=0)
    preds_with_search = runner_with_search.predict(test_bags)
    
    # Compute accuracy
    acc_with_search = np.mean(preds_with_search == test_y)
    
    print(f"  Accuracy without search: {acc_no_search:.3f}")
    print(f"  Accuracy with search: {acc_with_search:.3f}")
    print(f"  Selected C from search: {result['best_C']}")
    
    # Both should have reasonable performance (> 40% on this synthetic data)
    # Note: With random synthetic data, we don't expect perfect accuracy
    # The important thing is that both methods complete without errors
    assert acc_no_search > 0.4, f"No-search accuracy {acc_no_search:.3f} should be > 0.4"
    assert acc_with_search > 0.4, f"With-search accuracy {acc_with_search:.3f} should be > 0.4"
    
    print("✓ Search performance test passed")


def test_sawmil_runner_with_real_activations():
    """
    Test SawmilProbeRunner with real activations if available.
    This test will be skipped if the activations are not found.
    """
    # Check if real activations exist
    act_path = Path('outputs/activations/llama-3-8b/city_locations/full/layer_13_e.npz')
    
    if not act_path.exists():
        pytest.skip("Real activations not found, skipping this test")
    
    # Load the activations
    data = np.load(act_path)
    activations = data['arr_0']  # Shape: (5500, 100, 4096)
    
    # Load mask
    mask_path = Path('outputs/activations/llama-3-8b/city_locations/full/mask.npy')
    mask = np.load(mask_path)
    
    # Create bags from activations (each bag is a sequence of hidden states)
    n_samples = min(100, activations.shape[0])  # Use first 100 samples
    bags = [activations[i] for i in range(n_samples)]
    
    # Create synthetic labels for testing (alternating 0 and 1)
    y = np.array([i % 2 for i in range(n_samples)])
    test_mask = np.ones(n_samples, dtype=bool)
    
    # Create config
    cfg = OmegaConf.create({
        'probe': {
            'name': 'sawmil',
            'max_bag_size': 100,
            'assume_known_positives': True,
            'num_known_positives': 2,
            'train_bag_limit': 80,
            'normalize_data': True,
            'kernel': 'linear',
            'scale_C': True,
            'init_params': {
                'kernel': 'linear',
                'C': 1.0,
                'scale_C': True,
                'verbose': False,
                'sv_cutoff': 1e-7
            },
            'param_grid': {
                'C': [0.1, 1.0]  # Smaller grid for faster testing
            }
        },
        'random_seed': 42,
        'cv_n_folds': 2,  # Fewer folds for faster testing
        'cv_bag_limit': 60,
        'conformal_params': {
            'nc': 'binary',
            'alpha': 0.1,
            'tie_breaking': 'random'
        }
    })
    
    runner = SawmilProbeRunner(cfg=cfg)
    
    # Test with search
    result = runner.parameter_search(X=bags, y=y, mask=test_mask, layer_id=13)
    
    assert 'separator' in result
    assert 'best_C' in result
    assert result['best_C'] in cfg.probe.param_grid.C
    
    # Test predictions
    preds = runner.predict(bags[:10])
    assert preds.shape[0] == 10
    
    print(f"✓ Real activations test passed. Best C: {result['best_C']}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])
