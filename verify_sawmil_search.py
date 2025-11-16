#!/usr/bin/env python
"""
Manual verification script for sawmil runner with search option.
This script tests the sawmil runner with both search=True and search=False
to ensure the functionality works correctly.
"""

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from runners.runner_sawmil import SawmilProbeRunner
import sys


def create_synthetic_bags(n_bags=50, bag_sizes_range=(10, 30), n_features=128):
    """Create simple synthetic bag data for testing."""
    np.random.seed(42)
    bags = []
    y = []
    
    for i in range(n_bags):
        bag_size = np.random.randint(bag_sizes_range[0], bag_sizes_range[1])
        bag = np.random.randn(bag_size, n_features)
        
        # Make positive bags have shifted last instances
        if i % 2 == 0:
            bag[-2:, 0] += 2.0  # Shift last 2 instances
            y.append(1)
        else:
            y.append(0)
        
        bags.append(bag)
    
    return bags, np.array(y)


def test_without_search():
    """Test sawmil runner without hyperparameter search."""
    print("\n" + "="*70)
    print("TEST 1: Sawmil Runner WITHOUT Search (search=False)")
    print("="*70)
    
    bags, y = create_synthetic_bags(n_bags=50)
    mask = np.ones(len(y), dtype=bool)
    
    cfg = OmegaConf.create({
        'probe': {
            'name': 'sawmil',
            'max_bag_size': 25,
            'assume_known_positives': True,
            'num_known_positives': 2,
            'train_bag_limit': 40,
            'normalize_data': True,
            'init_params': {
                'kernel': 'linear',
                'C': 1.0,
                'scale_C': True,
                'verbose': False,
                'sv_cutoff': 1e-7
            }
        },
        'random_seed': 42,
        'conformal_params': {
            'nc': 'binary',
            'alpha': 0.1,
            'tie_breaking': 'random'
        }
    })
    
    try:
        runner = SawmilProbeRunner(cfg=cfg)
        result = runner.single_training(X=bags, y=y, mask=mask, layer_id=0)
        
        print(f"✓ Training completed successfully")
        print(f"  - Eta: {result['eta']:.3f}")
        print(f"  - Separator trained: {result['separator'] is not None}")
        
        # Test predictions
        test_bags = bags[:10]
        predictions = runner.predict(test_bags)
        scores = runner.decision_function(test_bags)
        
        print(f"✓ Predictions generated successfully")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - Predictions: {predictions[:5]}")
        print(f"  - Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        print("\n✅ TEST 1 PASSED: Sawmil runner works without search")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_with_search():
    """Test sawmil runner with hyperparameter search."""
    print("\n" + "="*70)
    print("TEST 2: Sawmil Runner WITH Search (search=True)")
    print("="*70)
    
    bags, y = create_synthetic_bags(n_bags=60)
    mask = np.ones(len(y), dtype=bool)
    
    cfg = OmegaConf.create({
        'probe': {
            'name': 'sawmil',
            'max_bag_size': 25,
            'assume_known_positives': True,
            'num_known_positives': 2,
            'train_bag_limit': 50,
            'normalize_data': True,
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
        'cv_bag_limit': 40,
        'conformal_params': {
            'nc': 'binary',
            'alpha': 0.1,
            'tie_breaking': 'random'
        }
    })
    
    try:
        runner = SawmilProbeRunner(cfg=cfg)
        result = runner.parameter_search(X=bags, y=y, mask=mask, layer_id=0)
        
        print(f"✓ Parameter search completed successfully")
        print(f"  - Best C: {result['best_C']}")
        print(f"  - Available C values: {cfg.probe.param_grid.C}")
        print(f"  - Separator trained: {result['separator'] is not None}")
        
        # Test predictions
        test_bags = bags[:10]
        predictions = runner.predict(test_bags)
        scores = runner.decision_function(test_bags)
        
        print(f"✓ Predictions generated successfully")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - Predictions: {predictions[:5]}")
        print(f"  - Scores range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        print("\n✅ TEST 2 PASSED: Sawmil runner works with search")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """Compare results with and without search."""
    print("\n" + "="*70)
    print("TEST 3: Comparison of search vs no-search")
    print("="*70)
    
    bags, y = create_synthetic_bags(n_bags=60)
    
    # Split into train and test
    n_train = 40
    train_bags = bags[:n_train]
    train_y = y[:n_train]
    train_mask = np.ones(n_train, dtype=bool)
    
    test_bags = bags[n_train:]
    test_y = y[n_train:]
    
    cfg = OmegaConf.create({
        'probe': {
            'name': 'sawmil',
            'max_bag_size': 25,
            'assume_known_positives': True,
            'num_known_positives': 2,
            'train_bag_limit': 35,
            'normalize_data': True,
            'init_params': {
                'kernel': 'linear',
                'C': 1.0,
                'scale_C': True,
                'verbose': False,
                'sv_cutoff': 1e-7
            },
            'param_grid': {
                'C': [0.1, 1.0]
            }
        },
        'random_seed': 42,
        'cv_n_folds': 2,
        'cv_bag_limit': 30,
        'conformal_params': {
            'nc': 'binary',
            'alpha': 0.1,
            'tie_breaking': 'random'
        }
    })
    
    try:
        # Without search
        runner1 = SawmilProbeRunner(cfg=cfg)
        runner1.single_training(X=train_bags, y=train_y, mask=train_mask, layer_id=0)
        preds1 = runner1.predict(test_bags)
        acc1 = np.mean(preds1 == test_y)
        
        # With search
        runner2 = SawmilProbeRunner(cfg=cfg)
        result2 = runner2.parameter_search(X=train_bags, y=train_y, mask=train_mask, layer_id=0)
        preds2 = runner2.predict(test_bags)
        acc2 = np.mean(preds2 == test_y)
        
        print(f"✓ Both methods completed successfully")
        print(f"  Without search:")
        print(f"    - Default C: {cfg.probe.init_params.C}")
        print(f"    - Test accuracy: {acc1:.3f}")
        print(f"  With search:")
        print(f"    - Selected C: {result2['best_C']}")
        print(f"    - Test accuracy: {acc2:.3f}")
        
        print("\n✅ TEST 3 PASSED: Comparison completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("SAWMIL RUNNER VERIFICATION SCRIPT")
    print("Testing search functionality (search=True and search=False)")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Without search", test_without_search()))
    results.append(("With search", test_with_search()))
    results.append(("Comparison", test_comparison()))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} : {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The sawmil runner works correctly with search.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED! Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
