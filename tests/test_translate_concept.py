"""Test translate_concept function to verify correct broadcasting."""

import torch
from response.interventions_utils import translate_concept


def test_translate_concept_2d():
    """Test that translate_concept works correctly with 2D input [B, H]."""
    batch_size = 4
    hidden_dim = 768
    delta = 2.5
    
    # Create random inputs
    X = torch.randn(batch_size, hidden_dim)
    direction = torch.randn(hidden_dim)
    
    # Apply translation
    result = translate_concept(X, direction, delta)
    
    # Verify shape preserved
    assert result.shape == X.shape, f"Expected shape {X.shape}, got {result.shape}"
    
    # Verify the translation is correct
    unit_dir = direction / direction.norm()
    expected = X + delta * unit_dir
    
    assert torch.allclose(result, expected, atol=1e-6), "Translation result doesn't match expected"
    
    # Verify it actually changed the values
    assert not torch.allclose(result, X, atol=1e-6), "Translation didn't modify the input"
    
    print("✓ 2D translation test passed")


def test_translate_concept_3d():
    """Test that translate_concept works correctly with 3D input [B, S, H]."""
    batch_size = 4
    seq_len = 10
    hidden_dim = 768
    delta = 2.5
    
    # Create random inputs
    X = torch.randn(batch_size, seq_len, hidden_dim)
    direction = torch.randn(hidden_dim)
    
    # Apply translation
    result = translate_concept(X, direction, delta)
    
    # Verify shape preserved
    assert result.shape == X.shape, f"Expected shape {X.shape}, got {result.shape}"
    
    # Verify the translation is correct
    unit_dir = direction / direction.norm()
    expected = X + delta * unit_dir
    
    assert torch.allclose(result, expected, atol=1e-6), "Translation result doesn't match expected"
    
    # Verify it actually changed the values
    assert not torch.allclose(result, X, atol=1e-6), "Translation didn't modify the input"
    
    print("✓ 3D translation test passed")


def test_intervention_usage_pattern():
    """Test the exact usage pattern from run_intervention.py."""
    # Simulate the actual usage: h[:, _start_token, :] is shape [B, H]
    batch_size = 1  # Usually 1 in inference
    seq_len = 20
    hidden_dim = 768
    start_token = -5
    delta = 2.5
    
    # Simulate layer output
    h = torch.randn(batch_size, seq_len, hidden_dim)
    direction = torch.randn(hidden_dim)
    
    # Store original token before intervention
    original_token = h[:, start_token, :].clone()
    
    # Apply intervention exactly as in run_intervention.py
    h[:, start_token, :] = translate_concept(
        h[:, start_token, :],
        direction,
        delta
    )
    
    # Verify the target token was modified
    assert not torch.allclose(h[:, start_token, :], original_token, atol=1e-6), \
        "Target token wasn't modified"
    
    # Verify the translation magnitude and direction are correct
    unit_dir = direction / direction.norm()
    expected_result = original_token + delta * unit_dir
    
    assert torch.allclose(h[:, start_token, :], expected_result, atol=1e-5), \
        "Translation didn't produce expected result"
    
    # Verify the shift has the correct magnitude
    actual_shift = h[:, start_token, :] - original_token
    assert torch.isclose(actual_shift.norm(), torch.tensor(delta), atol=1e-5), \
        f"Shift magnitude {actual_shift.norm()} doesn't match delta {delta}"
    
    print("✓ Intervention usage pattern test passed")


def test_direction_normalization():
    """Verify that the direction vector is properly normalized."""
    batch_size = 4
    hidden_dim = 768
    delta = 2.5
    
    X = torch.randn(batch_size, hidden_dim)
    direction = torch.randn(hidden_dim) * 100  # Unnormalized direction
    
    result = translate_concept(X, direction, delta)
    
    # The translation should be delta * unit_direction
    unit_dir = direction / direction.norm()
    shift = result - X
    
    # Verify the shift has the correct magnitude (should be exactly delta)
    shift_norm = shift[0].norm()  # Same for all batch elements
    assert torch.isclose(shift_norm, torch.tensor(delta), atol=1e-5), \
        f"Shift magnitude {shift_norm} doesn't match delta {delta}"
    
    print("✓ Direction normalization test passed")


if __name__ == "__main__":
    print("Testing translate_concept function...")
    print()
    
    test_translate_concept_2d()
    test_translate_concept_3d()
    test_intervention_usage_pattern()
    test_direction_normalization()
    
    print()
    print("✅ All tests passed! The translate_concept function works correctly.")
