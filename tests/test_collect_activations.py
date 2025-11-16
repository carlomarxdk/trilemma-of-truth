import numpy as np
import pytest
import tempfile
import os
from pathlib import Path


def test_memmap_indexing_for_last_aggregation():
    """Test that 'last' aggregation uses correct 2D indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate the 'last' aggregation scenario
        num_statements = 10
        hidden_size = 128
        batch_size = 3
        
        # Create a 2D memmap like in the 'last' case
        save_path = os.path.join(tmpdir, "test_last.npy")
        acts_memmap = np.memmap(save_path, dtype='float16', mode='w+',
                               shape=(num_statements, hidden_size))
        
        # Simulate embeddings for one batch (output[:, -1] produces 2D array)
        embeddings = np.random.randn(batch_size, hidden_size).astype(np.float16)
        
        # Test the indexing works correctly (2D indexing for 'last')
        _last_row = 0
        for i in range(batch_size):
            # This should work without error for 2D memmap
            acts_memmap[_last_row + i, :] = embeddings[i]
        
        acts_memmap.flush()
        
        # Verify the data was written correctly
        loaded = np.memmap(save_path, dtype='float16', mode='r',
                          shape=(num_statements, hidden_size))
        np.testing.assert_array_almost_equal(loaded[:batch_size], embeddings, decimal=3)


def test_memmap_indexing_for_full_aggregation():
    """Test that 'full' aggregation uses correct 3D indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate the 'full' aggregation scenario
        num_statements = 10
        max_length = 50
        hidden_size = 128
        batch_size = 3
        
        # Create a 3D memmap like in the 'full' case
        save_path = os.path.join(tmpdir, "test_full.npy")
        acts_memmap = np.memmap(save_path, dtype='float16', mode='w+',
                               shape=(num_statements, max_length, hidden_size))
        
        # Simulate embeddings for one batch (output produces 3D array)
        embeddings = np.random.randn(batch_size, max_length, hidden_size).astype(np.float16)
        
        # Test the indexing works correctly (3D indexing for 'full')
        _last_row = 0
        for i in range(batch_size):
            # This should work without error for 3D memmap
            acts_memmap[_last_row + i, :, :] = embeddings[i]
        
        acts_memmap.flush()
        
        # Verify the data was written correctly
        loaded = np.memmap(save_path, dtype='float16', mode='r',
                          shape=(num_statements, max_length, hidden_size))
        np.testing.assert_array_almost_equal(loaded[:batch_size], embeddings, decimal=3)


def test_wrong_indexing_fails_for_last_aggregation():
    """Test that using 3D indexing on 2D memmap fails (the bug we're fixing)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate the 'last' aggregation scenario with 2D memmap
        num_statements = 10
        hidden_size = 128
        batch_size = 3
        
        save_path = os.path.join(tmpdir, "test_wrong.npy")
        acts_memmap = np.memmap(save_path, dtype='float16', mode='w+',
                               shape=(num_statements, hidden_size))
        
        # Simulate embeddings for 'last' (2D)
        embeddings = np.random.randn(batch_size, hidden_size).astype(np.float16)
        
        # Using 3D indexing on 2D memmap should fail
        _last_row = 0
        with pytest.raises(IndexError):
            for i in range(batch_size):
                # This is the bug: using 3D indexing [:, :] on 2D array
                acts_memmap[_last_row + i, :, :] = embeddings[i]
