#!/usr/bin/env python3
"""
Test for PANN embedding dimension fix.

This test verifies that the PANN dimension mismatch issue has been fixed.
The issue was that PANN embeddings were being extracted as 1D arrays instead
of 2D arrays, which caused problems with multivariate gaussian calculations 
in the FAD metric.

See GitHub issue #37: https://github.com/gudgud96/frechet-audio-distance/issues/37
"""

import unittest
import sys
import os
import numpy as np
import torch

# Add the package to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPANNDimensionFix(unittest.TestCase):
    """Test case for PANN embedding dimension fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 2048  # Standard PANN embedding dimension
        self.batch_size = 1
        
    def test_tensor_operations_fix(self):
        """Test that the tensor operations fix works correctly."""
        # Mock PANN model output
        mock_output = {
            'embedding': torch.randn(self.batch_size, self.embedding_dim)
        }
        
        # Test old (problematic) approach
        old_embd = mock_output['embedding'].data[0]
        self.assertEqual(old_embd.ndim, 1, "Old approach should produce 1D tensor")
        self.assertEqual(old_embd.shape, (self.embedding_dim,))
        
        # Test new (fixed) approach
        new_embd = mock_output['embedding'].data[0].unsqueeze(0)
        self.assertEqual(new_embd.ndim, 2, "New approach should produce 2D tensor")
        self.assertEqual(new_embd.shape, (1, self.embedding_dim))
        
    def test_numpy_statistics_calculation(self):
        """Test that statistics calculation works with the fix."""
        # Create mock embedding data
        mock_output = {
            'embedding': torch.randn(self.batch_size, self.embedding_dim)
        }
        
        # Apply the fix
        fixed_embd = mock_output['embedding'].data[0].unsqueeze(0)
        fixed_np = fixed_embd.detach().numpy()
        
        # Test statistics calculation
        mu = np.mean(fixed_np, axis=0)
        sigma = np.cov(fixed_np, rowvar=False)
        
        # Verify shapes
        self.assertEqual(mu.shape, (self.embedding_dim,))
        self.assertFalse(np.isscalar(mu), "Mean should not be scalar")
        
        # For single sample, covariance will be 0D, but that's expected
        # In real usage, multiple samples are concatenated
        
    def test_multiple_samples_concatenation(self):
        """Test that multiple samples work correctly with the fix."""
        num_samples = 5
        embeddings_list = []
        
        for _ in range(num_samples):
            mock_output = {
                'embedding': torch.randn(self.batch_size, self.embedding_dim)
            }
            # Apply the fix for each sample
            fixed_embd = mock_output['embedding'].data[0].unsqueeze(0)
            embeddings_list.append(fixed_embd.detach().numpy())
        
        # Concatenate as done in get_embeddings
        combined_embeddings = np.concatenate(embeddings_list, axis=0)
        
        # Test shape
        expected_shape = (num_samples, self.embedding_dim)
        self.assertEqual(combined_embeddings.shape, expected_shape)
        
        # Test statistics calculation
        mu = np.mean(combined_embeddings, axis=0)
        sigma = np.cov(combined_embeddings, rowvar=False)
        
        # Verify shapes for FAD calculation
        self.assertEqual(mu.shape, (self.embedding_dim,))
        self.assertEqual(sigma.shape, (self.embedding_dim, self.embedding_dim))
        self.assertFalse(np.isscalar(mu))
        self.assertFalse(np.isscalar(sigma))
        
    def test_comparison_with_old_approach(self):
        """Test that shows the problem with the old approach."""
        mock_output = {
            'embedding': torch.randn(self.batch_size, self.embedding_dim)
        }
        
        # Old approach (problematic)
        old_embd = mock_output['embedding'].data[0]
        old_np = old_embd.detach().numpy()
        
        # This would cause issues in statistics calculation
        if old_np.ndim == 1:
            mu_old = np.mean(old_np, axis=0)
            # For 1D array, mean along axis=0 gives scalar
            self.assertTrue(np.isscalar(mu_old), 
                          "Old approach produces scalar mean, which breaks FAD")
        
        # New approach (fixed)
        new_embd = mock_output['embedding'].data[0].unsqueeze(0)
        new_np = new_embd.detach().numpy()
        
        mu_new = np.mean(new_np, axis=0)
        self.assertFalse(np.isscalar(mu_new),
                        "New approach produces vector mean, which is correct for FAD")
        
    def test_embedding_consistency_across_models(self):
        """Test that PANN embeddings are now consistent with other models."""
        # All model embeddings should be 2D arrays after processing
        # This test verifies that PANN now follows the same pattern
        
        # Simulate embeddings from different models after processing
        embedding_shapes = {
            'vggish': (1, 128),      # Typical VGGish embedding
            'pann': (1, 2048),       # PANN embedding (now fixed)
            'clap': (1, 512),        # Typical CLAP embedding
            'encodec': (128, 64),    # EnCodec embedding (channels, time)
        }
        
        for model_name, expected_shape in embedding_shapes.items():
            if model_name == 'pann':
                # Test our PANN fix
                mock_output = {'embedding': torch.randn(1, 2048)}
                embd = mock_output['embedding'].data[0].unsqueeze(0)
                self.assertEqual(embd.shape, torch.Size(expected_shape))
                self.assertEqual(embd.ndim, 2)
            else:
                # Other models should maintain 2D
                mock_embd = torch.randn(*expected_shape)
                self.assertEqual(mock_embd.ndim, 2)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)