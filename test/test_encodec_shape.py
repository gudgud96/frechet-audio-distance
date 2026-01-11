"""
Unit tests for Encodec embedding shape fix.
These tests use mocks to verify the transpose logic without requiring model downloads.
"""
import sys
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
import pytest

# Mock laion_clap before importing frechet_audio_distance to avoid download attempts
sys.modules['laion_clap'] = Mock()


def test_encodec_embedding_shape_transpose():
    """
    Test that Encodec embeddings are correctly transposed.
    
    This verifies that:
    1. Encodec embeddings are transposed from (d, n) to (n, d)
    2. The final shape is correct for multivariate Gaussian modeling
    3. Other models are not affected
    """
    # Import here to avoid issues with module-level imports
    from frechet_audio_distance import FrechetAudioDistance
    
    # Mock the EncodecModel to avoid downloads
    with patch('frechet_audio_distance.fad.EncodecModel') as mock_encodec_class:
        # Create mock model
        mock_model = Mock()
        mock_model.sample_rate = 24000
        
        # Encoder returns (1, d=128, n=75) - typical Encodec output for 1 second audio
        d, n = 128, 75
        mock_encoder_output = torch.randn(1, d, n)
        mock_encoder = Mock(return_value=mock_encoder_output)
        mock_model.encoder = mock_encoder
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.set_target_bandwidth = Mock()
        
        mock_encodec_class.encodec_model_24khz = Mock(return_value=mock_model)
        
        # Create FAD instance
        fad = FrechetAudioDistance(
            model_name="encodec",
            sample_rate=24000,
            channels=1,
            verbose=False
        )
        
        # Create dummy audio data (3 samples)
        num_samples = 3
        audio_samples = [np.random.randn(24000).astype(np.float32) for _ in range(num_samples)]
        
        # Get embeddings
        embeddings = fad.get_embeddings(audio_samples, sr=24000)
        
        # After the fix, embeddings should be (num_samples * n, d)
        expected_shape = (num_samples * n, d)
        assert embeddings.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {embeddings.shape}"
        
        # Verify statistics are computed over latent dimension
        mu, sigma = fad.calculate_embd_statistics(embeddings)
        
        assert mu.shape == (d,), f"Expected mean shape ({d},), got {mu.shape}"
        assert sigma.shape == (d, d), f"Expected covariance shape ({d}, {d}), got {sigma.shape}"


def test_non_encodec_models_not_transposed():
    """
    Test that non-Encodec models are not affected by the transpose.
    """
    from frechet_audio_distance import FrechetAudioDistance
    
    # Test PANN (mock)
    with patch('frechet_audio_distance.fad.Cnn14_16k') as mock_pann_class, \
         patch('frechet_audio_distance.fad.torch.hub.download_url_to_file'), \
         patch('frechet_audio_distance.fad.torch.load') as mock_load, \
         patch('frechet_audio_distance.fad.os.path.exists', return_value=True):
        
        mock_pann = Mock()
        mock_pann.to = Mock(return_value=mock_pann)
        mock_pann.eval = Mock()
        mock_pann.load_state_dict = Mock()
        
        # PANN returns a dict with 'embedding' that has .data[0] giving a 1D tensor
        pann_embd_dim = 2048
        
        # Create a mock data structure that behaves like tensor.data[0]
        def mock_pann_forward(audio, dummy):
            embd_tensor = torch.randn(1, pann_embd_dim)
            mock_data = Mock()
            mock_data.__getitem__ = Mock(return_value=embd_tensor[0])
            return {'embedding': Mock(data=mock_data)}
        
        mock_pann.side_effect = mock_pann_forward
        
        mock_pann_class.return_value = mock_pann
        mock_load.return_value = {'model': {}}
        
        fad_pann = FrechetAudioDistance(
            model_name="pann",
            sample_rate=16000,
            verbose=False
        )
        
        # Create dummy audio (3 samples)
        num_samples = 3
        audio_samples = [np.random.randn(16000).astype(np.float32) for _ in range(num_samples)]
        embeddings_pann = fad_pann.get_embeddings(audio_samples, sr=16000)
        
        # PANN returns 1D embeddings, concatenated along axis=0
        # So 3 samples of (2048,) become (6144,) = (3 * 2048,)
        expected_shape = (num_samples * pann_embd_dim,)
        assert embeddings_pann.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {embeddings_pann.shape}"
        
        # Verify no transpose was applied (shape is 1D, not 2D)
        assert embeddings_pann.ndim == 1, \
            f"PANN embeddings should be 1D, got {embeddings_pann.ndim}D"


def test_encodec_statistics_over_latent_dimension():
    """
    Test that statistics are computed over the latent dimension for Encodec.
    
    This ensures the fix correctly models multivariate-Gaussians over latent space
    rather than time.
    """
    from frechet_audio_distance import FrechetAudioDistance
    
    with patch('frechet_audio_distance.fad.EncodecModel') as mock_encodec_class:
        # Setup mock
        d, n = 128, 75
        mock_model = Mock()
        mock_model.sample_rate = 24000
        mock_encoder_output = torch.randn(1, d, n)
        mock_model.encoder = Mock(return_value=mock_encoder_output)
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        mock_model.set_target_bandwidth = Mock()
        
        mock_encodec_class.encodec_model_24khz = Mock(return_value=mock_model)
        
        fad = FrechetAudioDistance(
            model_name="encodec",
            sample_rate=24000,
            channels=1,
            verbose=False
        )
        
        # Get embeddings for 2 samples
        audio_samples = [np.random.randn(24000).astype(np.float32) for _ in range(2)]
        embeddings = fad.get_embeddings(audio_samples, sr=24000)
        
        # Verify we have more rows (time steps) than columns (latent dims)
        assert embeddings.shape[0] > embeddings.shape[1], \
            f"Expected more time steps than latent dims, got shape {embeddings.shape}"
        
        # The second dimension should be the latent dimension
        assert embeddings.shape[1] == d, \
            f"Expected latent dimension {d}, got {embeddings.shape[1]}"
        
        # Calculate stats and verify covariance is over latent dimension
        mu, sigma = fad.calculate_embd_statistics(embeddings)
        
        # Covariance should be (d x d), not (n x n)
        assert sigma.shape == (d, d), \
            f"Covariance should model Gaussian over {d} latent dims, got shape {sigma.shape}"


if __name__ == "__main__":
    # Allow running directly for quick testing
    test_encodec_embedding_shape_transpose()
    print("✓ test_encodec_embedding_shape_transpose passed")
    
    test_non_encodec_models_not_transposed()
    print("✓ test_non_encodec_models_not_transposed passed")
    
    test_encodec_statistics_over_latent_dimension()
    print("✓ test_encodec_statistics_over_latent_dimension passed")
    
    print("\nAll tests passed!")
