# Base case sine wave FAD calculation test.
# Adapted from: https://github.com/google-research/google-research/blob/master/frechet_audio_distance/gen_test_files.py#L86

import os
import shutil
import soundfile as sf
from .utils import *
from frechet_audio_distance import FrechetAudioDistance

def test_vggish_sine_wave():
    print("VGGish test")
    for traget, count, param in [("background", 10, None), ("test1", 5, 0.0001),
                                ("test2", 5, 0.00001)]:
        os.makedirs(traget, exist_ok=True)
        frequencies = np.linspace(100, 1000, count).tolist()
        for freq in frequencies:
            samples = gen_sine_wave(freq, param=param)
            filename = os.path.join(traget, "sin_%.0f.wav" % freq)
            print("Creating: %s with %i samples." % (filename, samples.shape[0]))
            sf.write(filename,samples, SAMPLE_RATE, "PCM_24")
    
    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False, 
        use_activation=False,
        verbose=False
    )

    tolerance_threshold = 2.0
    fad_score = frechet.score("background", "test1")
    assert abs(fad_score - 12.4375) < tolerance_threshold
    fad_score = frechet.score("background", "test2")
    assert abs(fad_score - 4.7680) < tolerance_threshold

    shutil.rmtree("background")
    shutil.rmtree("test1")
    shutil.rmtree("test2")


def test_pann_sine_wave():
    print("PANN 16k test")
    # TODO: find a better way to test PANN instead of distorted sines
    for traget, count, param in [("background", 10, None), ("test1", 5, 0.0001),
                                ("test2", 5, 0.00001)]:
        os.makedirs(traget, exist_ok=True)
        frequencies = np.linspace(100, 1000, count).tolist()
        for freq in frequencies:
            samples = gen_sine_wave(freq, param=param)
            filename = os.path.join(traget, "sin_%.0f.wav" % freq)
            print("Creating: %s with %i samples." % (filename, samples.shape[0]))
            sf.write(filename,samples, SAMPLE_RATE, "PCM_24")
    
    frechet = FrechetAudioDistance(
        sample_rate=16000,
        model_name="pann",
        use_pca=False, 
        use_activation=False,
        verbose=False
    )

    tolerance_threshold = 5e-4
    fad_score = frechet.score("background", "test1")
    assert abs(fad_score - 4e-4) < tolerance_threshold
    fad_score = frechet.score("background", "test2")
    assert abs(fad_score - 1e-4) < tolerance_threshold

    # TODO: PANN 32k and 8k needs test. Model download is slow though

    shutil.rmtree("background")
    shutil.rmtree("test1")
    shutil.rmtree("test2")


def test_encodec_embedding_shape():
    """
    Test that Encodec embeddings have the correct shape.
    
    This test ensures that for Encodec models:
    - Embeddings are transposed correctly (n, d) instead of (d, n)
    - Multivariate-Gaussians are modeled over latent dimension, not time
    - The covariance matrix has shape (d, d) where d is the latent dimension
    """
    print("Encodec embedding shape test")
    
    # Create a few dummy audio files
    target_dir = "test_encodec_embeddings"
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate 3 short audio samples
    # Use 24kHz sample rate for Encodec
    sample_rate_encodec = 24000
    num_samples = 3
    duration = 1.0  # 1 second
    
    for i in range(num_samples):
        # Create a simple sine wave
        freq = 440 + i * 100  # Different frequencies
        t = np.linspace(0, duration, int(duration * sample_rate_encodec))
        samples = np.sin(2 * np.pi * t * freq)
        samples = np.asarray(2**15 * samples, dtype=np.int16)
        
        filename = os.path.join(target_dir, f"audio_{i}.wav")
        sf.write(filename, samples, sample_rate_encodec, "PCM_16")
    
    # Initialize FAD with Encodec model
    frechet = FrechetAudioDistance(
        model_name="encodec",
        sample_rate=24000,
        channels=1,
        verbose=False
    )
    
    # Load audio files
    audio_list = frechet._FrechetAudioDistance__load_audio_files(target_dir)
    
    # Get embeddings
    embeddings = frechet.get_embeddings(audio_list, sr=24000)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate statistics
    mu, sigma = frechet.calculate_embd_statistics(embeddings)
    
    print(f"Mean shape: {mu.shape}")
    print(f"Covariance shape: {sigma.shape}")
    
    # For Encodec 24kHz model:
    # - Latent dimension d = 128 (this is the embedding dimension)
    # - Time dimension n varies with audio length
    # 
    # After the fix, embeddings should have shape (num_samples * n, d)
    # where d is smaller than n for short audio clips
    # 
    # The covariance matrix should be (d, d), modeling Gaussian over latent space
    
    # Verify that:
    # 1. The second dimension is the latent dimension (should be 128 for Encodec 24kHz)
    # 2. The covariance matrix is square with dimensions equal to latent dimension
    # 3. More time steps than latent dimensions (embeddings.shape[0] > embeddings.shape[1])
    
    latent_dim = embeddings.shape[1]
    time_steps = embeddings.shape[0]
    
    print(f"Time steps (rows): {time_steps}")
    print(f"Latent dimension (cols): {latent_dim}")
    
    # For Encodec 24kHz, latent dimension should be 128
    assert latent_dim == 128, f"Expected latent dimension 128, got {latent_dim}"
    
    # There should be more time steps than latent dimensions for short clips
    assert time_steps > latent_dim, f"Expected time_steps ({time_steps}) > latent_dim ({latent_dim})"
    
    # Covariance matrix should be (128, 128) for Encodec 24kHz
    assert sigma.shape == (128, 128), f"Expected covariance shape (128, 128), got {sigma.shape}"
    
    # Mean should be 128-dimensional
    assert mu.shape == (128,), f"Expected mean shape (128,), got {mu.shape}"
    
    print("✓ All shape assertions passed!")
    
    # Clean up
    shutil.rmtree(target_dir)