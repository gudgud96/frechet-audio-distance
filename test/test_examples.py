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