import numpy as np


SAMPLE_RATE = 16000


def add_noise(data, stddev):
  """Adds Gaussian noise to the samples.
  Args:
    data: 1d Numpy array containing floating point samples. Not necessarily
      normalized.
    stddev: The standard deviation of the added noise.
  Returns:
     1d Numpy array containing the provided floating point samples with added
     Gaussian noise.
  Raises:
    ValueError: When data is not a 1d numpy array.
  """
  if len(data.shape) != 1:
    raise ValueError("expected 1d numpy array.")
  max_value = np.amax(np.abs(data))
  num_samples = data.shape[0]
  gauss = np.random.normal(0, stddev, (num_samples)) * max_value
  return data + gauss


def gen_sine_wave(freq=600,
                  length_seconds=6,
                  sample_rate=SAMPLE_RATE,
                  param=None):
  """Creates sine wave of the specified frequency, sample_rate and length."""
  t = np.linspace(0, length_seconds, int(length_seconds * sample_rate))
  samples = np.sin(2 * np.pi * t * freq)
  if param:
    samples = add_noise(samples, param)
  return np.asarray(2**15 * samples, dtype=np.int16)