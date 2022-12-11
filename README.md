## Frechet Audio Distance in PyTorch

A lightweight library of Frechet Audio Distance calculation.

### Installation

`pip install frechet_audio_distance`

### Demo

```python
from frechet_audio_distance import FrechetAudioDistance

frechet = FrechetAudioDistance(
    use_pca=False, 
    use_activation=False,
    verbose=False
)
fad_score = frechet.score("/path/to/background/set", "/path/to/eval/set")

```

### Result validation

We hereby provide the FAD scores comparison w.r.t. to the original implementation in `google-research/frechet-audio-distance`

**Test 1: Distorted sine waves** (as provided [here](https://github.com/google-research/google-research/blob/master/frechet_audio_distance/gen_test_files.py#L86)) [[notes]()]



### References

VGGish in PyTorch: https://github.com/harritaylor/torchvggish

Frechet distance implementation: https://github.com/mseitzer/pytorch-fid

Frechet Audio Distance paper: https://arxiv.org/abs/1812.08466
