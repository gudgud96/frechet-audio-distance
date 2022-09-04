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
### References

VGGish in PyTorch: https://github.com/harritaylor/torchvggish

Frechet distance implementation: https://github.com/mseitzer/pytorch-fid

Frechet Audio Distance paper: https://arxiv.org/abs/1812.08466
