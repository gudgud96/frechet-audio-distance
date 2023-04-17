## Frechet Audio Distance in PyTorch

A lightweight library of Frechet Audio Distance calculation.

Currently, we support embedding from:
- `VGGish` by [S. Hershey et al.](https://arxiv.org/abs/1812.08466)
- `PANN` by [Kong et al.](https://arxiv.org/abs/1912.10211).

### Installation

`pip install frechet_audio_distance`

### Demo

```python
from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish"
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# to use `PANN`
frechet = FrechetAudioDistance(
    model_name="pann"
    use_pca=False, 
    use_activation=False,
    verbose=False
)
fad_score = frechet.score("/path/to/background/set", "/path/to/eval/set")

```

### Result validation

**Test 1: Distorted sine waves on vggish** (as provided [here](https://github.com/google-research/google-research/blob/master/frechet_audio_distance/gen_test_files.py#L86)) [[notes](https://jexrj22lgy.larksuite.com/docx/Vat2dr8Aqonim6xmE6nuoBVZsUe)]

FAD scores comparison w.r.t. to original implementation in `google-research/frechet-audio-distance`

|                              |   baseline vs test1   |     baseline vs test2    |
|:----------------------------:|:---------------------:|:------------------------:|
|        `google-research`     |          12.4375      |           4.7680         |
|    `frechet_audio_distance`  |          12.7398      |           4.9815         |

**Test 2: Distorted sine waves on PANN**

|                              |   baseline vs test1   |     baseline vs test2    |
|:----------------------------:|:---------------------:|:------------------------:|
|    `frechet_audio_distance`  |        0.000465       |          0.00008594      |

### References

VGGish in PyTorch: https://github.com/harritaylor/torchvggish

Frechet distance implementation: https://github.com/mseitzer/pytorch-fid

Frechet Audio Distance paper: https://arxiv.org/abs/1812.08466

PANN paper: https://arxiv.org/abs/1912.10211