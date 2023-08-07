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
    model_name="vggish",
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# to use `PANN`
frechet = FrechetAudioDistance(
    model_name="pann",
    use_pca=False, 
    use_activation=False,
    verbose=False
)
fad_score = frechet.score("/path/to/background/set", "/path/to/eval/set", dtype="float32")

```

When computing the Frechet Audio Distance, you can choose to save the embeddings for future use. This capability not only ensures consistency across evaluations but can also significantly reduce computation time, especially if you're evaluating multiple times using the same dataset.

```python
# Specify the paths to your saved embeddings
background_embds_path = "/path/to/saved/background/embeddings.npy"
eval_embds_path = "/path/to/saved/eval/embeddings.npy"

# Compute FAD score while reusing the saved embeddings (or saving new ones if paths are provided and embeddings don't exist yet)
fad_score = frechet.score(
    "/path/to/background/set",
    "/path/to/eval/set",
    background_embds_path=background_embds_path,
    eval_embds_path=eval_embds_path,
    dtype="float32"
)
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

### To contribute

- Run `python3 -m build` to build your version locally. The built wheel should be in `dist/`.
- `pip install` your local wheel version, and run `pytest test/` to validate your changes.

### References

VGGish in PyTorch: https://github.com/harritaylor/torchvggish

Frechet distance implementation: https://github.com/mseitzer/pytorch-fid

Frechet Audio Distance paper: https://arxiv.org/abs/1812.08466

PANN paper: https://arxiv.org/abs/1912.10211
