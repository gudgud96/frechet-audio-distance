"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import resampy
import soundfile as sf
import torch
import laion_clap

from multiprocessing.dummy import Pool as ThreadPool
from scipy import linalg
from torch import nn
from tqdm import tqdm

from .models.pann import Cnn14, Cnn14_8k, Cnn14_16k
from .utils import load_audio_task

from encodec import EncodecModel


class FrechetAudioDistance:
    def __init__(
        self,
        ckpt_dir=None,
        model_name="vggish",
        submodel_name="630k-audioset",  # only for CLAP
        sample_rate=16000,
        channels=1,
        use_pca=False,  # only for VGGish
        use_activation=False,  # only for VGGish
        verbose=False,
        audio_load_worker=8,
        enable_fusion=False,  # only for CLAP
    ):
        """
        Initialize FAD

        -- ckpt_dir: folder where the downloaded checkpoints are stored
        -- model_name: one between vggish, pann, clap or encodec
        -- submodel_name: only for clap models - determines which checkpoint to use. 
                          options: ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        -- sample_rate: one between [8000, 16000, 32000, 48000]. depending on the model set the sample rate to use
        -- channels: number of channels in an audio track
        -- use_pca: whether to apply PCA to the vggish embeddings
        -- use_activation: whether to use the output activation in vggish
        -- enable_fusion: whether to use fusion for clap models (valid depending on the specific submodel used)
        """
        assert model_name in ["vggish", "pann", "clap", "encodec"], "model_name must be either 'vggish', 'pann', 'clap' or 'encodec'"
        if model_name == "vggish":
            assert sample_rate == 16000, "sample_rate must be 16000"
        elif model_name == "pann":
            assert sample_rate in [8000, 16000, 32000], "sample_rate must be 8000, 16000 or 32000"
        elif model_name == "clap":
            assert sample_rate == 48000, "sample_rate must be 48000"
            assert submodel_name in ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        elif model_name == "encodec":
            assert sample_rate in [24000, 48000], "sample_rate must be 24000 or 48000"
            if sample_rate == 48000:
                assert channels == 2, "channels must be 2 for 48khz encodec model"
        self.model_name = model_name
        self.submodel_name = submodel_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.verbose = verbose
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        if self.device == torch.device('mps') and self.model_name == "clap":
            if self.verbose:
                print("[Frechet Audio Distance] CLAP does not support MPS device yet, because:")
                print("[Frechet Audio Distance] The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device.")
                print("[Frechet Audio Distance] Using CPU device instead.")
            self.device = torch.device('cpu')
        if self.verbose:
            print("[Frechet Audio Distance] Using device: {}".format(self.device))
        self.audio_load_worker = audio_load_worker
        self.enable_fusion = enable_fusion
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.hub.set_dir(ckpt_dir)
            self.ckpt_dir = ckpt_dir
        else:
            # by default `ckpt_dir` is `torch.hub.get_dir()`
            self.ckpt_dir = torch.hub.get_dir()
        self.__get_model(model_name=model_name, use_pca=use_pca, use_activation=use_activation)

    def __get_model(self, model_name="vggish", use_pca=False, use_activation=False):
        """
        Get ckpt and set model for the specified model_name

        Params:
        -- model_name: one between vggish, pann or clap
        -- use_pca: whether to apply PCA to the vggish embeddings
        -- use_activation: whether to use the output activation in vggish
        """
        # vggish
        if model_name == "vggish":
            # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
            self.model = torch.hub.load(repo_or_dir='harritaylor/torchvggish', model='vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
            self.model.device = self.device
        # pann
        elif model_name == "pann":
            # Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020)

            # choose the right checkpoint and model based on sample_rate
            if self.sample_rate == 8000:
                download_name = "Cnn14_8k_mAP%3D0.416.pth"
                self.model = Cnn14_8k(
                    sample_rate=8000,
                    window_size=256,
                    hop_size=80,
                    mel_bins=64,
                    fmin=50,
                    fmax=4000,
                    classes_num=527
                )
            elif self.sample_rate == 16000:
                download_name = "Cnn14_16k_mAP%3D0.438.pth"
                self.model = Cnn14_16k(
                    sample_rate=16000,
                    window_size=512,
                    hop_size=160,
                    mel_bins=64,
                    fmin=50,
                    fmax=8000,
                    classes_num=527
                )
            elif self.sample_rate == 32000:
                download_name = "Cnn14_mAP%3D0.431.pth"
                self.model = Cnn14(
                    sample_rate=32000,
                    window_size=1024,
                    hop_size=320,
                    mel_bins=64,
                    fmin=50,
                    fmax=16000,
                    classes_num=527
                )

            model_path = os.path.join(self.ckpt_dir, download_name)

            # download checkpoint
            if not (os.path.exists(model_path)):
                if self.verbose:
                    print("[Frechet Audio Distance] Downloading {}...".format(model_path))
                torch.hub.download_url_to_file(
                    url=f"https://zenodo.org/record/3987831/files/{download_name}",
                    dst=model_path
                )

            # load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
        # clap
        elif model_name == "clap":
            # choose the right checkpoint and model
            if self.submodel_name == "630k-audioset":
                if self.enable_fusion:
                    download_name = "630k-audioset-fusion-best.pt"
                else:
                    download_name = "630k-audioset-best.pt"
            elif self.submodel_name == "630k":
                if self.enable_fusion:
                    download_name = "630k-fusion-best.pt"
                else:
                    download_name = "630k-best.pt"
            elif self.submodel_name == "music_audioset":
                download_name = "music_audioset_epoch_15_esc_90.14.pt"
            elif self.submodel_name == "music_speech":
                download_name = "music_speech_epoch_15_esc_89.25.pt"
            elif self.submodel_name == "music_speech_audioset":
                download_name = "music_speech_audioset_epoch_15_esc_89.98.pt"

            model_path = os.path.join(self.ckpt_dir, download_name)

            # download checkpoint
            if not (os.path.exists(model_path)):
                if self.verbose:
                    print("[Frechet Audio Distance] Downloading {}...".format(model_path))
                torch.hub.download_url_to_file(
                    url=f"https://huggingface.co/lukewys/laion_clap/resolve/main/{download_name}",
                    dst=model_path
                )
            # init model and load checkpoint
            if self.submodel_name in ["630k-audioset", "630k"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    device=self.device)
            elif self.submodel_name in ["music_audioset", "music_speech", "music_speech_audioset"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    amodel='HTSAT-base',
                                                    device=self.device)
            self.model.load_ckpt(model_path)

            # init model and load checkpoint
            if self.submodel_name in ["630k-audioset", "630k"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    device=self.device)
            elif self.submodel_name in ["music_audioset", "music_speech", "music_speech_audioset"]:
                self.model = laion_clap.CLAP_Module(enable_fusion=self.enable_fusion,
                                                    amodel='HTSAT-base',
                                                    device=self.device)
            self.model.load_ckpt(model_path)

        # encodec
        elif model_name == "encodec":
            # choose the right model based on sample_rate
            # weights are loaded from the encodec repo: https://github.com/facebookresearch/encodec/
            if self.sample_rate == 24000:
                self.model = EncodecModel.encodec_model_24khz()
            elif self.sample_rate == 48000:
                self.model = EncodecModel.encodec_model_48khz()
            # 24kbps is the max bandwidth supported by both versions
            # these models use 32 residual quantizers
            self.model.set_target_bandwidth(24.0)

        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, x, sr):
        """
        Get embeddings using VGGish, PANN, CLAP or EnCodec models.
        Params:
        -- x    : a list of np.ndarray audio samples
        -- sr   : sampling rate.
        """
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):
                if self.model_name == "vggish":
                    embd = self.model.forward(audio, sr)
                elif self.model_name == "pann":
                    with torch.no_grad():
                        audio = torch.tensor(audio).float().unsqueeze(0).to(self.device)
                        out = self.model(audio, None)
                        embd = out['embedding'].data[0]
                elif self.model_name == "clap":
                    audio = torch.tensor(audio).float().unsqueeze(0)
                    embd = self.model.get_audio_embedding_from_data(audio, use_tensor=True)

                elif self.model_name == "encodec":
                    # add two dimensions
                    audio = torch.tensor(
                        audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    # if SAMPLE_RATE is 48000, we need to make audio stereo
                    if self.model.sample_rate == 48000:
                        if audio.shape[-1] != 2:
                            if self.verbose:
                                print(
                                    "[Frechet Audio Distance] Audio is mono, converting to stereo for 48khz model..."
                                )
                            audio = torch.cat((audio, audio), dim=1)
                        else:
                            # transpose to (batch, channels, samples)
                            audio = audio[:, 0].transpose(1, 2)

                    if self.verbose:
                        print(
                            "[Frechet Audio Distance] Audio shape: {}".format(
                                audio.shape
                            )
                        )

                    with torch.no_grad():
                        # encodec embedding (before quantization)
                        embd = self.model.encoder(audio)
                        embd = embd.squeeze(0)

                if self.verbose:
                    print(
                        "[Frechet Audio Distance] Embedding shape: {}".format(
                            embd.shape
                        )
                    )
                
                if embd.device != torch.device("cpu"):
                    embd = embd.cpu()
                
                if torch.is_tensor(embd):
                    embd = embd.detach().numpy()
                
                embd_lst.append(embd)
        except Exception as e:
            print("[Frechet Audio Distance] get_embeddings throw an exception: {}".format(str(e)))

        return np.concatenate(embd_lst, axis=0)

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def __load_audio_files(self, dir, dtype="float32"):
        task_results = []

        pool = ThreadPool(self.audio_load_worker)
        pbar = tqdm(total=len(os.listdir(dir)), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print("[Frechet Audio Distance] Loading audio from {}...".format(dir))
        for fname in os.listdir(dir):
            res = pool.apply_async(
                load_audio_task,
                args=(os.path.join(dir, fname), self.sample_rate, self.channels, dtype),
                callback=update,
            )
            task_results.append(res)
        pool.close()
        pool.join()

        return [k.get() for k in task_results]

    def score(self,
              background_dir,
              eval_dir,
              background_embds_path=None,
              eval_embds_path=None,
              dtype="float32"
              ):
        """
        Computes the Frechet Audio Distance (FAD) between two directories of audio files.

        Parameters:
        - background_dir (str): Path to the directory containing background audio files.
        - eval_dir (str): Path to the directory containing evaluation audio files.
        - background_embds_path (str, optional): Path to save/load background audio embeddings (e.g., /folder/bkg_embs.npy). If None, embeddings won't be saved.
        - eval_embds_path (str, optional): Path to save/load evaluation audio embeddings (e.g., /folder/test_embs.npy). If None, embeddings won't be saved.
        - dtype (str, optional): Data type for loading audio. Default is "float32".

        Returns:
        - float: The Frechet Audio Distance (FAD) score between the two directories of audio files.
        """
        try:
            # Load or compute background embeddings
            if background_embds_path is not None and os.path.exists(background_embds_path):
                if self.verbose:
                    print(f"[Frechet Audio Distance] Loading embeddings from {background_embds_path}...")
                embds_background = np.load(background_embds_path)
            else:
                audio_background = self.__load_audio_files(background_dir, dtype=dtype)
                embds_background = self.get_embeddings(audio_background, sr=self.sample_rate)
                if background_embds_path:
                    os.makedirs(os.path.dirname(background_embds_path), exist_ok=True)
                    np.save(background_embds_path, embds_background)

            # Load or compute eval embeddings
            if eval_embds_path is not None and os.path.exists(eval_embds_path):
                if self.verbose:
                    print(f"[Frechet Audio Distance] Loading embeddings from {eval_embds_path}...")
                embds_eval = np.load(eval_embds_path)
            else:
                audio_eval = self.__load_audio_files(eval_dir, dtype=dtype)
                embds_eval = self.get_embeddings(audio_eval, sr=self.sample_rate)
                if eval_embds_path:
                    os.makedirs(os.path.dirname(eval_embds_path), exist_ok=True)
                    np.save(eval_embds_path, embds_eval)

            # Check if embeddings are empty
            if len(embds_background) == 0:
                print("[Frechet Audio Distance] background set dir is empty, exiting...")
                return -1
            if len(embds_eval) == 0:
                print("[Frechet Audio Distance] eval set dir is empty, exiting...")
                return -1

            # Compute statistics and FAD score
            mu_background, sigma_background = self.calculate_embd_statistics(embds_background)
            mu_eval, sigma_eval = self.calculate_embd_statistics(embds_eval)

            fad_score = self.calculate_frechet_distance(
                mu_background,
                sigma_background,
                mu_eval,
                sigma_eval
            )

            return fad_score
        except Exception as e:
            print(f"[Frechet Audio Distance] An error occurred: {e}")
            return -1
