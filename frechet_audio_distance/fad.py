"""
Calculate Frechet Audio Distance betweeen two audio directories.

Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid

VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy
from multiprocessing.dummy import Pool as ThreadPool
from .models.pann import Cnn14_8k, Cnn14_16k, Cnn14


SAMPLE_RATE = 16000


def load_audio_task(fname, dtype="float32"):
    if dtype not in ['float64', 'float32', 'int32', 'int16']:
        raise ValueError(f"dtype not supported: {dtype}")

    wav_data, sr = sf.read(fname, dtype=dtype)
    # For integer type PCM input, convert to [-1.0, +1.0]
    if dtype == 'int16':
        wav_data = wav_data / 32768.0
    elif dtype == 'int32':
        wav_data = wav_data / float(2**31)
    
    # Convert to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    if sr != SAMPLE_RATE:
        wav_data = resampy.resample(wav_data, sr, SAMPLE_RATE)

    return wav_data


class FrechetAudioDistance:
    def __init__(
        self, 
        ckpt_dir=None,
        model_name="vggish", 
        sample_rate=16000,
        use_pca=False, 
        use_activation=False, 
        verbose=False, 
        audio_load_worker=8
    ):  
        assert model_name in ["vggish", "pann"], "model_name must be either 'vggish' or 'pann'"
        if model_name == "vggish":
            assert sample_rate == 16000, "sample_rate must be 16000"
        elif model_name == "pann":
            assert sample_rate in [8000, 16000, 32000], "sample_rate must be 8000, 16000 or 32000"
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
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
        Params:
        -- x   : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        if model_name == "vggish":
            # S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
            self.model = torch.hub.load(repo_or_dir='harritaylor/torchvggish', model='vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        
        elif model_name == "pann":
            # Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020)
            if self.sample_rate == 8000:
                model_path = os.path.join(self.ckpt_dir, "Cnn14_8k_mAP%3D0.416.pth")
                if not(os.path.exists(model_path)):
                    if self.verbose:
                        print("[Frechet Audio Distance] Downloading {}...".format(model_path))
                    torch.hub.download_url_to_file(
                        url='https://zenodo.org/record/3987831/files/Cnn14_8k_mAP%3D0.416.pth', 
                        dst=model_path
                    )
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
                model_path = os.path.join(self.ckpt_dir, "Cnn14_16k_mAP%3D0.438.pth")
                if not(os.path.exists(model_path)):
                    if self.verbose:
                        print("[Frechet Audio Distance] Downloading {}...".format(model_path))
                    torch.hub.download_url_to_file(
                        url='https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth', 
                        dst=model_path
                    )
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
                model_path = os.path.join(self.ckpt_dir, "Cnn14_mAP%3D0.431.pth")
                if not(os.path.exists(model_path)):
                    if self.verbose:
                        print("[Frechet Audio Distance] Downloading {}...".format(model_path))
                    torch.hub.download_url_to_file(
                        url='https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth', 
                        dst=model_path
                    )
                self.model = Cnn14(
                    sample_rate=32000, 
                    window_size=1024, 
                    hop_size=320, 
                    mel_bins=64, 
                    fmin=50, 
                    fmax=16000, 
                    classes_num=527
                )
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        self.model.eval()
    
    def get_embeddings(self, x, sr=SAMPLE_RATE):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):
                if self.model_name == "vggish":
                    embd = self.model.forward(audio, sr)
                elif self.model_name == "pann":
                    with torch.no_grad():
                        out = self.model(torch.tensor(audio).float().unsqueeze(0), None)
                        embd = out['embedding'].data[0]
                if self.device == torch.device('cuda'):
                    embd = embd.cpu()
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
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

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
                args=(os.path.join(dir, fname), dtype,), 
                callback=update
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
        - background_embds_path (str, optional): Path to save/load background audio embeddings. If None, embeddings won't be saved.
        - eval_embds_path (str, optional): Path to save/load evaluation audio embeddings. If None, embeddings won't be saved.
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
                embds_background = self.get_embeddings(audio_background)
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
                embds_eval = self.get_embeddings(audio_eval)
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