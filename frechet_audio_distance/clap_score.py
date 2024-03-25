"""
Calculate the CLAP score between a text list and an audio directory

CLAP score is an audio adaptation of the CLIP score: https://arxiv.org/abs/2104.08718

While CLIP score is defined in https://arxiv.org/abs/2104.08718 as:
    CLIPscore = mean(max(0, w * cosine_similarity(text_embeddings, audio_embeddings))

the CLAP score is implemented in https://arxiv.org/abs/2301.12661 and https://arxiv.org/abs/2401.04577 as:
    CLAPscore = mean(cosine_similarity(text_embeddings, audio_embeddings))
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

from .utils import load_audio_task


class CLAPScore:
    def __init__(
        self,
        ckpt_dir=None,
        submodel_name="630k-audioset",
        verbose=False,
        audio_load_worker=8,
        enable_fusion=False,
    ):
        """
        Initialize CLAP score

        -- ckpt_dir: folder where the downloaded checkpoints are stored
        -- submodel_name: determines which clap checkpoint to use. 
                          options: ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        -- enable_fusion: whether to use fusion for clap models (valid depending on the specific submodel used)
        """
        assert submodel_name in ["630k-audioset", "630k", "music_audioset", "music_speech", "music_speech_audioset"]
        self.submodel_name = submodel_name
        self.sample_rate = 48000  # CLAP model sample rate
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker
        self.enable_fusion = enable_fusion
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.hub.set_dir(ckpt_dir)
            self.ckpt_dir = ckpt_dir
        else:
            # by default `ckpt_dir` is `torch.hub.get_dir()`
            self.ckpt_dir = torch.hub.get_dir()
        self.__get_model()

    def __get_model(self):
        """
        Get ckpt and set model for the specified model_name
        """
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
                print("[CLAP Score] Downloading {}...".format(model_path))
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
        self.model.eval()

    def __load_text_file(self, text_path, text_column):
        """
        Load text file and return a list of text captions.

        Parameters:
        -- text_path (str): Path to the file containing text captions.
        -- text_column (str): Name of the column containing text captions.

        Returns:
        -- list: A list of text captions.
        """
        try:
            import pandas as pd
            df = pd.read_csv(text_path)
            text_data = df[text_column].tolist()
            return text_data
        except Exception as e:
            print("[CLAP score] __load_text_file threw an exception: {}".format(str(e)))
            return []

    def get_text_embeddings(self, text_data):
        """
        Compute text embeddings for a list of text captions.

        Parameters:
        -- text_data (list): A list of text captions.

        Returns:
        -- np.array: An array of text embeddings.
        """
        try:
            text_embds = self.model.get_text_embedding(text_data)
            return text_embds
        except Exception as e:
            print("[CLAP score] get_text_embeddings threw an exception: {}".format(str(e)))
            return np.array([])

    def get_audio_embeddings(self, x, sr):
        """
        Get audio embeddings.

        Params:
        -- x    : a list of np.ndarray audio samples
        -- sr   : sampling rate.
        """
        embd_lst = []
        try:
            for audio in tqdm(x, disable=(not self.verbose)):
                audio = torch.tensor(audio).float().unsqueeze(0)
                embd = self.model.get_audio_embedding_from_data(audio, use_tensor=True)

                if self.device == torch.device('cuda'):
                    embd = embd.cpu()

                if torch.is_tensor(embd):
                    embd = embd.detach().numpy()

                embd_lst.append(embd)
        except Exception as e:
            print("[CLAP score] get_audio_embeddings threw an exception: {}".format(str(e)))

        return np.concatenate(embd_lst, axis=0)

    def __load_audio_files(self, dir, dtype="float32"):
        task_results = []

        pool = ThreadPool(self.audio_load_worker)
        pbar = tqdm(total=len(os.listdir(dir)), disable=(not self.verbose))

        def update(*a):
            pbar.update()

        if self.verbose:
            print("[CLAP score] Loading audio from {}...".format(dir))
        for fname in os.listdir(dir):
            # assume CLAP input audio to always be mono channel
            res = pool.apply_async(
                load_audio_task,
                args=(os.path.join(dir, fname), self.sample_rate, 1, dtype),
                callback=update
            )
            task_results.append(res)
        pool.close()
        pool.join()

        return [k.get() for k in task_results]
    
    # def __cosine_similarity(self, x, y):
    #     """
    #     Compute cosine similarity between two matrices.

    #     Parameters:
    #     -- x (np.array): A matrix.
    #     -- y (np.array): A matrix.

    #     Returns:
    #     -- np.array: A vector of cosine similarities.
    #     """
    #     # Check x and y are 2-dimensional arrays and have the same shape
    #     assert len(x.shape) == 2 and len(y.shape) == 2
    #     assert x.shape == y.shape
        
    #     # Compute dot product of x and y
    #     dot_product = np.sum((x * y), axis=-1)
    #     # Compute L2 norm of x and y
    #     x_norm = np.linalg.norm(x, axis=-1) + 1e-8 # avoid division by zero
    #     y_norm = np.linalg.norm(y, axis=-1) + 1e-8
    #     # Compute cosine similarity
    #     sim_scores = dot_product / (x_norm * y_norm)
    #     return sim_scores
    
    def __cosine_similarity(self, x, y):
        """
        Compute cosine similarity between two matrices.

        As implemented in: https://github.com/microsoft/CLAP/blob/main/msclap/CLAPWrapper.py#L329

        Parameters:
        -- x (np.array): A matrix.
        -- y (np.array): A matrix.

        Returns:
        -- np.array: A vector of cosine similarities.
        """
        # Check x and y are 2-dimensional arrays and have the same shape
        assert len(x.shape) == 2 and len(y.shape) == 2
        assert x.shape == y.shape

        x = x/(np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8) # avoid division by zero
        y = y/(np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
        similarity_matrix = y @ x.T
        sim_scores = similarity_matrix.T.diagonal()
        return sim_scores
    
    def calculate_clap_score(self, text_embds, audio_embds, batch_size):
        """
        Calculate the CLAP score between text and audio embeddings.

        As implemented in: https://github.com/Text-to-Audio/Make-An-Audio/blob/main/wav_evaluation/cal_clap_score.py#L50

        Parameters:
        -- text_embds (np.array): An array of text embeddings.
        -- audio_embds (np.array): An array of audio embeddings.
        -- batch_size (int): Batch size for computing CLAP score.

        Returns:
        -- float: The mean CLAP score.
        -- float: The standard deviation of the CLAP score.
        """
        try:
            # Calculate CLAP score
            sim_scores_all = None
            for i in range(0, len(text_embds), batch_size):
                text_embds_batch = text_embds[i:i+batch_size]
                audio_embds_batch = audio_embds[i:i+batch_size]
                sim_scores = self.__cosine_similarity(text_embds_batch, audio_embds_batch)
                if sim_scores_all is None:
                    sim_scores_all = sim_scores
                else:
                    sim_scores_all = np.concatenate((sim_scores_all, sim_scores))
            clap_score_mean = np.mean(sim_scores_all.flatten())
            clap_score_std = np.std(sim_scores_all.flatten())
            return clap_score_mean, clap_score_std
        except Exception as e:
            print("[CLAP score] calculate_clap_score threw an exception: {}".format(str(e)))
            return -1, -1

    def score(self,
              text_path,
              audio_dir,
              text_column="caption",
              text_embds_path=None,
              audio_embds_path=None,
              batch_size=10,
              dtype="float32",
              ):
        """
        Computes the CLAP score between a text and audio embeddings.

        Parameters:
        -- text_path (str): Path to the file containing text captions.
        -- audio_dir (str): Path to the directory containing audio files.
        -- text_column (str, optional): Name of the column containing text captions. Default is "caption".
        -- text_embds_path (str, optional): Path to save/load text embeddings (e.g., /folder/txt_embds.npy). If None, embeddings won't be saved.
        -- audio_embds_path (str, optional): Path to save/load audio embeddings (e.g., /folder/test_embds.npy). If None, embeddings won't be saved.
        -- dtype (str, optional): Data type for loading audio. Default is "float32".

        Returns:
        -- float: The CLAP score between text and audio embeddings.
        """
        try:
            # Load or compute text embeddings
            if text_embds_path is not None and os.path.exists(text_embds_path):
                if self.verbose:
                    print(f"[CLAP score] Loading text embeddings from {text_embds_path}...")
                text_embds = np.load(text_embds_path)
            else:
                text_data = self.__load_text_file(text_path, text_column)
                text_embds = self.get_text_embeddings(text_data)
                if text_embds_path:
                    os.makedirs(os.path.dirname(text_embds_path), exist_ok=True)
                    np.save(text_embds_path, text_embds)
            if self.verbose:
                print(f"[CLAP score] Loaded {len(text_embds)} text embeddings of shape {text_embds.shape}")

            # Load or compute audio embeddings
            if audio_embds_path is not None and os.path.exists(audio_embds_path):
                if self.verbose:
                    print(f"[CLAP score] Loading audio embeddings from {audio_embds_path}...")
                audio_embds = np.load(audio_embds_path)
            else:
                audio_data = self.__load_audio_files(audio_dir, dtype=dtype)
                audio_embds = self.get_audio_embeddings(audio_data, sr=self.sample_rate)
                if audio_embds_path:
                    os.makedirs(os.path.dirname(audio_embds_path), exist_ok=True)
                    np.save(audio_embds_path, audio_embds)
            if self.verbose:
                print(f"[CLAP score] Loaded {len(audio_embds)} audio embeddings of shape {audio_embds.shape}")

            # Check if embeddings are empty
            if len(text_embds) == 0:
                print("[CLAP score] text embeddings is empty, exiting...")
                return -1
            if len(audio_embds) == 0:
                print("[CLAP score] audio embeddings is empty, exiting...")
                return -1

            # Check if embeddings have the same dimension
            if text_embds.shape != audio_embds.shape:
                print("[CLAP score] text and audio embeddings have different dimensions, exiting...")
                return -1

            # Compute CLAP score
            clap_score_mean, clap_score_std = self.calculate_clap_score(text_embds, audio_embds, batch_size)
            return clap_score_mean, clap_score_std
        except Exception as e:
            print(f"[CLAP score] An error occurred: {e}")
            return -1
