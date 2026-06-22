import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import math
import torch.optim as optim
import pickle

from torch._prims_common import Tensor
from torch.utils.data import TensorDataset, DataLoader
from rich.progress import track

from main.transformer_orch._model_orc import Model
from main.generator_config._generator_api import Generator

from main.config._model_config import ModelConfig
from main.config._model_config import ModelSave
from main.config._model_config import ConfigSave
from main.config._model_config import ConfigPath
from main.config._model_config import InferenceConfig
from main.config._model_config import DataLoaderConfig
from main.config._model_config import BackPropConfig
from main.config._model_config import TrainConfig
from main.config._model_config import ModelOrchestrator
from main.config._model_config import Indexes
from main.config._model_config import Data
from main.config._model_config import Locales

class PretrainedHandler:
    def __init__(self, model_path, config_path):
        self.model_path = model_path
        self.config_path = config_path
    
    def load(self):
        try:
            model = torch.load(self.model_path, weights_only=True)
            print(f"Model loaded successfully from {self.model_path}")

            with open(self.config_path, 'rb') as f:
                config = pickle.load(f)
            print(f"Config loaded successfully from {self.config_path}")
            return model, config
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def get_model(self, model, config):
        iconfig = config.config
        iModelOrchestrator = ModelOrchestrator(config=iconfig)

        imodel = Model(
            EmbeddingModel=iModelOrchestrator.EmbeddingModel,
            PositionalEmbeddingModel=iModelOrchestrator.PositionalEmbeddingModel,
            TransformerBlockLayers=iModelOrchestrator.TransformerBlockLayers,
            Transformer=iModelOrchestrator.TransformerModel,
            Device=iModelOrchestrator.Device
        )
        imodel.load_state_dict(model["model"])
        return imodel
    
    def get_config(self, config):
        imax_tokens=config.inference.max_tokens,
        itemperature=config.inference.temperature,
        itop_k=config.inference.top_k,
        itop_p=config.inference.top_p,
        itransform=config.transform,
        iconfig=config.config,
        iseq_len=config.locale.seq_len,
        idevice=config.locale.device,
        # print(config.transform)

        return imax_tokens, itemperature, itop_k, itop_p, itransform, iconfig, iseq_len, idevice

    def client(self, model, config, require_params=False, temperature=None, top_k=None, top_p=None, max_tokens=None):
        if model is None or config is None:
            print("Model or config is not loaded properly.")
            return None
        
        model_origin  = self.get_model(model, config)
        config_origin = self.get_config(config)

        imax_tokens, itemperature, itop_k, itop_p, itransform, iconfig, iseq_len, idevice = config_origin

        if require_params:
            itemperature = tuple([temperature]) if temperature is not None else itemperature
            itop_k = tuple([top_k]) if top_k is not None else itop_k
            itop_p = tuple([top_p]) if top_p is not None else itop_p
            imax_tokens = tuple([max_tokens]) if max_tokens is not None else imax_tokens

        generator = Generator(
            model=model_origin,
            max_tokens=imax_tokens[0],
            temperature=itemperature[0],
            top_k=itop_k[0],
            top_p=itop_p[0],
            transform=itransform[0],
            config=iconfig[0],
            seq_len=iseq_len[0],
            device=idevice[0],
        )

        return generator