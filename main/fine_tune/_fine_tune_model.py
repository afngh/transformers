import torch
import torch.nn as nn
import pickle
import json

from rich.progress import track

from main.seq2seq._gpt2_tokenizer import TokenCodec
from main.config._model_config import Locales
from main.config._model_config import ModelConfig
from main.config._model_config import ModelOrchestrator
from main.config._model_config import DataLoaderConfig
from main.config._model_config import BackPropConfig
from main.config._model_config import TrainConfig
from main.transformer_orch._model_orc import Model

class FineTuneModel():
    def __init__(self, checkpoint_path):
        checkpoint = self._load(checkpoint_path=checkpoint_path)

        self.model_weights = checkpoint["model"]
        self.optimizer_weights = checkpoint["optimizer"]
        self.scheduler_weights = checkpoint["scheduler"]

        self.transform = TokenCodec()
        self.locale = Locales()

        self.checkpoint_path = checkpoint_path

    def _load(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            print(f"Model & Optimizer & Scheduler loaded successfully from {checkpoint_path}")

            return checkpoint
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _encode_data(self, text):
        ids = list()
        ids.extend(self.transform.encode(text))
        ids.append(self.transform.EOS_IDX)

        return ids
    
    def encode_file(self, file_path):
        text = open(file_path).read()

        ids = self._encode_data(text)
        return ids

    def _get_model_arch(self, transform):
        config = ModelConfig(vocab_size=transform.vocab_size)
        orch = ModelOrchestrator(config=config)

        return orch

    def _load_datasets(self, ids):
        for i in range(len(ids) - self.locale.seq_len):
            self.locale.X.append(ids[i : i + self.locale.seq_len])
            self.locale.y.append(ids[i+1 : i + self.locale.seq_len + 1])

        X = torch.tensor(self.locale.X).to(self.locale.device)
        y = torch.tensor(self.locale.y).to(self.locale.device)

        return X, y

    def _load_uni_configs(self, X, y, model):
        dl = DataLoaderConfig(X=X, y=y)
        bp = BackPropConfig(model=model)
        tr = TrainConfig()

        bp = self._load_optimizer_scheduler(bpc=bp, optimizer_weights=self.optimizer_weights, scheduler_weights=self.scheduler_weights)

        TRAINING_SETTINGS = {
            "DataLoaderConfig":dl,
            "BackPropConfig":bp,
            "TrainConfig":tr
        }
        
        return TRAINING_SETTINGS

    def _load_pretrained_model(self, model_weights=None):
        ModelOrchestrator = self._get_model_arch(transform=self.transform)

        model = Model(
            EmbeddingModel=ModelOrchestrator.EmbeddingModel,
            PositionalEmbeddingModel=ModelOrchestrator.PositionalEmbeddingModel,
            TransformerBlockLayers=ModelOrchestrator.TransformerBlockLayers,
            Transformer=ModelOrchestrator.TransformerModel,
            Device=ModelOrchestrator.Device
        )

        model.load_state_dict(model_weights)

        return model
    
    def _load_optimizer_scheduler(self, bpc=None, optimizer_weights=None, scheduler_weights=None):
        bpc.optimizer.load_state_dict(optimizer_weights)
        bpc.scheduler.load_state_dict(scheduler_weights)

        return bpc

    def train(self, ids):
        X, y = self._load_datasets(ids)

        model = self._load_pretrained_model(self.model_weights)
        LoadUniConfigs = self._load_uni_configs(X=X, y=y, model=model)

        dl = LoadUniConfigs["DataLoaderConfig"]
        bp = LoadUniConfigs["BackPropConfig"]
        tr = LoadUniConfigs["TrainConfig"]

        for epoch in track(range(tr.EPOCHS),description="Training Vocab:"):
            model.train()
            el = None
            for X_data,y_true in dl.dataloader:
                y_pred = model(X_data)
                loss = bp.loss_fn(y_pred.reshape(-1, y_pred.size(-1)), y_true.reshape(-1))
                bp.optimizer.zero_grad()
                el = loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tr.NORM)
                bp.optimizer.step()
            bp.scheduler.step()
            print(f'epoch loss: {el}')

        print("training sucess & saved")

        self._save_model_optimizer_scheduler_data(
            model=model,
            optimizer=bp.optimizer,
            scheduler=bp.scheduler
        )



    def _save_model_optimizer_scheduler_data(self, model, optimizer, scheduler):
        self.origin_model = model
        self.origin_optimizer = optimizer
        self.origin_scheduler = scheduler

    def save(self, path=None):
        origin_model_weights = self.origin_model.state_dict()
        origin_optimizer_weights = self.origin_optimizer.state_dict()
        origin_scheduler_weights = self.origin_scheduler.state_dict()

        path = path or self.checkpoint_path

        torch.save({
            "model":origin_model_weights,
            "optimizer":origin_optimizer_weights,
            "scheduler":origin_scheduler_weights
        },path)

        print("model saved")