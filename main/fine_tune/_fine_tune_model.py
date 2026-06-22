import torch
import torch.nn as nn
import pickle
import json

from rich.progress import track
from huggingface_hub import hf_hub_download, upload_file

from main.seq2seq._gpt2_tokenizer import TokenCodec
from main.config._model_config import Locales
from main.config._model_config import ModelConfig
from main.config._model_config import ModelOrchestrator
from main.config._model_config import DataLoaderConfig
from main.config._model_config import BackPropConfig
from main.config._model_config import TrainConfig
from main.transformer_orch._model_orc import Model

REPO_ID = "afnhf/my-transformer"

class FineTuneModel():
    def __init__(self, checkpoint_path):
        checkpoint = self._load(checkpoint_path=checkpoint_path)

        if checkpoint is None:
            # first run — no checkpoint exists yet, start fresh
            print("No checkpoint found — initializing fresh model")
            self.model_weights = None
            self.optimizer_weights = None
            self.scheduler_weights = None
        else:
            self.model_weights = checkpoint["model"]
            self.optimizer_weights = checkpoint["optimizer"]
            self.scheduler_weights = checkpoint["scheduler"]

        self.transform = TokenCodec()
        self.locale = Locales()
        self.checkpoint_path = checkpoint_path

    def _load(self, checkpoint_path):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename="model.pt",
                local_dir="bin/model"
            )
            print("Checkpoint synced from Hugging Face")
        except Exception:
            print("No remote checkpoint found — using local")

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            print(f"Model & Optimizer & Scheduler loaded from {checkpoint_path}")
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
        text = open(file_path).read(1000)

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
        orch = self._get_model_arch(transform=self.transform)

        model = Model(
            EmbeddingModel=orch.EmbeddingModel,
            PositionalEmbeddingModel=orch.PositionalEmbeddingModel,
            TransformerBlockLayers=orch.TransformerBlockLayers,
            Transformer=orch.TransformerModel,
            Device=orch.Device
        )

        if model_weights is not None:
            model.load_state_dict(model_weights)
            print("Weights loaded from checkpoint")
        else:
            print("Fresh model — random initialization")

        return model
    
    def _load_optimizer_scheduler(self, bpc=None, optimizer_weights=None, scheduler_weights=None):
        if optimizer_weights is not None:
            bpc.optimizer.load_state_dict(optimizer_weights)
        if scheduler_weights is not None:
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
        path = path or self.checkpoint_path
        torch.save({
            "model": self.origin_model.state_dict(),
            "optimizer": self.origin_optimizer.state_dict(),
            "scheduler": self.origin_scheduler.state_dict(),
        }, path)
        print(f"model saved at {path}")

        upload_file(
            path_or_fileobj=path,
            path_in_repo="model.pt",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Checkpoint pushed to Hugging Face")