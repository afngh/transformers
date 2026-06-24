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
import os
from dotenv import load_dotenv

load_dotenv()
REPO_ID = os.getenv("HF_REPO_ID", "afnhf/dynamo")

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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
    
    def encode_file(self, file_path, chunk_size=500_000):
        """yields chunks of token IDs, with each chunk containing chunk_size tokens"""
        with open(file_path, 'r', encoding='utf-8') as f:
            buffer_ids = []
            char_block_size = chunk_size * 4
            char_block = f.read(char_block_size)
            while char_block:
                ids = self.transform.encode(char_block)
                buffer_ids.extend(ids)
                while len(buffer_ids) >= chunk_size:
                    yield buffer_ids[:chunk_size]
                    buffer_ids = buffer_ids[chunk_size:]
                char_block = f.read(char_block_size)
            if len(buffer_ids) > 0:
                buffer_ids.append(self.transform.EOS_IDX)
                yield buffer_ids

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

    def train(self, file_path):
        model = self._load_pretrained_model(self.model_weights)
        scaler = torch.amp.GradScaler('cuda') if self.locale.device.type == 'cuda' else None

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        for chunk_idx, ids in enumerate(self.encode_file(file_path)):
            locale = Locales()
            for i in range(0, len(ids) - locale.seq_len, locale.seq_len):
                locale.X.append(ids[i : i + locale.seq_len])
                locale.y.append(ids[i+1 : i + locale.seq_len + 1])

            if len(locale.X) == 0:
                print(f"Chunk {chunk_idx+1} has no data, skipping.")
                continue

            X = torch.tensor(locale.X).to(locale.device)
            y = torch.tensor(locale.y).to(locale.device)

            # Perform a 90/10 split
            val_size = max(1, int(len(locale.X) * 0.1))
            train_size = len(locale.X) - val_size

            if train_size > 0:
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
            else:
                X_train, X_val = X, X
                y_train, y_val = y, y

            LoadUniConfigs = self._load_uni_configs(X=X_train, y=y_train, model=model)
            dl = LoadUniConfigs["DataLoaderConfig"]
            bp = LoadUniConfigs["BackPropConfig"]
            tr = LoadUniConfigs["TrainConfig"]
            
            print(f"Chunk {chunk_idx+1} — {len(ids):,} tokens")

            for epoch in track(range(tr.EPOCHS), description=f"Chunk {chunk_idx+1}:"):
                model.train()
                for X_data, y_true in dl.dataloader:
                    bp.optimizer.zero_grad()
                    if scaler is not None:
                        with torch.amp.autocast(device_type='cuda'):
                            y_pred = model(X_data)
                            loss = bp.loss_fn(
                                y_pred.reshape(-1, y_pred.size(-1)),
                                y_true.reshape(-1)
                            )
                        scaler.scale(loss).backward()
                        scaler.unscale_(bp.optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), tr.NORM)
                        scaler.step(bp.optimizer)
                        scaler.update()
                    else:
                        y_pred = model(X_data)
                        loss = bp.loss_fn(
                            y_pred.reshape(-1, y_pred.size(-1)),
                            y_true.reshape(-1)
                        )
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), tr.NORM)
                        bp.optimizer.step()
                bp.scheduler.step()
            
            # Calculate validation loss
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_dl = DataLoaderConfig(X=X_val, y=y_val, batch_size=dl.batch_size, shuffle=False, drop_last=False)
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_dl.dataloader:
                    if scaler is not None:
                        with torch.amp.autocast(device_type='cuda'):
                            y_val_pred = model(X_val_batch)
                            loss = bp.loss_fn(
                                y_val_pred.reshape(-1, y_val_pred.size(-1)),
                                y_val_batch.reshape(-1)
                            )
                    else:
                        y_val_pred = model(X_val_batch)
                        loss = bp.loss_fn(
                            y_val_pred.reshape(-1, y_val_pred.size(-1)),
                            y_val_batch.reshape(-1)
                        )
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            self.latest_val_loss = avg_val_loss
            print(f"({file_path})Chunk {chunk_idx+1}  Validation Loss: {self.latest_val_loss:.4f}")
            model.train()

            if (chunk_idx + 1) % 50 == 0:
                self._save_model_optimizer_scheduler_data(
                    model=model,
                    optimizer=bp.optimizer,
                    scheduler=bp.scheduler
                )

                self.save(message=f"checkpoint {chunk_idx + 1} with file {file_path}")
            # free memory before next chunk
            del X_train, X_val, y_train, y_val
            del X, y, locale
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Capture final states for the save() call in train.py
        self._save_model_optimizer_scheduler_data(
            model=model,
            optimizer=bp.optimizer,
            scheduler=bp.scheduler
        )

    def _save_model_optimizer_scheduler_data(self, model, optimizer, scheduler):
        self.origin_model = model
        self.origin_optimizer = optimizer
        self.origin_scheduler = scheduler

    def save(self, path=None, message="training completed"):
        path = path or self.checkpoint_path
        torch.save({
            "model": self.origin_model.module.state_dict() if hasattr(self.origin_model, 'module') else self.origin_model.state_dict(),
            "optimizer": self.origin_optimizer.state_dict(),
            "scheduler": self.origin_scheduler.state_dict(),
        }, path)
        print(f"model saved at {path}")

        # Check if validation loss is available to add to the commit message
        if hasattr(self, 'latest_val_loss') and self.latest_val_loss is not None:
            if "validation loss" not in message:
                message = f"{message} with validation loss {self.latest_val_loss:.4f}"

        upload_file(
            path_or_fileobj=path,
            path_in_repo="model.pt",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=message
        )
        print("Checkpoint pushed to Hugging Face")