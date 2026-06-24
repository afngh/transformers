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
            self.start_chunk = 0
        else:
            self.model_weights = checkpoint["model"]
            self.optimizer_weights = checkpoint["optimizer"]
            self.scheduler_weights = checkpoint["scheduler"]
            self.start_chunk = checkpoint.get("chunk_idx", 0)

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
    
    def stream_token_chunks(self, pt_path, chunk_size=500_000):
        """
        Streams chunks of TOKENS (not characters) from a pre-tokenized .pt file.
        chunk_size=500_000 now means 500K actual tokens per chunk.
        """
        data = torch.load(pt_path)           # shape: (N,) long tensor
        total = len(data)
        start = 0
        chunk_idx = 0
        while start < total:
            end = min(start + chunk_size, total)
            chunk = data[start:end].tolist()  # list of ints
            yield chunk_idx, chunk
            start = end
            chunk_idx += 1

    def compute_val_loss(self, model, val_pt_path, seq_len, batch_size=64, max_batches=100):
        import os
        if not os.path.exists(val_pt_path):
            print(f"Validation file {val_pt_path} not found — skipping validation loss computation")
            return 0.0

        data = torch.load(val_pt_path)
        device = self.locale.device

        X, y = [], []
        for i in range(0, min(len(data) - seq_len, max_batches * batch_size * seq_len), seq_len):
            X.append(data[i : i + seq_len])
            y.append(data[i+1 : i + seq_len + 1])

        if len(X) == 0:
            return 0.0

        X = torch.stack(X).to(device)
        y = torch.stack(y).to(device)

        loss_fn = nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for b in range(0, len(X), batch_size):
                xb = X[b:b+batch_size]
                yb = y[b:b+batch_size]
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        logits = model(xb)
                        loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                else:
                    logits = model(xb)
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                total_loss += loss.item()
                batches += 1

        model.train()
        return total_loss / max(batches, 1)

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

    def train(self, file_path, start_chunk=None):
        model = self._load_pretrained_model(self.model_weights)
        scaler = torch.amp.GradScaler('cuda') if self.locale.device.type == 'cuda' else None

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        bp = BackPropConfig(model=model)
        bp = self._load_optimizer_scheduler(bpc=bp, optimizer_weights=self.optimizer_weights, scheduler_weights=self.scheduler_weights)
        tr = TrainConfig()

        resume_chunk = start_chunk if start_chunk is not None else self.start_chunk
        chunk_idx = resume_chunk - 1

        for chunk_idx, ids in self.stream_token_chunks(file_path):
            if chunk_idx < resume_chunk:
                continue
            locale = Locales()
            for i in range(0, len(ids) - locale.seq_len, locale.seq_len):
                locale.X.append(ids[i : i + locale.seq_len])
                locale.y.append(ids[i+1 : i + locale.seq_len + 1])

            if len(locale.X) == 0:
                print(f"Chunk {chunk_idx+1} has no data, skipping.")
                continue

            X = torch.tensor(locale.X)
            y = torch.tensor(locale.y)

            dl = DataLoaderConfig(X=X, y=y, shuffle=True)
            
            print(f"Chunk {chunk_idx+1} — {len(ids):,} tokens")

            for epoch in track(range(tr.EPOCHS), description=f"Chunk {chunk_idx+1}:"):
                model.train()
                for X_data, y_true in dl.dataloader:
                    X_data = X_data.to(locale.device, non_blocking=True)
                    y_true = y_true.to(locale.device, non_blocking=True)
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

                # Calculate and log validation loss after each epoch
                val_loss = self.compute_val_loss(model, "bin/data/simple_wiki_val.pt", seq_len=locale.seq_len)
                self.latest_val_loss = val_loss
                print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

            if (chunk_idx + 1) % 5 == 0:
                self._save_model_optimizer_scheduler_data(
                    model=model,
                    optimizer=bp.optimizer,
                    scheduler=bp.scheduler,
                    next_chunk_idx=chunk_idx + 1
                )

                self.save(message=f"checkpoint {chunk_idx + 1} with file {file_path}")
            # free memory before next chunk
            del X, y, locale
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Capture final states for the save() call in train.py
        self._save_model_optimizer_scheduler_data(
            model=model,
            optimizer=bp.optimizer,
            scheduler=bp.scheduler,
            next_chunk_idx=chunk_idx + 1
        )

    def _save_model_optimizer_scheduler_data(self, model, optimizer, scheduler, next_chunk_idx):
        self.origin_model = model
        self.origin_optimizer = optimizer
        self.origin_scheduler = scheduler
        self.next_chunk_idx = next_chunk_idx

    def save(self, path=None, message="training completed"):
        path = path or self.checkpoint_path
        save_dict = {
            "model": self.origin_model.module.state_dict() if hasattr(self.origin_model, 'module') else self.origin_model.state_dict(),
            "optimizer": self.origin_optimizer.state_dict(),
            "scheduler": self.origin_scheduler.state_dict(),
        }
        if hasattr(self, 'next_chunk_idx') and self.next_chunk_idx is not None:
            save_dict["chunk_idx"] = self.next_chunk_idx

        torch.save(save_dict, path)
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