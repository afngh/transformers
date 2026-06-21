import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import math
import torch.optim as optim

from torch._prims_common import Tensor
from torch.utils.data import TensorDataset, DataLoader
from rich.progress import track

from .seq2seq._enc_dec import EncDec
from .seq2seq._byte_codec import ByteCodec
from .seq2seq._gpt2_tokenizer import TokenCodec
from .transformer_orch._model_orc import Model
from .generator_config._generator_api import Generator
from .save_trained._model_save import SaveModel

from .config._model_config import ModelConfig
from .config._model_config import ModelSave
from .config._model_config import ConfigSave
from .config._model_config import ConfigPath
from .config._model_config import InferenceConfig
from .config._model_config import DataLoaderConfig
from .config._model_config import BackPropConfig
from .config._model_config import TrainConfig
from .config._model_config import ModelOrchestrator
from .config._model_config import Indexes
from .config._model_config import Data
from .config._model_config import Locales

text = open('./data/shakespeare.txt').read(1000)
print(f"Data length: {len(text)}")

transform = TokenCodec()
ids = list()

ids.extend(transform.encode(text))
ids.append(transform.EOS_IDX)

locale = Locales()
config = ModelConfig(vocab_size=transform.vocab_size)
ModelOrchestrator = ModelOrchestrator(config=config)

for i in range(len(ids) - locale.seq_len):
    locale.X.append(ids[i : i + locale.seq_len])
    locale.y.append(ids[i+1 : i + locale.seq_len + 1])

X = torch.tensor(locale.X).to(locale.device)
y = torch.tensor(locale.y).to(locale.device)

model = Model(
    EmbeddingModel=ModelOrchestrator.EmbeddingModel,
    PositionalEmbeddingModel=ModelOrchestrator.PositionalEmbeddingModel,
    TransformerBlockLayers=ModelOrchestrator.TransformerBlockLayers,
    Transformer=ModelOrchestrator.TransformerModel,
    Device=ModelOrchestrator.Device
)

dl = DataLoaderConfig(X=X, y=y)
bp = BackPropConfig(model=model)
tr = TrainConfig(EPOCHS=30, NORM=1.0)

for epoch in track(range(tr.EPOCHS),description="Training Vocab:"):
  model.train()
#   el = None
  for X_data,y_true in dl.dataloader:
    y_pred = model(X_data)
    loss = bp.loss_fn(y_pred.reshape(-1, y_pred.size(-1)), y_true.reshape(-1))
    bp.optimizer.zero_grad()
    # el = loss.item()
    loss.backward()
    bp.optimizer.step()
#   print(f"Epoch: {epoch} && Loss: {el}")
  bp.scheduler.step()



inference = InferenceConfig(max_tokens=20, temperature=1.5, top_k=5, top_p=0.9)

client = Generator(
    model=model,
    max_tokens=inference.max_tokens,
    temperature=inference.temperature,
    top_k=inference.top_k,
    top_p=inference.top_p,
    transform=transform,
    config=config,
    seq_len=locale.seq_len,
    device=locale.device,
)

print(f"Test generated response: [ {client.generate_response('war of')} ]")

ModelSaveData = ModelSave(
    model=model,
)

ConfigSaveData = ConfigSave(
    inference=inference,
    transform=transform,
    config=config,
    locale=locale,
)

ConfigPathData = ConfigPath(
    model_path="bin/model/model.pt",
    config_path="bin/data/config.pkl",
)

SaveModel = SaveModel(ModelSaveData=ModelSaveData, ConfigSaveData=ConfigSaveData, ConfigPathData=ConfigPathData)
SaveModel.save()

trainable_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")