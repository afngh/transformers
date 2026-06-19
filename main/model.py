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
from .transformer_orch._embedding import Embedding
from .transformer_orch._positional_embedding import PositionalEmbedding
from .transformer_orch._post_attention import PostAttention
from .transformer_orch._attention import Attention
from .transformer_orch._transformer import Transformer
from .transformer_orch._model_orc import Model
from .generator_config._generator_api import Generator
from .save_trained._model_save import SaveModel

from .config._model_config import ModelConfig
from .config._model_config import ModelSaveConfig
from .config._model_config import InferenceConfig
from .config._model_config import DataLoaderConfig
from .config._model_config import BackPropConfig
from .config._model_config import TrainConfig
from .config._model_config import ModelOrchestrator
from .config._model_config import Indexes
from .config._model_config import Data
from .config._model_config import Locales

text = open('./data/shakespeare.txt').read(300).lower().replace('.',' <EOS> <BOS> ')
print(f"Data length: {len(text)}")

data = text.split()
spcl = ['<BOS>','<EOS>','<PAD>','<UNK>']
words = spcl + sorted(list(set(data)))

wti = {word:i for i,word in enumerate(words)}
itw = {i:word for i,word in enumerate(words)}


transform = EncDec(data=Data(wti=wti, itw=itw), idxs=Indexes(wti=wti))
locale = Locales()
config = ModelConfig(words=words)
ModelOrchestrator = ModelOrchestrator(config=config)

for i in range(len(data) - locale.seq_len):
  X_data = transform.encode(data[i:i+locale.seq_len])
  y_data = transform.encode([data[i+locale.seq_len]])[0]
  locale.X.append(X_data)
  locale.y.append(y_data)

X = torch.tensor(locale.X).to(locale.device)
y = torch.tensor(locale.y).to(locale.device)


model = Model(
    EmbeddingModel=ModelOrchestrator.EmbeddingModel,
    PositionalEmbeddingModel=ModelOrchestrator.PositionalEmbeddingModel,
    Attention=ModelOrchestrator.AttentionModel,
    PostAttention=ModelOrchestrator.PostAttentionModel,
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
    loss = bp.loss_fn(y_pred.squeeze(1),y_true)
    bp.optimizer.zero_grad()
    # el = loss.item()
    loss.backward()
    bp.optimizer.step()
#   print(f"Epoch: {epoch} && Loss: {el}")
  bp.scheduler.step()


ModelSaveData = ModelSaveConfig(model=model, path="test/model.pth")
SaveModel = SaveModel(ModelSaveConfig=ModelSaveData)
SaveModel.save()

inference = InferenceConfig(max_tokens=20, temperature=0.7, top_k=5, top_p=0.9)

client = Generator(
    model=model,
    max_tokens=inference.max_tokens,
    temperature=inference.temperature,
    top_k=inference.top_k,
    top_p=inference.top_p,
    transform=transform,
    config=config,
    itw=itw,
    seq_len=locale.seq_len,
    device=locale.device,
    EOS_token='<EOS>'
)

print(f"Generated response: {client.generate_response('war')}")