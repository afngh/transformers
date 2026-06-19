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

text = open('./data/shakespeare.txt').read(1000).lower().replace('.',' <EOS> <BOS> ')

data = text.split()
spcl = ['<BOS>','<EOS>','<PAD>','<UNK>']
words = spcl + sorted(list(set(data)))

wti = {word:i for i,word in enumerate(words)}
itw = {i:word for i,word in enumerate(words)}

class Indexes:
  UNK_IDX = wti['<UNK>']
  PAD_IDX = wti['<PAD>']
  BOS_IDX = wti['<BOS>']
  EOS_IDX = wti['<EOS>']

class Data:
    def __init__(self, wti, itw):
        self.wti = wti
        self.itw = itw


class Locales:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 10
    X = list()
    y = list()


class ModelConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(words)
    dim = int(64)
    head = int(4)

   

transform = EncDec(data=Data(wti, itw), idxs=Indexes)
locale = Locales()
config = ModelConfig()


class orch:
    EmbeddingModel = Embedding(
        vocab_size=config.vocab_size,
        dims=config.dim,
    )

    PositionalEmbeddingModel = PositionalEmbedding(
        vocab_size=config.vocab_size,
        dims=config.dim,
    )

    AttentionModel = Attention(
        dims=config.dim,
        head=config.head,
    )

    PostAttentionModel = PostAttention(
        dims=config.dim,
    )

    TransformerModel = Transformer(
        dims=config.dim,
        vocab_size=config.vocab_size,
    )

    Device = config.device

for i in range(len(data) - locale.seq_len):
  X_data = transform.encode(data[i:i+locale.seq_len])
  y_data = transform.encode([data[i+locale.seq_len]])[0]
  locale.X.append(X_data)
  locale.y.append(y_data)

X = torch.tensor(locale.X).to(locale.device)
y = torch.tensor(locale.y).to(locale.device)

ModelOrchestrator = orch()

model = Model(
    EmbeddingModel=ModelOrchestrator.EmbeddingModel,
    PositionalEmbeddingModel=ModelOrchestrator.PositionalEmbeddingModel,
    Attention=ModelOrchestrator.AttentionModel,
    PostAttention=ModelOrchestrator.PostAttentionModel,
    Transformer=ModelOrchestrator.TransformerModel,
    Device=ModelOrchestrator.Device
)

class DataLoaderConfig:
    batch_size = int(64)
    shuffle = bool(True)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset,batch_size = 32,drop_last=True,shuffle=True)

class BackPropConfig:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

class TrainConfig:
   EPOCHS = int(30)
   NORM = float(1.0)

dl = DataLoaderConfig()
bp = BackPropConfig()
tr = TrainConfig()

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

class ModelSaveConfig:
    model = model.state_dict()
    path = "model/model.pth"

#save model
save_model = SaveModel(ModelSaveConfig())
save_model.save()

class InferenceConfig:
    max_tokens = int(200)
    temperature = float(2.0)
    top_k = int(0)
    top_p = float(0.75)

inference = InferenceConfig()

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