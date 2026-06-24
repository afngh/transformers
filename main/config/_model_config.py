import torch
import torch.nn as nn
import torch.optim as optim
from torch._prims_common import Tensor
from torch.utils.data import TensorDataset, DataLoader

from main.seq2seq._enc_dec import EncDec
from main.transformer_orch._embedding import Embedding
from main.transformer_orch._positional_embedding import PositionalEmbedding
from main.transformer_orch._post_attention import PostAttention
from main.transformer_orch._attention import Attention
from main.transformer_orch._transformer import Transformer
from main.transformer_orch._transformer_block import TransformerBlock
from main.transformer_orch._model_orc import Model


class Indexes:
  def __init__(self, wti):
    self.UNK_IDX = wti['<UNK>']
    self.PAD_IDX = wti['<PAD>']
    self.BOS_IDX = wti['<BOS>']
    self.EOS_IDX = wti['<EOS>']

class Data:
    def __init__(self, wti, itw):
        self.wti = wti
        self.itw = itw


class Locales:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 256
        self.X = list()
        self.y = list()



class DataLoaderConfig:
    def __init__(self, X, y, batch_size=64, shuffle=False, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, drop_last=drop_last, shuffle=self.shuffle)

class BackPropConfig:
    def __init__(self, model):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)

class ModelConfig:
    def __init__(self, vocab_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = int(vocab_size)
        self.dim = int(256)
        self.head = int(4)
        self.dropout = float(0.1)
        self.ntbl = int(6)

class TrainConfig:
    def __init__(self, EPOCHS=3, NORM=1.0):
        self.EPOCHS = int(EPOCHS)
        self.NORM = float(NORM)

class ModelOrchestrator:
    def __init__(self, config):
        self.EmbeddingModel = Embedding(
            vocab_size=config.vocab_size,
            dims=config.dim,
        )

        self.PositionalEmbeddingModel = PositionalEmbedding(
            # vocab_size=config.vocab_size,
            max_seq_len=256,
            dims=config.dim,
        )

        self.TransformerBlockLayers = nn.ModuleList([TransformerBlock(
            head=config.head,
            dims=config.dim,
            dropout=config.dropout) for _ in range(config.ntbl)])

        self.TransformerModel = Transformer(
            dims=config.dim,
            vocab_size=config.vocab_size,
        )

        self.Device = config.device
        self.ntbl = config.ntbl


class InferenceConfig:
    def __init__(self, max_tokens=20, temperature=0.7, top_k=5, top_p=0.9):
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)

class ModelSave:
    def __init__(self, model, optimizer, scheduler, path="model/model.pth"):
        self.model = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        self.optimizer = optimizer.state_dict()
        self.scheduler = scheduler.state_dict()
        self.path = path

class ConfigSave:
    def __init__(self, inference, transform, config, locale):
        self.inference = inference
        self.transform = transform
        self.config = config
        self.locale = locale
        self.EOS_token = '<EOS>'

class ConfigPath:
    def __init__(self, model_path="bin/model/model.pt", config_path="bin/data/config.pt"):
        self.model_path = model_path
        self.config_path = config_path