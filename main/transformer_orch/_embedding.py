import torch
import torch.nn as nn

class Embedding(nn.Module):
  def __init__(self, dims: int=64, vocab_size: int=5000, dropout: float=0.3):
    super().__init__()

    self.embed = nn.Embedding(vocab_size, dims)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(self.embed(x))