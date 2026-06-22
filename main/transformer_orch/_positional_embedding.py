import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
  def __init__(self, dims: int=64, max_seq_len: int=128):
    super().__init__()

    pe = torch.zeros(max_seq_len, dims)

    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dims, 2).float() * (-math.log(10000.0) / dims))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    x = x + self.pe[:, :x.size(1)]
    return x