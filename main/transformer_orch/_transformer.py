import torch
import torch.nn as nn

class Transformer(nn.Module):
  def __init__(self, dims: int=64, vocab_size: int=5000, dropout: float=0.3, head: int=4):
    super().__init__()

    self.output_layer = nn.Linear(dims, vocab_size)

  def forward(self, x):
    return self.output_layer(x)