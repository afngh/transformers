import torch
import torch.nn as nn
from ._swiglu_activation import SwiGLUFFN

class PostAttention(nn.Module):
  def __init__(self, dims: int=64, dropout: float=0.3):
    super().__init__()
    self.norm1 = nn.RMSNorm(dims)
    self.norm2 = nn.RMSNorm(dims)
    self.swiglu = SwiGLUFFN(dims, dims * 4)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x, attention):
    x = self.norm1(x + self.drop1(attention))
    x = self.norm2(x + self.drop2(self.swiglu(x)))
    return x