import torch
import torch.nn as nn
from ._swiglu_activation import SwiGLUFFN
from ._attention import Attention

class PostAttention(nn.Module):
  def __init__(self, dims: int=64, dropout: float=0.3, head: int=4):
    super().__init__()
    self.norm1 = nn.RMSNorm(dims)
    self.norm2 = nn.RMSNorm(dims)
    self.attention = Attention(dims=dims, dropout=dropout, head=head)
    self.swiglu = SwiGLUFFN(dims, dims * 4)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x):
    x = x + self.drop1(self.attention(self.norm1(x)))
    x = x + self.drop2(self.swiglu(self.norm2(x)))
    return x