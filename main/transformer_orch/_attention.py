import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self, dims: int=64, dropout: float=0.3, head: int=4):
    super().__init__()

    self.dims = dims
    self.head = head
    self.hdim = dims // head

    self.dropout = nn.Dropout(dropout)

    self.q = nn.Linear(dims, dims)
    self.k = nn.Linear(dims, dims)
    self.v = nn.Linear(dims, dims)

    self.projection = nn.Linear(dims, dims)

  def forward(self,x):

    B, S, D = x.size()

    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    q = q.view(B, S, self.head, self.hdim).transpose(1, 2)
    k = k.view(B, S, self.head, self.hdim).transpose(1, 2)
    v = v.view(B, S, self.head, self.hdim).transpose(1, 2)

    # scores = torch.matmul(q,k.transpose(-1,-2) / self.hdim ** 0.5)
    scores = torch.matmul(q, k.transpose(-1,-2)) / self.hdim ** 0.5
    mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(x.device)
    scores = scores.masked_fill(mask, float('-inf'))
    attention_weights = torch.softmax(scores, dim=-1)
    attention_weights = self.dropout(attention_weights)

    x = torch.matmul(attention_weights, v)
    x = x.transpose(1, 2).contiguous().view(B, S, D)
    x = self.projection(x)

    return x