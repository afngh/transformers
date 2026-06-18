import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, EmbeddingModel, PositionalEmbeddingModel, Attention, PostAttention, Transformer, Device):
    super().__init__()
    self.to(Device)
    self.e = EmbeddingModel
    self.pe = PositionalEmbeddingModel
    self.a = Attention
    self.pa = PostAttention
    self.t = Transformer

  def forward(self, x):
    wv = self.e(x)
    pwe = self.pe(wv)
    att = self.a(pwe)
    post = self.pa(pwe, att)

    last_vector = post[:,-1:,:]
    logits = self.t(last_vector)

    return logits