import torch
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, EmbeddingModel, PositionalEmbeddingModel, TransformerBlockLayers, Transformer, Device):
    super().__init__()
    self.e = EmbeddingModel
    self.pe = PositionalEmbeddingModel
    self.blayers = TransformerBlockLayers
    self.t = Transformer
    self.t.output_layer.weight = self.e.embed.weight 
    self.to(Device)

  def forward(self, x):
    wv = self.e(x)
    post = self.pe(wv)
    for tblayer in self.blayers:
      post = tblayer(post)

    # last_vector = post[:,-1:,:]
    logits = self.t(post)

    return logits