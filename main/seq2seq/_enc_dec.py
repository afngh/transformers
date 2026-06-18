from torch import Tensor

class EncDec:
  def __init__(self, data, idxs):
    self.itw = data.itw
    self.wti = data.wti

    self.idx = idxs

  def encode(self, text :list):
    data = [self.wti.get(word,self.idx.UNK_IDX) for word in text]
    return data

  def decoder(self, data :Tensor):
    return ' '.join([self.itw.get(word,'<UNK>') for word in data])