from main.transformer_orch._attention import Attention
from main.transformer_orch._post_attention import PostAttention

import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, dims, head, dropout):
        super().__init__()
        self.block = PostAttention(dims=dims, dropout=dropout, head=head)

    def forward(self, x):
        return self.block(x)