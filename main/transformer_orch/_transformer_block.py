from main.transformer_orch._attention import Attention
from main.transformer_orch._post_attention import PostAttention

import torch.nn as nn
import torch

class TransformerBlock(nn.Module):
    def __init__(self, dims, head, dropout):
        super().__init__()
        
        self.attention = Attention(
            dims=dims,
            head=head,
            dropout=dropout
        )

        self.post_attention = PostAttention(
            dims=dims,
            dropout=dropout
        )

    def forward(self, x):
        att = self.attention(x)
        x = self.post_attention(x, att)

        return x