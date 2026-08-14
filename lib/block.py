from lib.multihead import MultiHead
from lib.feedforward import FeedForward
import torch
import torch.nn as nn

torch.manual_seed(1337)

class Block(nn.Module):
    def __init__(self, n_embd,  n_heads, block_size):
        super().__init__()
        self.heads = MultiHead(n_embd, n_heads, block_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+ self.heads(self.ln1(x))
        x = x+ self.ff(self.ln2(x))
        return x