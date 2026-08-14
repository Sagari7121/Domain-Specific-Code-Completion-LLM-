import torch
import torch.nn as nn
from lib.head import Head

torch.manual_seed(1337)
dropout = 0.2
class MultiHead(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(n_heads)]) 
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
