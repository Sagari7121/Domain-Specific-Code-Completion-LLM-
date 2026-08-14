import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
dropout = 0.2

class Head(nn.Module):
    def __init__(self, head_size, n_emb, block_size):
        super().__init__()
        self.head_size = head_size

        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # 4, 8, 4

        k = self.key(x) # 4, 8 ,2 because key -> 4, 2
        q = self.query(x) # 4 , 8, 2
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) # 4, 8, 2 @ 4, 2, 8 

        wei = wei * self.head_size ** -0.5

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf')
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


