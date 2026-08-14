import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.block import Block

torch.manual_seed(1337)

class BigramModel(nn.Module):
    def __init__(self, vocal_size, block_size,  n_emb, n_heads,  n_layers):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocal_size, n_emb)
        self.position_emb_table = nn.Embedding(block_size, n_emb)

        self.blocks = nn.Sequential(*[Block(n_emb, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocal_size)

    def forward(self, idx, target=None):
        B, T = idx.shape

        tok_emb = self.token_embed_table(idx)
        pos_emb = self.position_emb_table(torch.arange(T))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = target.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss


    def generate(self, idx, block_size, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
