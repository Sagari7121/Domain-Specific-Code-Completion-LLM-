import torch
import torch.nn as nn

torch.manual_seed(1337)
dropout = 0.2

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4 ),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

