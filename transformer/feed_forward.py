import torch
from torch import nn

from transformer.config import AttentionConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(FeedForward, self).__init__()
        self.ff_w1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.relu1 = nn.ReLU()
        self.ff_w2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        # 做前馈
        x = self.ff_w1(x)
        x = self.relu1(x)
        x = self.ff_w2(x)
        x = self.relu2(x)

        return x
