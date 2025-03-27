import torch
from torch import nn

from xtransformer.embedding.config import EmbeddingConfig


class Embedding(nn.Module):
    def __init__(self, cfg: EmbeddingConfig):
        super(Embedding, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embedding = nn.Embedding(cfg.seq_token_num, cfg.hidden_dim)
        # post shape(1, cfg.seq_token_num)
        self.pos_t = torch.arange(0, cfg.seq_token_num).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch_size,seq_len)
        # word_embedding shape (batch_size,seq_len,hidden_dim)
        word_embedding = self.word_embedding(x)
        # cur_pos shape (1,seq_len)
        cur_pos = self.pos_t[:, :x.size(1)].to(x.device)
        pos_embedding = self.pos_embedding(cur_pos)
        # out shape (batch_size,seq_len,hidden_dim)
        out = word_embedding + pos_embedding
        return out
