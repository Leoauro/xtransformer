from torch import nn, Tensor

from xtransformer.transformer.add_norm import AddNorm
from xtransformer.transformer.attention import Attention
from xtransformer.transformer.config import AttentionConfig
from xtransformer.transformer.feed_forward import FeedForward


class EncoderOnlyBlock(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(EncoderOnlyBlock, self).__init__()
        self.cfg = cfg
        self.q_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.k_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.attention_block = Attention(cfg)
        self.add_norm1 = AddNorm(cfg)
        self.ff = FeedForward(cfg)
        self.add_norm2 = AddNorm(cfg)

    def forward(self, input_x: Tensor, mask: Tensor) -> Tensor:
        # input_x shape (batch_size,seq_len,hidden_dim)
        # mask shape (batch_size,seq_len)
        # 得到 q k v
        q = self.q_w(input_x)
        k = self.k_w(input_x)
        v = self.v_w(input_x)
        attention_score = self.attention_block(q, k, v, mask)
        attention_score = self.add_norm1(input_x, attention_score)

        ff_attention_score, _ = self.ff(attention_score)
        attention_score = self.add_norm2(attention_score, ff_attention_score)

        return attention_score


class EncoderOnly(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(EncoderOnly, self).__init__()
        self.cfg = cfg
        self.block_list = nn.ModuleList(
            [
                EncoderOnlyBlock(cfg) for _ in range(cfg.attention_layer)
            ]
        )
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, input_x: Tensor, mask: Tensor) -> Tensor:
        for block in self.block_list:
            input_x = block(input_x, mask)
        out = self.output_layer(input_x)
        return out
