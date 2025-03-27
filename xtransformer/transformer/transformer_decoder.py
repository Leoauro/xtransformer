import torch
from torch import nn, Tensor

from xtransformer.transformer.add_norm import AddNorm
from xtransformer.transformer.attention import Attention
from xtransformer.transformer.config import AttentionConfig
from xtransformer.transformer.feed_forward import FeedForward


class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(TransformerDecoderBlock, self).__init__()
        self.cfg = cfg
        self.q_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.k_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.encoder_attention_k_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.encoder_attention_k_v = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        self.attention_block = Attention(cfg)
        self.add_norm1 = AddNorm(cfg)

        self.add_norm2 = AddNorm(cfg)

        self.cross_attention_block = Attention(cfg)

        self.ff = FeedForward(cfg)

        self.add_norm3 = AddNorm(cfg)

    def forward(self, target_x: Tensor, target_mask: Tensor, encoder_attention: Tensor, input_mask: Tensor) -> Tensor:
        # input_x shape (batch_size,seq_len,hidden_dim)
        # mask shape (batch_size,seq_len)
        # 得到 q k v
        q = self.q_w(target_x)
        k = self.k_w(target_x)
        v = self.v_w(target_x)
        attention_score = self.attention_block(q, k, v, target_mask, have_causal_mask=True)
        attention_score = self.add_norm1(target_x, attention_score)

        encoder_attention_k = self.encoder_attention_k_w(encoder_attention)
        encoder_attention_v = self.encoder_attention_k_v(encoder_attention)

        cross_attention = self.cross_attention_block(
            attention_score,
            encoder_attention_k,
            encoder_attention_v,
            input_mask,
        )

        cross_attention = self.add_norm2(attention_score, cross_attention)

        ff_cross_attention, _ = self.ff(cross_attention)

        ff_cross_attention = self.add_norm3(cross_attention, ff_cross_attention)

        return ff_cross_attention


class TransformerDecoder(torch.nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(TransformerDecoder, self).__init__()
        self.cfg = cfg
        self.block_list = nn.ModuleList(
            [
                TransformerDecoderBlock(cfg) for _ in range(cfg.attention_layer)
            ]
        )

    def forward(self, target_x: Tensor, target_mask: Tensor, encoder_attention: Tensor, input_mask: Tensor) -> Tensor:
        for block in self.block_list:
            target_x = block(target_x, target_mask, encoder_attention, input_mask)
        return target_x
