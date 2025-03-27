import torch
from torch import nn, Tensor

from xtransformer.mla import MLAAttention, MlaConfig
from xtransformer.transformer.add_norm import AddNorm
from xtransformer.transformer.config import AttentionConfig
from xtransformer.transformer.feed_forward import FeedForward


class DecoderOnlyBlock(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(DecoderOnlyBlock, self).__init__()
        self.cfg = cfg
        mla_cfg = MlaConfig(
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.head_num
        )
        self.attention_block = MLAAttention(mla_cfg)
        self.add_norm1 = AddNorm(cfg)
        self.ff = FeedForward(cfg)
        self.add_norm2 = AddNorm(cfg)

    def forward(self, input_x: Tensor, mask: Tensor) -> Tensor:
        # input_x shape (batch_size,seq_len,hidden_dim)
        # mask shape (batch_size,seq_len)
        # 得到 q k v
        attention_score = self.attention_block(input_x, input_x, mask=mask, have_causal_mask=True)
        attention_score = self.add_norm1(input_x, attention_score)

        ff_attention_score, _ = self.ff(attention_score)
        attention_score = self.add_norm2(attention_score, ff_attention_score)

        return attention_score


class DecoderOnly(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(DecoderOnly, self).__init__()
        self.cfg = cfg
        self.block_list = nn.ModuleList(
            [
                DecoderOnlyBlock(cfg) for _ in range(cfg.attention_layer)
            ]
        )
        self.output_layer = nn.Linear(cfg.hidden_dim, cfg.output_dim)

    def forward(self, input_x: Tensor, mask: Tensor) -> Tensor:
        for block in self.block_list:
            input_x = block(input_x, mask)
        out = self.output_layer(input_x)
        return out


if __name__ == "__main__":
    cfg = AttentionConfig()
    decoder = DecoderOnly(cfg)
    input_x = torch.randn([2, 26, 1024])
    mask = torch.zeros([2, 26])
    mask[:, -10:] = 1
    out = decoder(input_x, mask)
    print(out.shape)
