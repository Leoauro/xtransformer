import torch
from torch import nn

from xtransformer.mla.config import MlaConfig


class AttentionScore(nn.Module):
    def __init__(self, cfg: MlaConfig):
        super(AttentionScore, self).__init__()
        self.cfg = cfg

    def forward(self, q_nope_with_k_up: torch.Tensor, compressed_kv: torch.Tensor,
                rope_attention: torch.Tensor, mask: torch.Tensor,
                have_causal_mask: bool):
        compressed_kv = compressed_kv.unsqueeze(-3)
        nope_attention = q_nope_with_k_up @ compressed_kv.transpose(-1, -2)
        attention = rope_attention + nope_attention
        attention = attention / (self.cfg.rope_dim + self.cfg.nope_dim) ** 0.5

        if mask is not None:
            mask = mask.unsqueeze(-2).unsqueeze(-2)
            if have_causal_mask:
                causal_mask = torch.ones([attention.size(-2), attention.size(-1)], device=attention.device)
                causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
                mask = torch.where(causal_mask == 1, causal_mask, mask)
            attention.masked_fill_(mask == 1, -float('inf'))

        attention = torch.softmax(attention, dim=-1)

        atten_score = attention @ compressed_kv
        return atten_score


if __name__ == "__main__":
    attention_score = AttentionScore(MlaConfig())
    q_nope_with_k_up = torch.randn([2, 16, 1, 512])
    compressed_kv = torch.randn([2, 12, 512])
    rope_attention = torch.randn([2, 16, 1, 12])
    atten_score = attention_score(q_nope_with_k_up, compressed_kv, rope_attention)
    print(atten_score.shape)
