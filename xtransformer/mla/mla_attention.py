import torch
from torch import nn

from xtransformer.mla.attention_score import AttentionScore
from xtransformer.mla.config import MlaConfig
from xtransformer.mla.k_up_absorb import KUpAbsorb
from xtransformer.mla.kv_down_proj import KVDownProj
from xtransformer.mla.kv_up_proj import KVUpProj
from xtransformer.mla.q_proj import QProj
from xtransformer.mla.rope_attention import RopeAttention
from xtransformer.mla.v_up_absorb import VUpAbsorb


class MLAAttention(nn.Module):
    def __init__(self, cfg: MlaConfig):
        super(MLAAttention, self).__init__()
        self.q_proj = QProj(cfg)
        self.kv_down_proj = KVDownProj(cfg)
        self.rope_attention = RopeAttention()
        self.kv_up_proj = KVUpProj(cfg)
        self.k_up_absorb = KUpAbsorb()
        self.attention_score = AttentionScore(cfg)
        self.v_up_absorb = VUpAbsorb(cfg)

    def forward(self, q, kv, *, mask=None, q_position: int = None, have_causal_mask=False):
        q_rope, q_nope = self.q_proj(q)
        k_rope, compressed_kv = self.kv_down_proj(kv)
        rope_attention = self.rope_attention(q_rope, k_rope, q_position=q_position)
        k_up, v_up = self.kv_up_proj()
        q_nope_with_k_up = self.k_up_absorb(k_up, q_nope)
        atten_score = self.attention_score(q_nope_with_k_up, compressed_kv, rope_attention, mask, have_causal_mask)
        atten_score = self.v_up_absorb(v_up, atten_score)
        return atten_score


if __name__ == "__main__":
    mla_attention = MLAAttention(MlaConfig())
    q = torch.randn(2, 1, 1024)
    kv = torch.randn(2, 13, 1024)
    atten_score = mla_attention(q, kv, q_position=12)
    print(atten_score.shape)

    q = torch.randn(2, 13, 1024)
    mask = torch.zeros([2, 13])
    mask[:, -2:] = 1
    atten_score = mla_attention(q, q, mask=mask)
    print(atten_score.shape)
    atten_score2 = mla_attention(q, q, mask=mask)
    print(atten_score2.shape)

    atten_score = mla_attention(q, q, mask=mask, have_causal_mask=True)
    print(atten_score.shape)
