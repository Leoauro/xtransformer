import torch
from torch import nn

from xtransformer.rotary_pos.rotary_pos import apply_seq_pos


class RopeAttention(nn.Module):
    def __init__(self):
        super(RopeAttention, self).__init__()

    def forward(self, q_rope: torch.Tensor, k_rope: torch.Tensor, *, q_position: int = None) -> torch.Tensor:
        if q_position is not None:
            q_rope = apply_seq_pos(q_rope, pos=q_position)
        else:
            q_rope = apply_seq_pos(q_rope)
        k_rope = k_rope.unsqueeze(-3)
        k_rope = apply_seq_pos(k_rope)
        rope_attention = q_rope @ k_rope.transpose(-2, -1)
        return rope_attention


if __name__ == '__main__':
    ra = RopeAttention()
    q_rope = torch.randn(2, 16, 1, 64)
    k_rope = torch.randn(2, 12, 64)
    attention = ra(q_rope, k_rope)
    print(attention.shape)
