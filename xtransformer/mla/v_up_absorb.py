import torch
from torch import nn

from xtransformer.mla.config import MlaConfig


class VUpAbsorb(nn.Module):
    def __init__(self, cfg: MlaConfig):
        super(VUpAbsorb, self).__init__()
        self.out_proj = nn.Linear(cfg.n_heads * cfg.nope_dim, cfg.hidden_dim)

    def forward(self, v_up: torch.Tensor, atten_score: torch.Tensor):
        atten_score = atten_score @ v_up.transpose(-1, -2)
        atten_score = atten_score.transpose(-2, -3)
        atten_score = atten_score.reshape(atten_score.size(0), atten_score.size(1), -1)
        atten_score = self.out_proj(atten_score)
        return atten_score


if __name__ == "__main__":
    v_up_absorb = VUpAbsorb(MlaConfig())
    v_up = torch.randn([6, 128, 512])
    atten_score = torch.randn([2, 6, 1, 512])

    final_atten_socre = v_up_absorb(v_up, atten_score)
    print(final_atten_socre.shape)
