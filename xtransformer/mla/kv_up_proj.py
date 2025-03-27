import torch
from torch import nn

from xtransformer.mla.config import MlaConfig


class KVUpProj(nn.Module):
    def __init__(self, cfg: MlaConfig):
        super(KVUpProj, self).__init__()
        self.cfg = cfg
        self.kv_up = nn.Linear(cfg.kv_compress_dim, cfg.n_heads * (cfg.nope_dim + cfg.nope_dim))

    def forward(self) -> (torch.Tensor, torch.Tensor):
        weight = self.kv_up.weight.view(self.cfg.n_heads, -1, self.cfg.kv_compress_dim)
        k_up = weight[:, self.cfg.nope_dim:, :]
        v_up = weight[:, :self.cfg.nope_dim, :]
        return k_up, v_up


if __name__ == '__main__':
    cfg = MlaConfig()
    kv_up_proj = KVUpProj(cfg)
    k_up, v_up = kv_up_proj()
    print(k_up.shape, v_up.shape)
