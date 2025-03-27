import torch
from torch import nn

from xtransformer.mla.config import MlaConfig


class KVDownProj(nn.Module):
    def __init__(self, cfg: MlaConfig):
        super(KVDownProj, self).__init__()
        self.cfg = cfg
        self.kv_down_proj = nn.Linear(cfg.hidden_dim, cfg.rope_dim + cfg.kv_compress_dim)
        self.layer_norm = nn.LayerNorm(cfg.kv_compress_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.kv_down_proj(x)
        # k_rope shape(batch_size,seq_len,rpe_dim)
        k_rope = x[..., :self.cfg.rope_dim]
        # compressed_kv shape(batch_size,seq_len,kv_compress_dim)
        compressed_kv = x[..., self.cfg.rope_dim:]
        compressed_kv = self.layer_norm(compressed_kv)
        return k_rope, compressed_kv


if __name__ == "__main__":
    cfg = MlaConfig()
    model = KVDownProj(cfg)
    x = torch.randn([10, 12, 1024])
    k_rope, compressed_kv = model(x)
    print(k_rope.shape, compressed_kv.shape)
