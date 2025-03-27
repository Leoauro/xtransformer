import torch
from torch import nn

from xtransformer.mla.config import MlaConfig


class QProj(nn.Module):
    def __init__(self, cfg: MlaConfig):
        super(QProj, self).__init__()
        self.cfg = cfg
        self.q_down_proj = nn.Linear(cfg.hidden_dim, cfg.q_down_proj_dim)
        self.q_up_proj = nn.Linear(
            cfg.q_down_proj_dim,
            cfg.n_heads * (cfg.rope_dim + cfg.nope_dim)
        )
        self.layer_norm = nn.LayerNorm(cfg.q_down_proj_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.q_down_proj(x)
        x = self.layer_norm(x)
        x = self.q_up_proj(x)
        # x shape (batch_size,seq_len, n_heads, rope_dim + q_head_nope_dim)
        x = x.view(x.size(0), x.size(1), self.cfg.n_heads, -1)
        # q_rope (batch_size,seq_len, n_heads, rope_dim)
        q_rope = x[..., :self.cfg.rope_dim]
        # q_nope (batch_size,seq_len, n_heads, q_head_nope_dim)
        q_nope = x[..., self.cfg.rope_dim:]
        q_rope = q_rope.transpose(-2, -3)
        q_nope = q_nope.transpose(-2, -3)
        return q_rope, q_nope


if __name__ == '__main__':
    cfg = MlaConfig()
    model = QProj(cfg)
    q_rope, q_nope = model(torch.rand([2, 10, 1024]))
    print(q_rope.shape, q_nope.shape)
