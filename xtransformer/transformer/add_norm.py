from torch import Tensor, nn

from xtransformer.transformer.config import AttentionConfig


class AddNorm(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, pre: Tensor, cur: Tensor) -> Tensor:
        # 标准化
        ret = self.layer_norm(cur)
        # 残差
        ret = ret + pre
        ret = self.dropout(ret)
        return ret
