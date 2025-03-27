from xtransformer.moe import SparseMoe, MoeConfig

from xtransformer.transformer.config import AttentionConfig


# class FeedForward(nn.Module):
#     def __init__(self, cfg: AttentionConfig):
#         super(FeedForward, self).__init__()
#         self.ff_w1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.ff_w2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
#         self.relu2 = nn.ReLU()
#
#     def forward(self, x) -> torch.Tensor:
#         # 做前馈
#         x = self.ff_w1(x)
#         x = self.relu1(x)
#         x = self.ff_w2(x)
#         x = self.relu2(x)
#
#         return x


# 将前馈网络改为混合专家模型
class FeedForward(SparseMoe):
    def __init__(self, config: AttentionConfig):
        moe_cfg = MoeConfig(
            in_feature=config.hidden_dim,
            hidden_dim=config.expert_hidden_dim,
            out_feature=config.hidden_dim,

            expert_num=config.expert_num,
            top_k=config.top_k,
            share_num=config.share_num
        )
        super(FeedForward, self).__init__(moe_cfg)
