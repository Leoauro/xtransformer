from dataclasses import dataclass


@dataclass
class MoeConfig():
    in_feature: int = 1024  # 输入的维度
    out_feature: int = 1024  # 输出的维度
    hidden_dim: int = 2048  # 隐藏层的维度

    expert_num: int = 6  # 专家数量
    top_k: int = 2  # 每次选择的top_k 专家
    share_num: int = 2  # 共享专家数量
