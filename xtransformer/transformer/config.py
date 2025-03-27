from dataclasses import dataclass


@dataclass
class AttentionConfig():
    head_num: int = 8
    attention_layer: int = 6
    hidden_dim: int = 1024
    output_dim: int = 1024

    expert_num: int = 6  # 专家数量
    top_k: int = 2  # 每次选择的top_k 专家
    share_num: int = 2  # 共享专家数量
    expert_hidden_dim: int = 512
