from dataclasses import dataclass


@dataclass
class AttentionConfig():
    head_num: int = 4
    attention_layer: int = 6
    hidden_dim: int = 512
    output_dim: int = 512
