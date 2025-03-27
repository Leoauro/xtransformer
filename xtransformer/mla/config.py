from dataclasses import dataclass


@dataclass
class MlaConfig:
    q_down_proj_dim: int = 512  # q 的降秩维度

    # kv 低秩投影维度 rope_dim + kv_compress_dim = 512 + 64 = 576
    kv_compress_dim = 512  # kv 压缩维度

    nope_dim: int = 128  # 不编码维度
    rope_dim = 64  # 需要进行旋转位置编码的维度
    n_heads: int = 8

    hidden_dim: int = 1024
