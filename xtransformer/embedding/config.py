from dataclasses import dataclass


@dataclass
class EmbeddingConfig():
    vocab_size: int = 30000
    hidden_dim: int = 1024
    seq_token_num: int = 200
