import torch
from torch import nn


class KUpAbsorb(nn.Module):
    def __init__(self):
        super(KUpAbsorb, self).__init__()

    def forward(self, k_up: torch.Tensor, q_nope: torch.Tensor) -> torch.Tensor:
        q_nope_with_k_up = q_nope @ k_up
        return q_nope_with_k_up


if __name__ == "__main__":
    k_up_absorb = KUpAbsorb()
    k_up = torch.randn([16, 128, 512])
    q_nope = torch.randn([2, 16, 1, 128])
    q_nope_with_k_up = k_up_absorb(k_up, q_nope)
    print(q_nope_with_k_up.shape)
