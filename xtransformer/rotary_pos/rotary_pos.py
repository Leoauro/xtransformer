import math
from typing import List

import torch
from torch import nn


# 旋转位置编码是相对位置编码
# 旋转位置编码要在拆分多头之后再应用， 因为旋转位置编码在不同维度上旋转的角度不同，
# 如果在拆分多头前应用旋转位置编码，那么将导致拆分后的每个子空间位置编码信息不一致，因为频率不一致
# 导致模型学习能力下降

class RotaryPos(nn.Module):
    def __init__(self):
        super(RotaryPos, self).__init__()

    def forward(self, d: int, *, seq_len: int = 1024, cur_pos: int = None) -> (torch.Tensor, torch.Tensor):
        sin_list = []
        cos_list = []
        if cur_pos is not None:
            sin_m, cos_m = self.sin_cos(d, cur_pos)
            sin_list.append(sin_m)
            cos_list.append(cos_m)
            return torch.tensor(sin_list), torch.tensor(cos_list)
        for m in range(seq_len):
            sin_m, cos_m = self.sin_cos(d, m)
            sin_list.append(sin_m)
            cos_list.append(cos_m)
        return torch.tensor(sin_list), torch.tensor(cos_list)

    def sin_cos(self, d: int, cur_pos: int) -> (List[float], List[float]):
        sin_m = []
        cos_m = []
        for i in range(0, d, 2):
            theta = cur_pos * (1 / 10000 ** (i / d))
            sin_m = sin_m + [math.sin(theta)] * 2
            cos_m = cos_m + [math.cos(theta)] * 2
        return sin_m, cos_m


def apply_seq_pos(x: torch.Tensor, *, pos: int = None) -> torch.Tensor:
    rope = RotaryPos()
    sin_matrix, cos_matrix = rope(x.size(-1), seq_len=x.size(-2), cur_pos=pos)
    sin_matrix, cos_matrix = sin_matrix.to(device=x.device), cos_matrix.to(device=x.device)
    even = [
        i for i in range(0, x.size(-1), 2)
    ]
    odd = [
        i for i in range(1, x.size(-1), 2)
    ]
    x_even = x[..., even]
    x_odd = x[..., odd]
    x_odd = x_odd * -1
    merged_x = torch.cat([x_odd, x_even], dim=-1)
    index = []
    for i in range(x_odd.size(-1)):
        index.append(i)
        index.append(i + x_odd.size(-1))

    sin_x = merged_x[..., index]

    ret = x * cos_matrix + sin_x * sin_matrix
    return ret
