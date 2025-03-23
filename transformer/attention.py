import torch
from torch import nn, Tensor

from transformer.config import AttentionConfig


class Attention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super(Attention, self).__init__()
        self.cfg = cfg

        self.attention_score_w = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor, *, have_causal_mask: bool = False) -> Tensor:
        # 拆分为多头
        # q k v shape (batch_size,head_num, seq_len, hidden_dim/head_num)
        q = q.reshape(q.size(0), q.size(1), self.cfg.head_num, -1).transpose(-2, -3)
        k = k.reshape(k.size(0), k.size(1), self.cfg.head_num, -1).transpose(-2, -3)
        v = v.reshape(v.size(0), v.size(1), self.cfg.head_num, -1).transpose(-2, -3)

        # 计算注意力权重
        # attention_weight shape(batch_size,head_num, seq_len,seq_len)
        attention_weight = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)

        # 掩码计算
        # 行掩码
        # mask shape (batch_size,1,1,seq_len)
        # 行掩码和列掩码都需要进行，以确保填充位置不会影响有效位置的注意力权重
        mask = mask.unsqueeze(-2).unsqueeze(-2)

        # 此处注意，不要做列掩码， 如果做列掩码，会导致全行都为 -inf，
        # softmax计算后的值出现nan
        # mask = torch.where(mask == 1, mask, mask.transpose(-2, -1))
        if have_causal_mask:
            causal_mask = torch.ones([q.size(2), k.size(2)]).to(k.device)

            # causal_mask shape (1,1, seq_len,seq_len)
            causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

            # mask shape (batch_size,1 seq_len,seq_len)
            mask = torch.where(causal_mask == 1, causal_mask, mask)

        # 进行掩码覆盖
        attention_weight.masked_fill_(mask == 1, -float('inf'))

        # 归一化
        # attention_weight shape(batch_size,head_num, seq_len,seq_len)
        attention_weight = torch.softmax(attention_weight, dim=-1)

        # 计算注意力分数 attention_score shape(batch_size, head_num,seq_len, hidden_dim/head_num )
        attention_score = attention_weight @ v

        # 交换并合并
        attention_score = attention_score.transpose(-2, -3)

        # attention_score shape(batch_size,seq_len,hidden_dim)
        attention_score = attention_score.reshape(attention_score.size(0), attention_score.size(1), -1)

        # 进行线性变换
        attention_score = self.attention_score_w(attention_score)

        return attention_score
