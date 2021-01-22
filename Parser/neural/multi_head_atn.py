from typing import *

import torch
from torch import Tensor, LongTensor
from torch.nn import Module, Linear, Dropout
from opt_einsum import contract


def multihead_attn_fn(queries: Tensor, keys: Tensor, values: Tensor,
                      mask: Optional[LongTensor] = None) -> Tensor:
    return mh_scaled_dot_product(queries, keys, values, mask)


def mh_scaled_dot_product(queries: Tensor, keys: Tensor, values: Tensor,
                          mask: Optional[LongTensor] = None) -> Tensor:
    dk, num_heads = keys.shape[-2:]
    dividend = torch.sqrt(torch.tensor(dk, device=queries.device, dtype=torch.float))

    weights = contract('bidh,bodh->bioh', queries, keys) / dividend
    if mask is not None:
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = weights.softmax(dim=-2)
    return torch.einsum('bioh,bodh->bidh', weights, values).flatten(-2)


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, d_q_in: int, d_k_in: int, d_v_in: int,
                 d_atn: int, d_v: int, d_out: int, dropout_rate: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.q_transformation = Linear(in_features=d_q_in, out_features=d_atn*num_heads, bias=False)
        self.k_transformation = Linear(in_features=d_k_in, out_features=d_atn*num_heads, bias=False)
        self.v_transformation = Linear(in_features=d_v_in, out_features=d_v*num_heads, bias=False)
        self.wo = Linear(in_features=num_heads * d_v, out_features=d_out, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Optional[LongTensor] = None) -> Tensor:
        qs = self.q_transformation(queries).view(queries.shape[0], queries.shape[1], -1, self.num_heads)
        ks = self.k_transformation(keys).view(keys.shape[0], keys.shape[1], -1, self.num_heads)
        vs = self.v_transformation(values).view(values.shape[0], values.shape[1], -1, self.num_heads)
        mha = multihead_attn_fn(qs, ks, vs, mask)
        mha = self.dropout(mha)
        return self.wo(mha)
