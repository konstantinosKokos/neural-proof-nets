from typing import *

import torch
from torch import Tensor, LongTensor
from torch.nn import Module, ModuleList, Linear, Dropout
from torch.nn import functional as F


def multihead_attn_fn(queries: Tensor, keys: Tensor, values: Tensor,
                      qts: ModuleList, kts: ModuleList, vts: ModuleList,
                      mask: Optional[LongTensor] = None) -> Tensor:
    qs = [qt(queries) for qt in qts]
    ks = [kt(keys) for kt in kts]
    vs = [vt(values) for vt in vts]
    outputs = [scaled_dot_product(qs[i], ks[i], vs[i], mask) for i in range(len(qs))]
    return torch.cat(outputs, dim=-1)


def scaled_dot_product(queries: Tensor, keys: Tensor, values: Tensor,
                       mask: Optional[LongTensor] = None) -> Tensor:
    dk = keys.shape[-1]
    dividend = torch.tensor(dk, device=queries.device, dtype=torch.float)

    weights = torch.bmm(queries, keys.transpose(2, 1)) / torch.sqrt(dividend)  # [B, M, N]
    if mask is not None:
        weights = weights.masked_fill_(mask == 0, value=-1e10)
    weights = F.softmax(weights, dim=-1)
    return torch.bmm(weights, values)


class MultiHeadAttention(Module):
    def __init__(self, num_heads: int, d_model: int, d_k: int, d_v: int, dropout_rate: float = 0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        self.q_transformations = ModuleList([Linear(in_features=d_model, out_features=d_k, bias=False)
                                             for _ in range(num_heads)])
        self.k_transformations = ModuleList([Linear(in_features=d_model, out_features=d_k, bias=False)
                                             for _ in range(num_heads)])
        self.v_transformations = ModuleList([Linear(in_features=d_model, out_features=d_v, bias=False)
                                             for _ in range(num_heads)])
        self.Wo = Linear(in_features=num_heads * d_v, out_features=d_model, bias=False)
        self.dropout = Dropout(dropout_rate)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                mask: Optional[LongTensor] = None) -> Tensor:
        mha = multihead_attn_fn(queries, keys, values, self.q_transformations, self.k_transformations,
                                self.v_transformations, mask)
        mha = self.dropout(mha)
        return self.Wo(mha)
