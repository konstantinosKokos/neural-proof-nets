from typing import Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.distributions.gumbel import Gumbel


def logsumexp(inputs: Tensor, dim: int = 0, keepdim: bool = False) -> Tensor:
    if dim == 0:
        inputs = inputs.view(-1)
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def norm(x: Tensor, dim: int) -> Tensor:
    return x - logsumexp(x, dim=dim, keepdim=True)


def sinkhorn_step(x: Tensor) -> Tensor:
    return norm(norm(x, dim=1), dim=2)


def sinkhorn_fn(x: Tensor, tau: float, iters: int, eps: float = 1e-20) -> Tensor:
    x = x / tau
    for _ in range(iters):
        x = sinkhorn_step(x)
    return torch.exp(x) + eps


def sinkhorn_fn_no_exp(x: Tensor, tau: float, iters: int, eps: float = 1e-20) -> Tensor:
    x = x / tau
    for _ in range(iters):
        x = sinkhorn_step(x)
    return x


def gumbel_sinkhorn(x: Tensor, tau: float, iters: int, noise: float, eps: float = 1e-20) -> Tensor:
    gumbel = Gumbel(0, 1).sample(x.shape).to(x.device)
    return sinkhorn_fn(x + gumbel * noise, tau, iters, eps)


def averaged_gumbel_sinkhorn(x: Tensor, tau: float, iters: int, noise: float, reps: int, eps: float = 1e-20) -> Tensor:
    x = x.repeat(reps, 1, 1, 1)
    x = gumbel_sinkhorn(x, tau, iters, noise, eps)
    return x.sum(dim=0) / reps


class SinkhornFn(Function):
    @staticmethod
    def forward(ctx, x: Tensor, tau: float, iters: int, eps: float = 1e-20) -> Tensor:
        return sinkhorn_fn(x, tau, iters, eps)

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Tuple[Tensor, None, None, None]:
        return grad_outputs, None, None, None


def sinkhorn(x: Tensor, tau: float, iters: int, eps: float = 1e-20):
    return SinkhornFn.apply(x, tau, iters, eps)
