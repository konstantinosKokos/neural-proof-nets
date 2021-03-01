from math import ceil, cos, radians
from typing import *

from torch.optim import Optimizer

from torch.utils.data.dataloader import DataLoader

from ..data.preprocessing import Sample
from ..neural.utils import batchify_vectorized_samples, Item


def make_cyclic_triangular_schedule(max_lr: float, warmup_steps: int, triangle_decay: int, decay_over: int) \
        -> Callable[[int], float]:
    linear_factor = max_lr / warmup_steps
    cos_window = make_cosine_window(max_lr, warmup_steps, decay_over - warmup_steps)

    def schedule(step: int):
        if step < warmup_steps:
            return linear_factor * step
        num_triangles = (step - warmup_steps) // triangle_decay
        init_step = num_triangles * triangle_decay
        init_lr = cos_window(init_step + warmup_steps)
        down_factor = init_lr / triangle_decay
        return init_lr - down_factor * (step - init_step - warmup_steps)
    return schedule



def make_cosine_schedule(max_lr: float, warmup_steps: int, decay_over: int) -> Callable[[int], float]:
    """
        Makes a schedule that increases the lr from 0 to max_lr over warmup_steps,
        then reduces it to 0 over decay_over steps.
    """
    linear_factor = max_lr / warmup_steps
    cos_window = make_cosine_window(max_lr, warmup_steps, decay_over - warmup_steps)

    def cosine_schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_factor * step
        return cos_window(step)

    return cosine_schedule


def make_cosine_window(max_lr: float, offset: int, decay_over: int) -> Callable[[int], float]:
    f = 90 / decay_over
    b = - f * offset

    def schedule(step: int) -> float:
        angle = f * step + b
        return cos(radians(angle)) * max_lr

    return schedule


class Scheduler:
    def __init__(self, opt: Optimizer, schedule: Callable[[int], float], lr_scales: Optional[list[float]] = None):
        self.opt = opt
        self.schedule = schedule
        self.step_num = 0
        self.lr = 0.
        if lr_scales is None:
            lr_scales = (1,) * len(self.opt.param_groups)
        self.lr_scales = lr_scales

    def step(self) -> None:
        self.step_num += 1
        self.lr = self.schedule(self.step_num)
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lr * self.lr_scales[i]
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()


def make_dataloader(samples: list[Item], spad: int, wpad: int, max_difficulty: int, exclude_singular: bool,
                    batch_size: int, shuffle: bool) -> DataLoader:

    return DataLoader(samples, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda x: batchify_vectorized_samples(inps=x, padding_value_word=wpad,
                                                                       padding_value_symbol=spad,
                                                                       max_difficulty=max_difficulty,
                                                                       exclude_singular=exclude_singular))


def get_nbatches(trainset: List[Sample], batch_size: int) -> int:
    return ceil(len(trainset) / batch_size)
