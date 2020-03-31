from math import cos, radians, ceil
from typing import *

from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from PermutationParser.data.sample import Sample


def make_cosine_schedule(max_lr: float, warmup_steps: int, decay_over: int) -> Callable[[int], float]:
    """
        Makes a schedule that increases the lr from 0 to max_lr over warmup_steps,
        then reduces it to 0 over decay_over steps.
    """
    linear_factor = max_lr / warmup_steps
    cos_window = make_cosine_window(max_lr, warmup_steps, decay_over)

    def cosine_schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_factor * step
        else:
            return cos_window(step)

    return cosine_schedule


def make_cosine_window(max_lr: float, offset: int, decay_over: int) -> Callable[[int], float]:
    f = 90 / decay_over
    b = - f * offset

    def schedule(step: int) -> float:
        angle = f * step + b
        return cos(radians(angle)) * max_lr

    return schedule


def make_cosine_schedule_with_restarts(max_lr: float, warmup_steps: int, restart_every: int,
                                       decay_over: int) -> Callable[[int], float]:
    linear_factor = max_lr / warmup_steps
    outer_cos_fn = make_cosine_window(max_lr, offset=warmup_steps, decay_over=decay_over)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return linear_factor * step
        else:
            outer_factor = outer_cos_fn(step)
            current_restart = (step - warmup_steps) // restart_every
            offset = warmup_steps + current_restart * restart_every
            inner_cos_fn = make_cosine_window(max_lr=outer_factor,
                                              offset=offset,
                                              decay_over=restart_every)

            return inner_cos_fn(step)

    return schedule


class Scheduler(object):
    def __init__(self, opt: Optimizer, schedule: Callable[[int], float], param_group_scales: Sequence[float] = (1,)):
        self.opt = opt
        self.schedule = schedule
        self.step_num = 0
        self.lr = 0
        self.param_group_scales = param_group_scales

    def step(self) -> None:
        self.step_num += 1
        self.lr = self.schedule(self.step_num)
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lr * self.param_group_scales[i]
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()


class ParseData(Dataset):
    def __init__(self, samples: List[Sample]):
        super(ParseData, self).__init__()
        self.samples = samples

    def __getitem__(self, item: int) -> Sample:
        return self.samples[item]

    def __len__(self) -> int:
        return len(self.samples)


def make_dataloader(samples: List[Sample], batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    dataset = ParseData(samples)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)


def get_max_len(epoch: int) -> int:
    """
        Returns maximum output size as a function of epoch.
    """
    grace_period = 0
    min_len = 20
    increase_every = 20
    increase_by = 10
    max_len = 80
    if epoch < grace_period:
        return min_len
    return min(increase_by * ((epoch - grace_period) // increase_every) + min_len, max_len)


def get_nbatches(epoch_len: int, trainset: List[Sample], batch_size: int) -> int:
    """
        Returns number of batches as a function of epoch.
    """
    dataset_size = len([sample for sample in trainset if len(sample.polish) <= epoch_len])
    return ceil(dataset_size / batch_size)


def lens_to_schedule(batch_lens: Sequence[int], max_lr: float = 5e-04) -> Callable[[int], float]:
    warmup_epoch = list(filter(lambda i: batch_lens[i] != min(batch_lens), range(len(batch_lens))))[0]
    warmup_steps = sum(batch_lens[:warmup_epoch])
    decay_over = sum(batch_lens) - warmup_steps
    return make_cosine_schedule(max_lr=max_lr, warmup_steps=warmup_steps, decay_over=decay_over)


def make_link_weighter(batch_lens: Sequence[int]) -> Callable[[int], float]:
    reach_max_at = list(filter(lambda i: batch_lens[i] == max(batch_lens), range(len(batch_lens))))[0]
    return lambda epoch: min(0.5, max(0, (epoch - reach_max_at) / 2 * reach_max_at * 0.5))
