""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise, k-decay.

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
import numpy as np
import torch

from .scheduler_main import Scheduler


_logger = logging.getLogger(__name__)


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """
    print("1111111111111111111111111111111111111111111111111")
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min: float = 0.,
                 cycle_mul: float = 1.,
                 cycle_decay: float = 1.,
                 cycle_limit: int = 1,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 k_decay=1.0,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning("Cosine annealing scheduler will have no effect on the learning "
                           "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        self.k_decay = k_decay
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        
        print("22222222222222222222222222222222222222")
        print(t,self.warmup_t)
        if t < self.warmup_t:
            print("2.1")
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            print("2.2")
            if self.warmup_prefix:
                print("2.2.1")
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                print("2.2.2")
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                print("2.2.3")
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)
            print("2.3")
            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay
            print("2.4")
            if i < self.cycle_limit:
                print("2.4.1")
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                print("2.4.2")
                lrs = [self.lr_min for _ in self.base_values]
        print(lrs)
        return lrs

    def get_epoch_values(self, epoch: int):
        print("333333333333333333333333333333333333333333")
        print(self.t_in_epochs)
        if self.t_in_epochs:
            print("3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1 3.1")
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        print("4444444444444444444444444444444444444444444")
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        print("555555555555555555555555555555555555555555")
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))