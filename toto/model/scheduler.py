# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import math

import torch


class WarmupStableDecayLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 1e-5,
        base_lr: float = 1e-3,
        last_epoch: int = -1,
    ):
        """
        Enhanced Warmup-Stable-Decay (WSD) Scheduler with 1-sqrt decay.

        Parameters:
        - optimizer: The optimizer to apply the learning rate schedule to.
        - warmup_steps: Number of steps to linearly increase the learning rate.
        - max_steps: Total number of training steps.
        - stable_ratio: Proportion of stable steps in the remaining steps (0 <= stable_ratio <= 1).
        - min_lr: Minimum learning rate.
        - base_lr: Target learning rate after warmup.
        """
        # Automatically compute phase lengths
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps

        self.min_lr = min_lr
        self.base_lr = base_lr
        self.total_steps = warmup_steps + stable_steps + decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup phase
            factor = step / self.warmup_steps
            return [self.min_lr + factor * (self.base_lr - self.min_lr) for _ in self.optimizer.param_groups]
        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase
            return [self.base_lr for _ in self.optimizer.param_groups]
        elif step < self.total_steps:
            # 1-sqrt decay phase
            decay_progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
            factor = 1 - math.sqrt(decay_progress)
            return [self.min_lr + factor * (self.base_lr - self.min_lr) for _ in self.optimizer.param_groups]
        else:
            # Beyond total steps, use min_lr
            return [self.min_lr for _ in self.optimizer.param_groups]
