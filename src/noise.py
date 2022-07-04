#!/usr/bin/env python3

import copy
import numpy
import torch

from numpy import array
from torch import Tensor
from typing import Optional


class Noise:

    def __init__(self,
                 action_size: int,
                 scale_maximum: float,
                 scale_minimum: float,
                 scale_decay: float,
                 mu: float = 0,
                 theta: float = 0.15,
                 sigma: float = 0.20) -> None:
        self.action_size: int = action_size

        self.scale_maximum: float = scale_maximum
        self.scale_minimum: float = scale_minimum
        self.scale_decay: float = scale_decay
        self.scale_now: float = 0

        self.mu: float = mu * numpy.ones(action_size)
        self.theta: float = theta
        self.sigma: float = sigma
        self.state: Optional[float] = None

        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)
        self.scale_now = self.scale_maximum

    def get_action_noise(self) -> Tensor:
        x: array = self.state
        dx: array = self.theta * (self.mu - x) + self.sigma * numpy.random.randn(self.action_size)
        self.state = x + dx

        return torch.tensor(self.state, dtype=torch.float)

    def get_action_noise_scale(self) -> Tensor:
        self.scale_now = max(self.scale_minimum, self.scale_now * self.scale_decay)
        return self.get_action_noise() * self.scale_now
