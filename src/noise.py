#!/usr/bin/env python3

import numpy
import torch

from numpy import array
from torch import Tensor


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

        self.mu: float = mu
        self.mu_array: array = numpy.ones(self.action_size) * self.mu
        self.theta: float = theta
        self.sigma: float = sigma
        self.state: array = self.mu_array.copy()

        self.reset()

    def reset(self):
        self.state = self.mu_array.copy()
        self.scale_now = self.scale_maximum

    def get_action_noise(self) -> Tensor:
        x: array = self.state
        dx: array = self.theta * (self.mu_array - x) + self.sigma * numpy.random.randn(len(x))
        self.state = x + dx

        return torch.tensor(self.state, dtype=torch.float)

    def get_action_noise_scale(self) -> Tensor:
        self.scale_now = max(self.scale_minimum, self.scale_now * self.scale_decay)
        return self.get_action_noise() * self.scale_now
