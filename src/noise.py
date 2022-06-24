#!/usr/bin/env python3

import numpy
import torch

from numpy import array
from torch import Tensor


class Noise:

    def __init__(self, action_size: int, scale_maximum, scale_minimum, scale_decay, mu=0, theta=0.15, sigma=0.2):
        self.action_size: int = action_size

        self.scale_maximum: float = scale_maximum
        self.scale_minimum: float = scale_minimum
        self.scale_decay: float = scale_decay
        self.scale_now: float = 0

        self.mu: float = mu
        self.theta: float = theta
        self.sigma: float = sigma
        self.state: array = numpy.ones(self.action_size) * self.mu

        self.reset()

    def reset(self):
        self.scale_now = self.scale_maximum

    def get_action_noise(self) -> Tensor:
        x: array = self.state
        dx: array = self.theta * (self.mu - x) + self.sigma * numpy.random.randn(len(x))
        self.state = x + dx

        self.scale_now = max(self.scale_now * self.scale_decay, self.scale_minimum)
        return torch.tensor(self.state * self.scale_now, dtype=torch.float)
