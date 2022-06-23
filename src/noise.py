#!/usr/bin/env python3

import numpy
import torch

from numpy import array
from torch import Tensor


class Noise:

    def __init__(self, action_size: int, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_size: int = action_size
        self.scale: float = scale
        self.mu: float = mu
        self.theta: float = theta
        self.sigma: float = sigma
        self.state: array = numpy.ones(self.action_size) * self.mu

    def noise(self) -> Tensor:
        x: array = self.state
        dx: array = self.theta * (self.mu - x) + self.sigma * numpy.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale, dtype=torch.float)
