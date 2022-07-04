#!/usr/bin/env python3

import random

from collections import deque

import numpy
import torch
from torch import Tensor
from typing import List
from typing import Tuple


class Buffer:
    def __init__(self, size):
        self.size = size
        self.transitions = deque(maxlen=self.size)
        self.weights = deque(maxlen=self.size)

    def push(self,
             transition: Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
             weight: float) -> None:
        self.transitions.append(transition)
        self.weights.append(weight)

    def batch(self, sample_size) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        samples = random.choices(self.transitions, self.normalized_weights(), k=sample_size)
        # Note: the line below can be uncommented for unweighted sampling
        # samples = random.sample(self.transitions, k=sample_size)
        samples = [list(sample) for sample in samples]

        i_range: int = len(samples[0])
        j_range: int = len(samples[0][0])
        tensor_list = [[[] for _ in range(j_range)] for _ in range(i_range)]
        for i in range(i_range):
            for j in range(j_range):
                tensor_list[i][j] = torch.stack([sample[i][j] for sample in samples])

        return tuple(tensor_list)

    def normalized_weights(self):
        weights: List[float] = list(self.weights)
        maximum: float = max(weights)
        minimum: float = min(weights)

        if maximum != minimum:
            normalized_weights = [(float(weight) - minimum + 0.01) / (maximum - minimum) for weight in weights]
            return normalized_weights

        if maximum == minimum:
            normalized_weights = numpy.ones(len(weights))
            return normalized_weights

    def __len__(self):
        return len(self.transitions)
