#!/usr/bin/env python3

import random

from collections import deque

import torch
from torch import Tensor
from typing import List
from typing import Tuple


class Buffer:
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition: Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]) -> None:
        self.deque.append(transition)

    def batch(self, sample_size) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        samples = random.sample(self.deque, sample_size)
        samples = [list(sample) for sample in samples]

        i_range: int = len(samples[0])
        j_range: int = len(samples[0][0])
        tensor_list = [[[] for _ in range(j_range)] for _ in range(i_range)]
        for i in range(i_range):
            for j in range(j_range):
                # TODO: check typehint below
                tensor_list[i][j] = torch.stack([sample[i][j] for sample in samples])

        return tuple(tensor_list)

    def __len__(self):
        return len(self.deque)
