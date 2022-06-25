#!/usr/bin/env python3

import random

from collections import deque

import torch
from torch import Tensor
from typing import Tuple


class Buffer:
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> None:
        self.deque.append(transition)

    def batch(self, sample_size):
        samples = random.sample(self.deque, sample_size)
        tensors_lists = [[sample[index] for sample in samples] for index in range(len(samples[0]))]
        tensors = [torch.stack(tensors_list) for tensors_list in tensors_lists]

        # a = []
        # for tensors_list in tensors_lists:
        #     print(tensors_list)
        #     b = torch.stack(tensors_list)
        #     print(b)

        return tuple(tensors)

    def __len__(self):
        return len(self.deque)
