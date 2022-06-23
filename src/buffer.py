#!/usr/bin/env python3

import random

from collections import deque


class Buffer:
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition):
        """push into the buffer"""
        self.deque.append(transition)

    def sample(self, sample_size):
        """sample from the buffer"""
        return random.sample(self.deque, sample_size)

    def __len__(self):
        return len(self.deque)
