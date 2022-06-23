#!/usr/bin/env python3

import torch

from collections import OrderedDict


class Hyperparameters:
    """
    class to hold hyperparameters
    """

    def __init__(self):
        """
        constructor for hyperparameters class
        """
        hp = OrderedDict()

        hp["episodes"] = 10000          # maximum episodes
        hp["steps"] = 1000              # steps per episode

        hp["buffer_size"] = 100         # replay buffer size
        hp["buffer_frequency"] = 2      # replay buffer learning frequency (number of steps)
        hp["buffer_sample_size"] = 4    # replay buffer sample size

        hp["noise"] = 1.00              # maximum noise (starting noise)
        hp["noise_minimum"] = 0.01      # minimum noise (ending noise)
        hp["noise_reduction"] = 0.99    # noise reduction factor per steps

        hp["tau"] = 0.5                 # tau (0 < tau < 1)
        hp["discount"] = 0.9            # discount factor

        hp["actor_layers"] = [128, 64]
        hp["actor_activation_function"] = torch.nn.ReLU
        hp["actor_output_function"] = torch.nn.Tanh
        hp["actor_optimizer"] = torch.optim.Adam
        hp["actor_optimizer_learning_rate"] = 0.005

        hp["critic_layers"] = [128, 64]
        hp["critic_activation_function"] = torch.nn.ReLU
        hp["critic_output_function"] = torch.nn.ReLU
        hp["critic_optimizer"] = torch.optim.Adam
        hp["critic_optimizer_learning_rate"] = 0.005

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters as an OrderedDict
        :return: hyperparameters in an OrderedDict
        """
        return self.hp
