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

        hp["episodes"] = 1000           # maximum episodes
        hp["steps"] = 1000              # steps per episode

        hp["tau"] = 0.1                 # tau (0 < tau < 1)
        hp["discount"] = 0.9            # discount factor

        # buffer parameters
        hp["buffer_size"] = 1000        # replay buffer size
        hp["buffer_frequency"] = 8      # replay buffer learning frequency (number of steps)
        hp["buffer_sample_size"] = 16   # replay buffer sample size

        # action noise parameters
        hp["noise_maximum"] = 1.00      # maximum noise (starting noise)
        hp["noise_minimum"] = 0.01      # minimum noise (ending noise)
        hp["noise_decay"] = 0.99        # noise reduction factor per steps

        # actor parameters
        hp["actor_layers"] = [128, 64]                          # actor layers
        hp["actor_activation_function"] = torch.nn.ReLU         # actor activation function
        hp["actor_output_function"] = torch.nn.Tanh             # actor output function
        hp["actor_optimizer"] = torch.optim.Adam                # actor optimizer
        hp["actor_optimizer_learning_rate"] = 0.005             # actor optimizer learning rate

        # critic parameters
        hp["critic_layers"] = [128, 64]                         # critic layers
        hp["critic_activation_function"] = torch.nn.ReLU        # critic activation function
        hp["critic_output_function"] = torch.nn.ReLU            # critic output function
        hp["critic_optimizer"] = torch.optim.Adam               # critic optimizer
        hp["critic_optimizer_learning_rate"] = 0.005            # critic optimizer learning rate

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters as an OrderedDict
        :return: hyperparameters in an OrderedDict
        """
        return self.hp
