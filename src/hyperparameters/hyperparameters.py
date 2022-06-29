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

        # episode parameters
        hp["episodes"] = 10000                                  # maximum episodes
        hp["steps"] = 1000                                      # steps per episode

        # reward parameters
        hp["discount"] = 0.99                                   # discount factor

        # buffer parameters
        hp["buffer_size"] = 100000                              # replay buffer size
        hp["buffer_frequency"] = 1                              # replay buffer learning frequency (number of steps)
        hp["buffer_sample_size"] = 256                          # replay buffer sample size (batch size)
        hp["buffer_sample_iterations"] = 1                      # replay buffer sample learning iterations

        # action noise parameters
        hp["noise_maximum"] = 1.00                              # maximum noise (starting noise)
        hp["noise_minimum"] = 0.10                              # minimum noise (ending noise)
        hp["noise_decay"] = 0.999                               # noise reduction factor per steps

        # network parameters
        hp["tau"] = 0.001                                       # tau (0 < tau < 1)

        # actor parameters
        hp["actor_layers"] = [256, 128]                         # actor layers
        hp["actor_activation_function"] = torch.nn.ReLU         # actor activation function
        hp["actor_output_function"] = None                      # actor output function
        hp["actor_optimizer"] = torch.optim.Adam                # actor optimizer
        hp["actor_optimizer_learning_rate"] = 0.0001            # actor optimizer learning rate

        # critic parameters
        hp["critic_layers"] = [256, 128]                        # critic layers
        hp["critic_activation_function"] = torch.nn.ReLU        # critic activation function
        hp["critic_output_function"] = None                     # critic output function
        hp["critic_optimizer"] = torch.optim.Adam               # critic optimizer
        hp["critic_optimizer_learning_rate"] = 0.0001           # critic optimizer learning rate

        self.hp = hp

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters as an OrderedDict
        :return: hyperparameters in an OrderedDict
        """
        return self.hp
