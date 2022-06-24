#!/usr/bin/env python3

import torch

from collections import OrderedDict


class HyperparametersRange:
    """
    class to hold hyperparameters range
    """

    def __init__(self):
        """
        constructor for hyperparameters range class
        """
        hpr = OrderedDict()

        hpr["episodes"] = [10000]        # maximum episodes
        hpr["steps"] = [1000]            # steps per episode

        hpr["tau"] = [0.5]              # tau (0 < tau < 1)
        hpr["discount"] = [0.9]         # discount factor

        # buffer parameters
        hpr["buffer_size"] = [100]       # replay buffer size
        hpr["buffer_frequency"] = [2]    # replay buffer learning frequency (number of steps)
        hpr["buffer_sample_size"] = [4]  # replay buffer sample size

        # action noise parameters
        hpr["noise_maximum"] = [1.00]    # maximum noise (starting noise)
        hpr["noise_minimum"] = [0.01]    # minimum noise (ending noise)
        hpr["noise_decay"] = [0.99]      # noise reduction factor per steps

        # actor parameters
        hpr["actor_layers"] = [[128, 64]]                       # actor layers
        hpr["actor_activation_function"] = [torch.nn.ReLU]      # actor activation function
        hpr["actor_output_function"] = [torch.nn.Tanh]          # actor output function
        hpr["actor_optimizer"] = [torch.optim.Adam]             # actor optimizer
        hpr["actor_optimizer_learning_rate"] = [0.005]          # actor optimizer learning rate

        # critic parameters
        hpr["critic_layers"] = [[128, 64]]                      # critic layers
        hpr["critic_activation_function"] = [torch.nn.ReLU]     # critic activation function
        hpr["critic_output_function"] = [torch.nn.ReLU]         # critic output function
        hpr["critic_optimizer"] = [torch.optim.Adam]            # critic optimizer
        hpr["critic_optimizer_learning_rate"] = [0.005]         # critic optimizer learning rate

        self.hpr = hpr

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters range as an OrderedDict
        :return: hyperparameters range in an OrderedDict
        """
        return self.hpr
