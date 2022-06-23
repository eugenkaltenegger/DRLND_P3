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

        hpr["buffer_size"] = [100]       # replay buffer size
        hpr["buffer_frequency"] = [2]    # replay buffer learning frequency (number of steps)
        hpr["buffer_sample_size"] = [4]  # replay buffer sample size

        hpr["noise"] = [1.00]            # maximum noise (starting noise)
        hpr["noise_minimum"] = [0.01]    # minimum noise (ending noise)
        hpr["noise_reduction"] = [0.99]  # noise reduction factor per steps

        hpr["tau"] = [0.5]               # tau (0 < tau < 1)
        hpr["discount"] = [0.9]          # discount factor

        hpr["actor_layers"] = [[128, 64]]
        hpr["actor_activation_function"] = [torch.nn.ReLU]
        hpr["actor_output_function"] = [torch.nn.Tanh]
        hpr["actor_optimizer"] = [torch.optim.Adam]
        hpr["actor_optimizer_learning_rate"] = [0.005]

        hpr["critic_layers"] = [[128, 64]]
        hpr["critic_activation_function"] = [torch.nn.ReLU]
        hpr["critic_output_function"] = [torch.nn.ReLU]
        hpr["critic_optimizer"] = [torch.optim.Adam]
        hpr["critic_optimizer_learning_rate"] = [0.005]

        self.hpr = hpr

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters range as an OrderedDict
        :return: hyperparameters range in an OrderedDict
        """
        return self.hpr
