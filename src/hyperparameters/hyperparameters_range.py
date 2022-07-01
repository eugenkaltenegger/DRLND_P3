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

        # episode parameters
        hpr["episodes"] = [10000]                               # maximum episodes
        hpr["steps"] = [1000]                                   # steps per episode

        # reward parameters
        hpr["discount"] = [0.99]                                # discount factor

        # buffer parameters
        hpr["buffer_size"] = [10000]                            # replay buffer size
        hpr["buffer_frequency"] = [1, 2, 4]                           # replay buffer learning frequency (number of steps)
        hpr["buffer_sample_size"] = [256, 128]                       # replay buffer sample size (batch size)
        hpr["buffer_sample_iterations"] = [1, 2, 4]                   # replay buffer sample learning iterations

        # action noise parameters
        hpr["noise_maximum"] = [1.00]                           # maximum noise (starting noise)
        hpr["noise_minimum"] = [0.10]                           # minimum noise (ending noise)
        hpr["noise_decay"] = [0.9999]                           # noise reduction factor per steps

        # network parameters
        hpr["tau"] = [0.1, 0.01, 0.001]                                     # tau (0 < tau < 1)

        # actor parameters
        hpr["actor_layers"] = [[256, 128], [512, 256]]                      # actor layers
        hpr["actor_activation_function"] = [torch.nn.ReLU]      # actor activation function
        hpr["actor_output_function"] = [None, torch.nn.Tanh]                   # actor output function
        hpr["actor_optimizer"] = [torch.optim.Adam]             # actor optimizer
        hpr["actor_optimizer_learning_rate"] = [0.001, 0.0001]         # actor optimizer learning rate

        # critic parameters
        hpr["critic_layers"] = [[256, 128], [512, 256]]                     # critic layers
        hpr["critic_activation_function"] = [torch.nn.ReLU]     # critic activation function
        hpr["critic_output_function"] = [None]                  # critic output function
        hpr["critic_optimizer"] = [torch.optim.Adam]            # critic optimizer
        hpr["critic_optimizer_learning_rate"] = [0.001, 0.0001]         # critic optimizer learning rate

        self.hpr = hpr

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters range as an OrderedDict
        :return: hyperparameters range in an OrderedDict
        """
        return self.hpr
