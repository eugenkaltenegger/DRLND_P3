#!/usr/bin/env python3

import torch

from collections import OrderedDict


class HyperparametersRange:
    """
    class to hold hyperparameters range
    Note: the hyperparameters are documented on the file hyperparameters.py
    """

    def __init__(self):
        """
        constructor for hyperparameters range class
        """
        hpr = OrderedDict()

        # episode parameters
        hpr["episodes"] = [10000]
        hpr["steps"] = [1000]

        # reward parameters
        hpr["discount"] = [0.99]

        # buffer parameters
        hpr["buffer_size"] = [10000]
        hpr["learning_frequency"] = [1]
        hpr["batch_size"] = [1000]
        hpr["buffer_iterations"] = [10]
        hpr["sample_iterations"] = [1]

        # action noise parameters
        hpr["noise_maximum"] = [1.00]
        hpr["noise_minimum"] = [0.10]
        hpr["noise_decay"] = [0.9995]

        # network parameters
        hpr["tau"] = [0.01]

        # actor parameters
        hpr["actor_layers"] = [[512, 256]]
        hpr["actor_activation_function"] = [torch.nn.ReLU]
        hpr["actor_output_function"] = [torch.nn.Tanh]
        hpr["actor_optimizer"] = [torch.optim.Adam]
        hpr["actor_optimizer_learning_rate"] = [0.001]

        # critic parameters
        hpr["critic_layers"] = [[512, 256]]
        hpr["critic_activation_function"] = [torch.nn.ReLU]
        hpr["critic_output_function"] = [None]
        hpr["critic_optimizer"] = [torch.optim.Adam]
        hpr["critic_optimizer_learning_rate"] = [0.0001]

        self.hpr = hpr

    def get_dict(self) -> OrderedDict:
        """
        function to get hyperparameters range as an OrderedDict
        :return: hyperparameters range in an OrderedDict
        """
        return self.hpr
