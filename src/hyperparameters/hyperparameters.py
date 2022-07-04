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
        hp["discount"] = 0.995                                  # discount factor (gamma)

        # buffer parameters
        hp["buffer_size"] = 10000                               # replay buffer size (overall size of the buffer deque)
        hp["learning_frequency"] = 1                            # learning frequency (after how many episodes to learn)
        hp["batch_size"] = 1000                                 # replay buffer sample size
        hp["buffer_iterations"] = 10                            # number of samples to draw from the buffer and learn
        hp["sample_iterations"] = 1                             # number of iterations a sample is learned

        # action noise parameters
        hp["noise_maximum"] = 1.00                              # maximum noise (starting noise)
        hp["noise_minimum"] = 0.50                              # minimum noise (ending noise)
        hp["noise_decay"] = 0.9995                              # noise reduction factor per steps

        # network parameters
        hp["tau"] = 0.01                                        # tau (0 < tau < 1)

        # actor parameters
        hp["actor_layers"] = [512, 256]                          # actor layers (has to have a length of 2)
        hp["actor_activation_function"] = torch.nn.ReLU         # actor activation function
        hp["actor_output_function"] = torch.nn.Tanh             # actor output function
        hp["actor_optimizer"] = torch.optim.Adam                # actor optimizer
        hp["actor_optimizer_learning_rate"] = 0.001             # actor optimizer learning rate

        # critic parameters
        hp["critic_layers"] = [512, 256]                         # critic layers (has to have a length of 2)
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
