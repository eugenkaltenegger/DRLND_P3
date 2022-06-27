#!/usr/bin/env python3

import logging
from collections import OrderedDict

import numpy
import torch

from matplotlib import pyplot
from torch import Tensor
from typing import List


class Utils:
    """
    class to hold machine learning utilities
    """
    
    @staticmethod
    def plot_scores(scores: List[float], plot: bool = False, filename: str = None) -> None:
        """
        function to create a plot of the given scores
        :param scores: scores to plot
        :param plot: if true the plot will be shown
        :param filename: if not None the plot will be stored to the given destination
        :return: None
        """
        pyplot.figure()
        pyplot.plot(numpy.arange(len(scores)), scores)
        pyplot.ylabel('Score')
        pyplot.xlabel('Episode')
        if plot:
            pyplot.show()
        if filename is not None:
            pyplot.savefig(filename)

    @staticmethod
    def print_hyperparameters(hyperparameters: OrderedDict) -> None:
        """
        function to print the given hyperparameters
        :param hyperparameters: hyperparameters to print
        :return: None
        """
        for key, value in hyperparameters.items():
            logging.info("\r{}: {}".format(key, value))

    @staticmethod
    def local_to_global(local_view: List[Tensor], dim=0) -> Tensor:
        return torch.cat(tuple(local_view), dim=dim)

    @staticmethod
    def global_to_local(global_view: Tensor, agents):
        return [local_view.t() for local_view in torch.split(global_view.t(), int(global_view.shape[1]/agents), dim=0)]
