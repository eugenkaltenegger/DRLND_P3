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
    def plot_scores(scores: List[float], means: List[float], show: bool = False, filename: str = None) -> None:
        """
        function to create a plot of the given scores
        :param scores: scores to plot
        :param means: means to plot
        :param show: if true the plot will be shown
        :param filename: if not None the plot will be stored to the given destination
        :return: None
        """
        pyplot.figure()
        x = numpy.arange(len(scores))
        pyplot.plot(x, scores, label="score")
        pyplot.plot(x, means, label="mean over 100 scores")
        pyplot.ylabel('Score')
        pyplot.xlabel('Episode')
        if show:
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
