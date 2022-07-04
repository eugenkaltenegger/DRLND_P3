#!/usr/bin/env python3

import logging
import numpy


from collections import OrderedDict

import torch
from matplotlib import pyplot
from torch import device
from typing import List

from src.agent_group import AgentGroup


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
        pyplot.legend(loc="upper left")
        pyplot.ylabel('Score')
        pyplot.xlabel('Episode')
        if filename is not None:
            pyplot.savefig(filename)
        if show:
            pyplot.show()

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
    def save_agent_group(agent_group: AgentGroup, filename: str = "checkpoint.pth"):
        checkpoint_dict = agent_group.to_checkpoint_dict()
        torch.save(checkpoint_dict, filename)
        logging.info("\rAGENT SAVED (FILE: {})".format(filename))


    @staticmethod
    def load_agent_group(device: device, filename: str = "checkpoint.pth") -> AgentGroup:
        checkpoint_dict = torch.load(filename)
        agent_group = AgentGroup().from_checkpoint_dict(checkpoint_dict=checkpoint_dict, device=device)
        return agent_group
