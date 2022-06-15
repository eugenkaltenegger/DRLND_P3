#!/usr/bin/env python3

import sys

import numpy
import torch

from torch import device
from typing import List, Optional

from environment import Environment


class CollaborativeCompetition:

    def __init__(self) -> None:
        # device variable
        self._device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # environment variables
        self._environment: Optional[Environment] = None
        self._environment_graphics: Optional[bool] = None
        self._environment_training: Optional[bool] = None

    def enable_training(self) -> None:
        """
        function to enable training mode
            training mode has to be either enabled or disabled
        :return: None
        """
        self._environment_graphics = False
        self._environment_training = True

    def disable_training(self) -> None:
        """
        function to disable training mode
            training mode has to be either enabled or disabled
        :return: None
        """
        self._environment_graphics = True
        self._environment_training = False

    def reset_environment(self):
        """
        function to reset the environment
        :return: None
        """
        if self._environment is None:
            self._environment = Environment(enable_graphics=self._environment_graphics)
            self._environment.reset(train_environment=self._environment_training)

        if self._environment is not None:
            self._environment.reset(train_environment=self._environment_training)

    def reset_agent(self):
        pass

    def run(self, mode) -> None:
        pass

    def train(self) -> List[float]:
        pass

    def tune(self) -> List[float]:
        pass

    def show(self) -> None:
        pass


if __name__ == "__main__":
    # ARG 1: OPERATION MODE
    mode_arg = sys.argv[1]
    CollaborativeCompetition().run(mode=mode_arg)
