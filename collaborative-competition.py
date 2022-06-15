#!/usr/bin/env python3

import sys
import torch

from typing import List


class CollaborativeCompetition:

    def __init__(self) -> None:
        # device variable
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # environment variables
        self._environment_graphics = None
        self._environment_training = None

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
        pass

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
