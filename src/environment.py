#!/usr/bin/env python3

import os
import torch

from torch import Tensor

from typing import Tuple
from typing import Optional

from unityagents import BrainParameters
from unityagents import UnityEnvironment


class Environment:
    """
    class to wrap the unity environment given for this exercise
    """

    def __init__(self, enable_graphics: bool = False) -> None:
        """
        initializer for the Environment class
        :param enable_graphics: parameter to set whether the environment is visualized or not visualized
        """
        relative_file_path = "../env/Tennis_Linux/Tennis.x86_64"

        current_directory: str = os.path.dirname(__file__)
        absolut_file_path: str = os.path.join(current_directory, relative_file_path)

        self._environment: UnityEnvironment = UnityEnvironment(file_name=absolut_file_path, no_graphics=not enable_graphics)
        self._default_brain: BrainParameters = self._environment.brains[self._environment.brain_names[0]]

        self._number_of_agents: Optional[int] = None
        self._state: Optional[Tensor] = None

    def reset(self, brain: BrainParameters = None, train_environment: bool = True) -> None:
        """
        function to reset environment
        :param brain: brain for which the environment is reset
        :param train_environment: parameter to set whether the environment is for training or for evaluation
        :return: NoReturn
        """
        brain = brain if brain is not None else self._default_brain
        info = self._environment.reset(train_mode=train_environment)[brain.brain_name]
        state = torch.tensor(info.vector_observations, dtype=torch.float)
        self._state = state

    def state(self) -> Tensor:
        """
        function to get the state of the environment
        :return: the state of the environment
        """
        return self._state

    def step(self, action: Tensor, brain: BrainParameters = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        function to make a step in the environment
        :param action: action for the agent
        :param brain: brain for which will execute the actions
        :return: state, reward, done following the execution of the action
        """
        brain = brain if brain is not None else self._default_brain

        action = action.tolist()

        info = self._environment.step(action)[brain.brain_name]

        state: Tensor = torch.tensor(info.vector_observations, dtype=torch.float)
        reward: Tensor = torch.tensor(info.rewards, dtype=torch.float)
        done: Tensor = torch.tensor(info.local_done, dtype=torch.float)

        self._state = state

        return state, reward, done

    def close(self) -> None:
        """
        function to close an environment
        :return: None (to write None on environment on this function call)
        """
        self._environment.close()
        return None

    def number_of_agents(self) -> int:
        """
        function to get the number of agents in the environment
        :return: number of agents in the environment
        """
        if self._number_of_agents is None:
            self._number_of_agents = len(self._environment.reset()[self._default_brain.brain_name].agents)

        return self._number_of_agents

    def state_size(self, brain: BrainParameters = None) -> int:
        """
        function to get the size of the state vector
        :param brain: brain for which the size of the state vector is returned
        :return: size of the state vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return int(brain.vector_observation_space_size)

    def action_size(self, brain: BrainParameters = None) -> int:
        """
        function to get the size of the action vector
        :param brain: brain for which the size of the action vector is returned
        :return: size of the action vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return int(brain.vector_action_space_size)
