#!/usr/bin/env python3

import os
import torch

from torch import Tensor
from typing import Tuple, List
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

        self._environment: UnityEnvironment = UnityEnvironment(file_name=absolut_file_path,
                                                               no_graphics=not enable_graphics)
        self._default_brain: BrainParameters = self._environment.brains[self._environment.brain_names[0]]

        self._number_of_agents: Optional[int] = None
        self._states: Optional[List[Tensor]] = None

    def reset(self, brain: BrainParameters = None, train_environment: bool = True) -> None:
        """
        function to reset environment
        :param brain: brain for which the environment is reset
        :param train_environment: parameter to set whether the environment is for training or for evaluation
        :return: None
        """
        brain = brain if brain is not None else self._default_brain
        info = self._environment.reset(train_mode=train_environment)[brain.brain_name]
        states = [info.vector_observations[agent] for agent in range(len(info.agents))]
        states = [torch.tensor(state, dtype=torch.float) for state in states]
        self._number_of_agents = len(info.agents)
        self._states = states

    def states(self) -> List[Tensor]:
        """
        function to get the state of the environment
        :return: the state of the environment
        """
        return self._states

    def step(self, actions: [Tensor], brain: BrainParameters = None) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        function to make a step in the environment
        :param actions: action for the agent
        :param brain: brain for which will execute the actions
        :return: state (for all agents), rewards (for all agents), dones (for all agents) following the execution of the action
        """
        brain = brain if brain is not None else self._default_brain

        actions = [action_value for action in actions for action_value in action.tolist()]

        info = self._environment.step(actions)[brain.brain_name]

        i_range = range(self._number_of_agents)
        rewards: [Tensor] = [torch.tensor([info.rewards[i]], dtype=torch.float) for i in i_range]
        dones: [Tensor] = [torch.tensor([info.local_done[i]], dtype=torch.bool) for i in i_range]
        states: [Tensor] = [torch.tensor(info.vector_observations[i], dtype=torch.float) for i in i_range]

        self._states = states

        return rewards, dones, states

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
            self.reset()

        return self._number_of_agents

    def state_size(self, brain: BrainParameters = None) -> int:
        """
        function to get the size of the state vector
        :param brain: brain for which the size of the state vector is returned
        :return: size of the state vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return int(brain.vector_observation_space_size * brain.num_stacked_vector_observations)

    def action_size(self, brain: BrainParameters = None) -> int:
        """
        function to get the size of the action vector
        :param brain: brain for which the size of the action vector is returned
        :return: size of the action vector for the given brain
        """
        brain = brain if brain is not None else self._default_brain
        return int(brain.vector_action_space_size)
