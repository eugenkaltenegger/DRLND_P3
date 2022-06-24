#!/usr/bin/env python3

import logging
import sys
from collections import OrderedDict

import torch

from torch import device
from torch import Tensor
from typing import List
from typing import Optional

from agent_group import AgentGroup
from buffer import Buffer
from environment import Environment
from hyperparameters.hyperparameters import Hyperparameters
from utils import Utils


class CollaborativeCompetition:

    def __init__(self) -> None:
        # device variable
        self._device: device = torch.device("cpu")  # TODO: FOR DEVELOPMENT ONLY! USE LINE BELOW FOR FINAL VERSION
        # self._device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # hyperparameter variables
        self._hyperparameters: OrderedDict = Hyperparameters().get_dict()
        # self._hyperparameters_range: OrderedDict = HyperparametersRange().get_dict()

        # environment variables
        self._environment: Optional[Environment] = None
        self._environment_graphics: Optional[bool] = None
        self._environment_training: Optional[bool] = None

        # agent group variables
        self._agent_group: Optional[AgentGroup] = None

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

    def reset_environment(self) -> None:
        if self._environment is None:
            self._environment = Environment(enable_graphics=self._environment_graphics)
            self._environment.reset(train_environment=self._environment_training)

        if self._environment is not None:
            self._environment.reset(train_environment=self._environment_training)

    def reset_agent_group(self) -> None:
        if self._environment is None:
            logging.error("\rENVIRONMENT IS REQUIRED TO SETUP AGENT GROUP (PARAMETERS FROM THE ENVIRONMENT ARE REQUIRED)")

        self._agent_group = AgentGroup(device=self._device,
                                       agents=self._environment.number_of_agents(),
                                       state_size=self._environment.state_size(),
                                       action_size=self._environment.action_size(),
                                       hyperparameters=self._hyperparameters)

    def run(self, mode) -> None:
        if mode not in ["train", "tune", "show"]:
            logging.error("\rINVALID OPERATION MODE {} (ALLOWED: train, tune and show)".format(mode))

        if mode == "train":
            score = self.train()

        if mode == "tune":
            score = self.tune()

        if mode == "show":
            self.show()

    def train(self) -> List[float]:
        self.enable_training()
        self.reset_environment()
        self.reset_agent_group()

        best_score = None
        best_episode = None

        buffer = Buffer(self._hyperparameters["buffer_size"])

        for episode in range(1, self._hyperparameters["episodes"] + 1):

            self._environment.reset()
            states: List[Tensor] = self._environment.states()
            episode_rewards = []

            for step in range(1, self._hyperparameters["steps"] + 1):
                actions = self._agent_group.act(states=states, noise=True)

                next_states, rewards, dones = self._environment.step(actions=actions)

                transition = (states, actions, rewards, dones, next_states)

                g_state = Utils.global_view(states)
                g_action = Utils.global_view(actions)
                g_reward = Utils.global_view(rewards)
                g_done = Utils.global_view(dones)
                g_next_state = Utils.global_view(next_states)

                buffer.push(transition=transition)
                episode_rewards.append(Utils.global_view(rewards).tolist())

                if len(buffer) > self._hyperparameters["buffer_sample_size"]:
                    samples = buffer.sample(self._hyperparameters["buffer_sample_size"])

                    for sample in samples:
                        self._agent_group.update(*sample)
                    # self._agent_group.update(*samples)

                    self._agent_group.update_target()

                states = next_states

                done = Utils.global_view(states).tolist()
                if any(done):
                    break

            # TODO start: the following lines are for debugging purposes only - use proper logger instead
            score0 = sum([x[0] for x in episode_rewards])
            score1 = sum([x[1] for x in episode_rewards])
            score = round(max(score0, score1) * 10000) / 10000
            if best_score is None or score > best_score:
                best_score = score
                best_episode = episode
            print("episode {:4d}: score: {:2.2f} [best score: {:2.2f}, episode: {:4d} ]".format(episode, score, best_score, best_episode))
            # TODO: end

    def tune(self) -> List[float]:
        pass

    def show(self) -> None:
        pass


if __name__ == "__main__":
    # ARG 1: OPERATION MODE
    mode_arg = sys.argv[1]
    CollaborativeCompetition().run(mode=mode_arg)
