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

    def reset_environment(self) -> List[Tensor]:
        if self._environment is None:
            self._environment = Environment(enable_graphics=self._environment_graphics)
            self._environment.reset(train_environment=self._environment_training)

        if self._environment is not None:
            self._environment.reset(train_environment=self._environment_training)

        return self._environment.states()

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
            scores = self.train()
            Utils.plot_scores(scores=scores, plot=True)

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
        best_score_occurance = None

        buffer = Buffer(self._hyperparameters["buffer_size"])

        scores = []
        for episode in range(1, self._hyperparameters["episodes"] + 1):

            states: List[Tensor] = self.reset_environment()
            rewards_in_this_episode = []

            for step in range(1, self._hyperparameters["steps"] + 1):
                local_action = self._agent_group.act(states=states, noise=True)
                local_reward, local_done, local_next_state = self._environment.step(actions=local_action)

                global_state = Utils.local_to_global(states)
                global_action = Utils.local_to_global(local_action)
                global_reward = Utils.local_to_global(local_reward)
                global_done = Utils.local_to_global(local_done)
                global_next_state = Utils.local_to_global(local_next_state)

                buffer.push(transition=(global_state, global_action, global_reward, global_done, global_next_state))
                rewards_in_this_episode.append(global_reward.tolist())

                if len(buffer) > self._hyperparameters["buffer_sample_size"]:
                    batch = buffer.batch(self._hyperparameters["buffer_sample_size"])
                    self._agent_group.update(*batch)

                states = local_next_state

                if any(global_done):
                    # print("Steps in episode: {:4d} (current buffer size: {:6d})".format(step, len(buffer)))
                    break

            # TODO start: the following lines are for debugging purposes only - use proper logger instead
            score0 = sum([x[0] for x in rewards_in_this_episode])
            score1 = sum([x[1] for x in rewards_in_this_episode])
            score = round(max(score0, score1) * 10000) / 10000
            scores.append(score)
            if best_score is None or score > best_score:
                best_score = score
                best_episode = episode
                best_score_occurance = 1

            if score == best_score:
                best_score_occurance += 1

            print("episode {:4d}: score: {:2.2f} [best score: {:2.2f}, episode: {:4d}, appearance: {:3d}]"
                  .format(episode, score, best_score, best_episode, best_score_occurance))
            # TODO: end
        return scores

    def tune(self) -> List[float]:
        pass

    def show(self) -> None:
        pass


if __name__ == "__main__":
    # ARG 1: OPERATION MODE
    mode_arg = sys.argv[1]
    CollaborativeCompetition().run(mode=mode_arg)
