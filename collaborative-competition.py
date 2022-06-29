#!/usr/bin/env python3

import itertools
import logging
import numpy
import sys
import torch

from collections import OrderedDict
from torch import device
from torch import Tensor
from typing import List
from typing import Tuple
from typing import Optional

from src.agent_group import AgentGroup
from src.buffer import Buffer
from src.environment import Environment
from src.hyperparameters.hyperparameters import Hyperparameters
from src.hyperparameters.hyperparameters_range import HyperparametersRange
from src.utils import Utils


class CollaborativeCompetition:

    def __init__(self) -> None:
        # device variable
        self._device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # hyperparameter variables
        self._hyperparameters: OrderedDict = Hyperparameters().get_dict()
        self._hyperparameters_range: OrderedDict = HyperparametersRange().get_dict()

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
            scores, means, _ = self.train()
            Utils.plot_scores(scores=scores, means=means, show=True, filename="training.png")

        if mode == "tune":
            scores, means, _ = self.tune()
            Utils.plot_scores(scores=scores, means=means, show=True, filename="tuning.png")

        if mode == "show":
            self.show()

    def train(self) -> Tuple[List[float], List[float], bool]:
        self.enable_training()
        self.reset_environment()
        self.reset_agent_group()

        logging.info("\r-------------------------- Hyperparameters ---------------------------")
        Utils.print_hyperparameters(self._hyperparameters)
        logging.info("\r----------------------------------------------------------------------")

        best_score = None
        best_episode = None
        best_score_occurrence = 0

        buffer = Buffer(self._hyperparameters["buffer_size"])

        scores = []
        means = []
        for episode in range(1, self._hyperparameters["episodes"] + 1):

            local_states: List[Tensor] = self.reset_environment()
            rewards_in_this_episode = []

            for step in range(1, self._hyperparameters["steps"] + 1):
                local_states = [local_state.to(device=self._device) for local_state in local_states]
                local_actions = self._agent_group.act(states=local_states, add_noise=True, )
                local_rewards, local_dones, local_next_states = self._environment.step(actions=local_actions)

                local_actions = [local_action.to(device=self._device) for local_action in local_actions]
                local_rewards = [local_reward.to(device=self._device) for local_reward in local_rewards]
                local_dones = [local_done.to(device=self._device) for local_done in local_dones]
                local_next_states = [local_next_state.to(device=self._device) for local_next_state in local_next_states]

                buffer.push(transition=(local_states, local_actions, local_rewards, local_dones, local_next_states))
                rewards_in_this_episode.append(torch.cat(local_rewards).tolist())

                if len(buffer) > self._hyperparameters["buffer_sample_size"]:
                    if len(buffer) % self._hyperparameters["buffer_frequency"] == 0:
                        batch = buffer.batch(self._hyperparameters["buffer_sample_size"])
                        for _ in range(self._hyperparameters["buffer_sample_iterations"]):
                            self._agent_group.update(*batch)

                local_states = local_next_states

                if any(local_dones):
                    # print("Steps in episode: {:4d} (current buffer size: {:6d})".format(step, len(buffer)))
                    break

            # TODO start: the following lines are for debugging purposes only - use proper logger instead
            score0 = sum([x[0] for x in rewards_in_this_episode])
            score1 = sum([x[1] for x in rewards_in_this_episode])
            score = max(score0, score1)
            scores.append(score)

            print_frequency = 100
            steps_until_now = len(scores)

            if steps_until_now >= 100:
                mean = numpy.array(scores[-100:]).mean()
            else:
                mean = 0

            means.append(mean)

            if best_score is None or score > best_score:
                best_score = score
                best_episode = episode
                best_score_occurrence = 1

            if score == best_score and episode != best_episode:
                best_score_occurrence += 1

            if steps_until_now % print_frequency == 0:
                non_null_percentage = sum([1 if score != 0 else 0 for score in scores[-print_frequency:]]) / print_frequency * 100
                print("episode {:6d}: best score: {:6.2f}, occurrence: {:6d}, not zero: {:6.2f}% - MEAN: {:8.4f}"
                      .format(episode, best_score, best_score_occurrence, non_null_percentage, mean))
                best_score = None
                best_episode = None
                best_score_occurrence = 0

            if mean >= 0.5:
                logging.info("\rENVIRONMENT SOLVED!")
                return scores, means, True
            # TODO: end
        return scores, means, False

    def tune(self) -> Tuple[List[float], List[float], bool]:
        for hp_key, hpr_key in zip(self._hyperparameters.keys(), self._hyperparameters_range.keys()):
            if not hp_key == hpr_key:
                logging.error("\rINVALID HYPERPARAMETERS FOR TUNING")
                exit()

        hp_iterators = [iter(hpr) for hpr in self._hyperparameters_range.values()]
        hp_combinations = itertools.product(*hp_iterators)

        best_scores = None
        best_means = None
        best_solve = False
        best_hp = None

        for hp_combination in hp_combinations:
            self._hyperparameters = OrderedDict(zip(self._hyperparameters_range.keys(), hp_combination))

            scores, means, solve = self.train()

            if solve:
                best_scores = [] if best_scores is None else best_scores
                best_means = [] if best_means is None else best_means
                best_solve = True

                if len(scores) < len(best_scores):
                    best_scores = scores
                    best_means = means
                    best_hp = self._hyperparameters.copy()

        logging.info("TUNING FINISHED")
        logging.info("BEST RUN EPISODES: {}".format(len(best_scores)))
        logging.info("BEST RUN MEAN SCORE (OVER 100 EPISODES): {}".format(best_means[-1]))

        Utils.print_hyperparameters(best_hp)

        self._environment.close()

        return best_scores, best_means, best_solve

    def show(self) -> None:
        pass


if __name__ == "__main__":
    # ARG 1: OPERATION MODE
    mode_arg = sys.argv[1]
    CollaborativeCompetition().run(mode=mode_arg)
