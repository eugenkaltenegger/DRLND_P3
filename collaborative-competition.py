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

LOGGING_FREQUENCY = 100


class CollaborativeCompetition:

    def __init__(self) -> None:
        # device variable
        self.device: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # hyperparameter variables
        self.hp: OrderedDict = Hyperparameters().get_dict()
        self.hpr: OrderedDict = HyperparametersRange().get_dict()

        # environment variables
        self.environment: Optional[Environment] = None
        self.environment_graphics: Optional[bool] = None
        self.environment_training: Optional[bool] = None

        # agent group variables
        self._agent_group: Optional[AgentGroup] = None

    def enable_training(self) -> None:
        """
        function to enable training mode
            training mode has to be either enabled or disabled
        :return: None
        """
        self.environment_graphics = False
        self.environment_training = True

    def disable_training(self) -> None:
        """
        function to disable training mode
            training mode has to be either enabled or disabled
        :return: None
        """
        self.environment_graphics = True
        self.environment_training = False

    def reset_environment(self) -> List[Tensor]:
        if self.environment is None:
            self.environment = Environment(enable_graphics=self.environment_graphics)
            self.environment.reset(train_environment=self.environment_training)

        if self.environment is not None:
            self.environment.reset(train_environment=self.environment_training)

        return self.environment.states()

    def reset_agent_group(self, from_checkpoint: bool = False) -> None:
        if self.environment is None:
            logging.error(
                "\rENVIRONMENT IS REQUIRED TO SETUP AGENT GROUP (PARAMETERS FROM THE ENVIRONMENT ARE REQUIRED)")

        if not from_checkpoint:
            self._agent_group = AgentGroup(device=self.device,
                                           agents=self.environment.number_of_agents(),
                                           state_size=self.environment.state_size(),
                                           action_size=self.environment.action_size(),
                                           hyperparameters=self.hp)
        if from_checkpoint:
            self._agent_group = Utils.load_agent_group(device=self.device, filename="collaborative-competition.pth")

    def run(self, mode) -> None:
        if mode not in ["train", "tune", "show"]:
            logging.error("\rINVALID OPERATION MODE {} (ALLOWED: train, tune and show)".format(mode))

        if mode == "train":
            scores, means, _ = self.train()
            Utils.plot_scores(scores=scores, means=means, show=True, filename="collaborative-competition.png")

        if mode == "tune":
            scores, means, _ = self.tune()
            Utils.plot_scores(scores=scores, means=means, show=True, filename="tuning.png")

        if mode == "show":
            self.show()

    def train(self) -> Tuple[List[float], List[float], bool]:
        self.enable_training()
        self.reset_environment()
        self.reset_agent_group(from_checkpoint=False)

        logging.info("\r-------------------------- HYPERPARAMETERS ---------------------------")
        Utils.print_hyperparameters(self.hp)
        logging.info("\r----------------------------------------------------------------------")

        best_score = None
        best_episode = None
        best_score_occurrence = 0

        buffer = Buffer(self.hp["buffer_size"])

        scores = []
        means = []
        steps = []
        for episode in range(1, self.hp["episodes"] + 1):

            local_states: List[Tensor] = self.reset_environment()
            rewards = []

            for step in range(1, self.hp["steps"] + 1):
                local_states = [local_state.to(device=self.device) for local_state in local_states]
                local_actions = self._agent_group.act(states=local_states, add_noise=True)
                local_actions = [local_action[0] for local_action in local_actions]
                local_rewards, local_dones, local_next_states = self.environment.step(actions=local_actions)

                local_actions = [local_action.to(device=self.device) for local_action in local_actions]
                local_rewards = [local_reward.to(device=self.device) for local_reward in local_rewards]
                local_dones = [local_done.to(device=self.device) for local_done in local_dones]
                local_next_states = [local_next_state.to(device=self.device) for local_next_state in local_next_states]

                weight = max(torch.cat(local_rewards).tolist())
                buffer.push(transition=(local_states, local_actions, local_rewards, local_dones, local_next_states),
                            weight=weight)
                rewards.append(torch.cat(local_rewards).tolist())

                local_states = local_next_states

                if any(local_dones):
                    steps.append(step)
                    break

            if len(buffer) > self.hp["batch_size"] and episode % self.hp["learning_frequency"] == 0:
                for _ in range(self.hp["buffer_iterations"]):
                    batch = buffer.batch(self.hp["batch_size"])
                    for _ in range(self.hp["sample_iterations"]):
                        self._agent_group.update(*batch)

            score = max([sum([round(x[index], 4) for x in rewards]) for index in range(len(self._agent_group))])
            scores.append(score)

            mean = 0 if episode < 100 else numpy.array(scores[-100:]).mean()
            means.append(mean)

            if best_score is None or score > best_score:
                best_score = score

            if episode % LOGGING_FREQUENCY == 0:
                not_zero = sum([1 if score != 0 else 0 for score in scores[-LOGGING_FREQUENCY:]])
                not_zero_percentage = sum(steps[-LOGGING_FREQUENCY:]) / LOGGING_FREQUENCY * 100
                steps_done = sum(steps[-LOGGING_FREQUENCY:])
                logging.info("\repisode {:6d}: steps: {:6d}, best score: {:6.2f}, not zero: {:6.2f}% - MEAN: {:8.4f}".
                             format(episode, steps_done, best_score, not_zero, mean))
                best_score = None

            if mean >= 0.5:
                logging.info("\rENVIRONMENT SOLVED AFTER {} EPISODES!".format(episode))
                Utils.save_agent_group(self._agent_group, filename="collaborative-competition.pth")
                return scores, means, True

        return scores, means, False

    def tune(self) -> Tuple[List[float], List[float], bool]:
        for hp_key, hpr_key in zip(self.hp.keys(), self.hpr.keys()):
            if not hp_key == hpr_key:
                logging.error("\rINVALID HYPERPARAMETERS FOR TUNING")
                exit()

        hp_iterators = [iter(hpr) for hpr in self.hpr.values()]
        hp_combinations = itertools.product(*hp_iterators)

        best_scores = None
        best_means = None
        best_solve = False
        best_hp = None

        for hp_combination in hp_combinations:
            self.hp = OrderedDict(zip(self.hpr.keys(), hp_combination))

            scores, means, solve = self.train()

            if solve:
                best_scores = [] if best_scores is None else best_scores
                best_means = [] if best_means is None else best_means
                best_solve = True

                if len(scores) < len(best_scores):
                    best_scores = scores
                    best_means = means
                    best_hp = self.hp.copy()

        logging.info("\r-------------------------- TUNING FINISHED ---------------------------")
        logging.info("\rBEST RUN EPISODES: {}".format(len(best_scores)))
        logging.info("\rBEST RUN MEAN SCORE (OVER 100 EPISODES): {}".format(best_means[-1]))

        Utils.print_hyperparameters(best_hp)

        self.environment.close()

        return best_scores, best_means, best_solve

    def show(self) -> None:
        self.disable_training()
        self.reset_environment()
        self.reset_agent_group(from_checkpoint=True)

        local_states: List[Tensor] = self.reset_environment()

        for _ in range(8192):
            local_states = [local_state.to(device=self.device) for local_state in local_states]
            local_actions = self._agent_group.target_act(states=local_states, add_noise=False)
            local_actions = [local_action[0] for local_action in local_actions]
            _, local_dones, local_states = self.environment.step(actions=local_actions)

            if any(local_dones):
                local_states = self.reset_environment()


if __name__ == "__main__":
    # ARG 1: OPERATION MODE
    mode_arg = sys.argv[1]
    CollaborativeCompetition().run(mode=mode_arg)
