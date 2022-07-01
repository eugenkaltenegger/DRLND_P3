#!/usr/bin/env python3

import torch

from collections import OrderedDict
from torch import Tensor
from typing import List

from src.agent import Agent
from src.networks.network import Network


class AgentGroup:

    def __init__(self,
                 device: torch.device,
                 agents: int,
                 state_size: int,
                 action_size: int,
                 hyperparameters: OrderedDict):
        self._device: torch.device = device
        self._state_size = state_size
        self._action_size = action_size
        self._agents_number = agents
        self._agents: List[Agent] = [self.create_agent(hyperparameters=hyperparameters) for _ in range(self._agents_number)]
        self._tau = hyperparameters["tau"]
        self._discount = hyperparameters["discount"]
        self._loss_function = torch.nn.MSELoss()

    def create_agent(self, hyperparameters: OrderedDict) -> Agent:
        return Agent(device=self._device,
                     noise_maximum=hyperparameters["noise_maximum"],
                     noise_minimum=hyperparameters["noise_minimum"],
                     noise_decay=hyperparameters["noise_decay"],
                     state_size=self._state_size,
                     action_size=self._action_size,
                     actor_layers=hyperparameters["actor_layers"],
                     actor_activation_function=hyperparameters["actor_activation_function"],
                     actor_output_function=hyperparameters["actor_output_function"],
                     actor_optimizer=hyperparameters["actor_optimizer"],
                     actor_learning_rate=hyperparameters["actor_optimizer_learning_rate"],
                     critic_layers=hyperparameters["critic_layers"],
                     critic_activation_function=hyperparameters["critic_activation_function"],
                     critic_output_function=hyperparameters["critic_output_function"],
                     critic_optimizer=hyperparameters["critic_optimizer"],
                     critic_learning_rate=hyperparameters["critic_optimizer_learning_rate"])

    def act(self, states: List[Tensor], add_noise: bool) -> List[Tensor]:
        return [agent.act(state=state, add_noise=add_noise) for agent, state in zip(self._agents, states)]

    def target_act(self, states: List[Tensor], add_noise: bool) -> List[Tensor]:
        return [agent.target_act(state=state, add_noise=add_noise) for agent, state in zip(self._agents, states)]

    def update(self, local_states, local_actions, local_rewards, local_dones, local_next_states):
        """update the critics and actors of all the agents """

        local_states = [local_state.detach() for local_state in local_states]
        local_actions = [local_action.detach() for local_action in local_actions]
        local_rewards = [local_reward.detach() for local_reward in local_rewards]
        local_dones = [local_done.detach() for local_done in local_dones]
        local_next_states = [local_next_state.detach() for local_next_state in local_next_states]

        local_dones = [torch.gt(local_done, 0).int().to(device=self._device) for local_done in local_dones]

        for own_index, agent in enumerate(self._agents):
            other_index = 1 if own_index == 0 else 0
            # ---------------------------- update critic ---------------------------- #
            global_state = torch.cat((local_states[own_index], local_states[other_index]), dim=1).to(device=self._device)
            global_action = torch.cat((local_actions[own_index], local_actions[other_index]), dim=1).to(device=self._device)
            global_next_state = torch.cat((local_next_states[own_index], local_next_states[other_index]), dim=1).to(device=self._device)
            global_reward = torch.cat((local_rewards[own_index], local_rewards[other_index]), dim=1).to(device=self._device)
            global_done = torch.cat((local_dones[own_index], local_dones[other_index]), dim=1).to(device=self._device)

            own_target_action_prediction = agent.target_actor(local_states[own_index])
            other_target_action_prediction = agent.target_actor(local_states[other_index])
            global_target_action_prediction = torch.cat((own_target_action_prediction, other_target_action_prediction), dim=1)
            global_target_action_prediction = global_target_action_prediction.to(device=self._device)

            q_targets_next = agent.target_critic(global_next_state, global_target_action_prediction)

            q_targets = local_rewards[own_index] + (self._discount * q_targets_next * (1 - local_dones[own_index]))
            # q_targets = global_reward + (gamma * q_targets_next * (1 - global_done))

            q_expected = agent.critic(global_state, global_action)

            critic_loss = self._loss_function(q_expected, q_targets)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            # ---------------------------- update actor ---------------------------- #
            own_action_prediction = agent.actor(local_states[own_index])
            other_action_prediction = agent.actor(local_states[other_index]).detach()
            global_action_prediction = torch.cat((own_action_prediction, other_action_prediction), dim=1)
            global_action_prediction = global_action_prediction.to(device=self._device)

            actor_loss = -agent.critic(global_state, global_action_prediction).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            # ----------------------- update target networks ----------------------- #
            Network.soft_update(source_network=agent.critic, target_network=agent.target_critic, tau=self._tau)
            Network.soft_update(source_network=agent.actor, target_network=agent.target_actor, tau=self._tau)
