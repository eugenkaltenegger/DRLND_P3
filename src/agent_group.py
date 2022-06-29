#!/usr/bin/env python3
import torch

from collections import OrderedDict
from torch import Tensor
from typing import List

from src.agent import Agent
from src.network_utils import NetworkUtils
from src.utils import Utils

DEBUGGING = False


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

    def update(self, global_state, global_action, global_reward, global_done, global_next_state):
        """update the critics and actors of all the agents """

        local_states = Utils.global_to_local(global_state, agents=len(self._agents))
        local_rewards = Utils.global_to_local(global_reward, agents=len(self._agents))
        local_dones = Utils.global_to_local(global_done, agents=len(self._agents))
        local_next_states = Utils.global_to_local(global_next_state, agents=len(self._agents))
        local_numeric_dones = [torch.gt(agent_done, 0).int() for agent_done in local_dones]

        for agent_index, agent in enumerate(self._agents):
            # --------------------------------------------- optimize critic --------------------------------------------
            local_next_target_actions = [agent.target_act(local_next_state, add_noise=False) for local_next_state in local_next_states]
            global_next_target_action = Utils.local_to_global(local_next_target_actions, dim=1)

            target_critic_input = torch.cat((global_next_state, global_next_target_action), dim=1)

            # with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

            # y = global_reward + self._discount * q_next * (1 - global_numeric_dones).to(self._device)
            y = local_rewards[agent_index] + self._discount * q_next * (1 - local_numeric_dones[agent_index]).to(self._device)

            critic_input = torch.cat((global_state, global_action.detach()), dim=1).to(self._device)
            q = agent.critic(critic_input)

            loss_function = torch.nn.MSELoss()
            critic_loss = loss_function(q, y)

            if DEBUGGING:
                print("actor index: {} - critic loss: {}".format(agent_index, critic_loss))

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            # --------------------------------------------- optimize actor ---------------------------------------------
            q_input = [agent.act(local_state, add_noise=False) for local_state in local_states]
            q_input = Utils.local_to_global(q_input, dim=1)
            critic_input = torch.cat((global_state, q_input), dim=1)
            actor_loss = -agent.critic(critic_input).mean()

            if DEBUGGING:
                print("actor index: {} - actor loss: {}".format(agent_index, actor_loss))

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            # ----------------------------------------------------------------------------------------------------------

        self.update_target()

    def update_target(self) -> None:
        for agent in self._agents:
            NetworkUtils.soft_update(target_network=agent.target_actor, source_network=agent.actor, tau=self._tau)
            NetworkUtils.soft_update(target_network=agent.target_critic, source_network=agent.critic, tau=self._tau)
