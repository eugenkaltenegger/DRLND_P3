#!/usr/bin/env python3
import numpy
import torch
import torch.nn.functional as functional

from src.agent import Agent

from collections import OrderedDict

from src.network import Network

from src.network_utils import NetworkUtils

from torch import Tensor

from typing import List
from typing import Optional

from src.utils import Utils


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

    def actors(self) -> List[Network]:
        return [agent.actor() for agent in self._agents]

    def target_actors(self) -> List[Network]:
        return [agent.target_actor() for agent in self._agents]

    def act(self, states: List[Tensor], noise: bool) -> [Tensor]:
        actions = [agent.act(state=state, noise=noise).detach().cpu().numpy() for agent, state in zip(self._agents, states)]
        actions = numpy.array(actions)
        return [torch.tensor(action, dtype=torch.float).to(device=self._device) for action in actions]

    def target_act(self, states: List[Tensor], noise: bool) -> Tensor:
        actions = [agent.target_act(state=state, noise=noise).detach().cpu().numpy() for agent, state in zip(self._agents, states)]
        actions = numpy.array(actions)
        return torch.tensor(actions, dtype=torch.float).to(device=self._device)

    def update(self, global_state, global_action, global_reward, global_done, global_next_state):
        """update the critics and actors of all the agents """

        local_state = Utils.global_to_local(global_state, agents=len(self._agents))
        local_action = Utils.global_to_local(global_action, agents=len(self._agents))
        local_reward = Utils.global_to_local(global_reward, agents=len(self._agents))
        local_done = Utils.global_to_local(global_done, agents=len(self._agents))
        local_next_state = Utils.global_to_local(global_next_state, agents=len(self._agents))
        local_next_action = [agent_action.detach() for agent_action in self.act(local_state, noise=False)]
        local_target_action = [agent_action for agent_action in self.target_act(local_state, noise=False)]
        local_target_next_action = [agent_action for agent_action in self.target_act(local_state, noise=False)]
        local_numeric_done = [torch.gt(agent_done, 0).int() for agent_done in local_done]

        global_target_action = Utils.local_to_global(local_target_action, dim=1)
        global_target_next_action = Utils.local_to_global(local_target_next_action, dim=1)
        global_numeric_done = torch.gt(global_done, 0).int()

        for agent_index, agent in enumerate(self._agents):
            # --------------------------------------------- optimize critic --------------------------------------------
            target_critic_input = torch.cat((global_next_state, global_target_next_action), dim=1)
            q_target_next = agent.target_critic(target_critic_input)
            q_target = local_reward[agent_index] + self._discount * q_target_next * (1 - local_numeric_done[agent_index])

            critic_input = torch.cat((global_state, global_action), dim=1)
            q_expected = agent.critic(critic_input)

            critic_loss = functional.mse_loss(q_expected, q_target)

            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            # --------------------------------------------- optimize actor ---------------------------------------------
            local_next_action_copy = local_next_action.copy()
            local_next_action_copy[agent_index] = agent.actor(local_state[agent_index])
            global_next_action = Utils.local_to_global(local_next_action_copy, dim=1)
            critic_input = torch.cat((global_state, global_next_action), dim=1)

            actor_loss = -agent.critic(critic_input).mean()

            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            agent.actor_optimizer.step()
            # ----------------------------------------------------------------------------------------------------------

            self.update_target()

    def update_target(self) -> None:
        for agent in self._agents:
            NetworkUtils.soft_update(target_network=agent.target_actor, source_network=agent.actor, tau=self._tau)
            NetworkUtils.soft_update(target_network=agent.target_critic, source_network=agent.critic, tau=self._tau)
