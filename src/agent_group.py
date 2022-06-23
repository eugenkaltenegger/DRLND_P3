#!/usr/bin/env python3

import torch

from agent import Agent

from collections import OrderedDict

from network import Network

from network_utils import NetworkUtils

from torch import Tensor

from typing import List
from typing import Optional


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
        self._agents: List[Agent] = [self.create_agent(hyperparameters=hyperparameters) for _ in range(agents)]
        self._tau = hyperparameters["tau"]
        self._discount = hyperparameters["discount"]

    def create_agent(self, hyperparameters: OrderedDict) -> Agent:
        return Agent(device=self._device,
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

    def act(self, states: List[Tensor], noise: Optional[float] = None) -> Tensor:
        # TODO:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        actions = [agent.act(state=state, noise=noise).detach().numpy() for agent, state in zip(self._agents, states)]
        return torch.tensor(actions, dtype=torch.float).to(device=self._device)

    def target_act(self, states: List[Tensor], noise: Optional[float] = None) -> Tensor:
        # TODO:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        actions = [agent.target_act(state=state, noise=noise).detach().numpy() for agent, state in zip(self._agents, states)]
        return torch.tensor(actions, dtype=torch.float).to(device=self._device)

    @staticmethod
    def transpose_to_tensor(input_list):
        return list(map(lambda x: torch.tensor(x, dtype=torch.float), zip(*input_list)))

    def update(self, state, action, reward, done, next_state):
        """update the critics and actors of all the agents """

        # state_all: Tensor(2,24)
        # next_state_all: Tensor(2,24)
        # state = state[0]
        # state_full(Tensor: 48,) = torch.cat([s for s in state])
        # action
        # action_full(Tensor: 4,) = Tensor(2,2)
        # target_action_full(Tensor: 4,) = agent_group.target_act([s for s in state_all])
        # reward
        # next_state = next_state_all[0]
        # next_state_full(Tensor: 48,) = torch.cat([s for s in next_state])

        state_full = torch.cat([s for s in state])
        action_full = torch.cat([a for a in action])
        next_state_full = torch.cat([s for s in next_state])
        target_action_full = torch.cat([a for a in self.target_act([s for s in state])])

        for agent_index, agent in enumerate(self._agents):
            agent_state = state[agent_index]
            agent_action = action[agent_index]
            agent_reward = reward[agent_index]

            # --------------------------------------------- optimize critic --------------------------------------------
            agent.critic_optimizer.zero_grad()

            target_critic_input = torch.cat((next_state_full, target_action_full))
            critic_input = torch.cat((state_full, action_full))

            with torch.no_grad():
                q_next = agent.target_critic(target_critic_input)

            q = agent.critic(critic_input)

            # TODO check if view is necessary
            if done[agent_index]:
                y = reward[agent_index].view(-1, 1)
            else:
                y = reward[agent_index].view(-1, 1) + self._discount * q_next
            y = y.detach()

            huber_loss = torch.nn.SmoothL1Loss()
            critic_loss = huber_loss(q, y)
            critic_loss.backward()

            agent.critic_optimizer.step()
            # --------------------------------------------- optimize actor ---------------------------------------------
            agent.critic_optimizer.step()

            agent.actor_optimizer.zero_grad()

            q_input = [self._agents[i].actor(s) if i == agent_index else self._agents[i].actor(s).detach() for i, s in enumerate(state)]
            q_input = torch.cat((q_input[0], q_input[1]))
            q_input = torch.cat((state_full, q_input))

            actor_loss = -agent.critic(q_input).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()
            # ----------------------------------------------------------------------------------------------------------

    def update_target(self) -> None:
        for agent in self._agents:
            NetworkUtils.soft_update(target_network=agent.target_actor, source_network=agent.actor, tau=self._tau)
            NetworkUtils.soft_update(target_network=agent.target_critic, source_network=agent.critic, tau=self._tau)
