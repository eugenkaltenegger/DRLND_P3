#!/usr/bin/env python3
import numpy
import torch

from agent import Agent

from collections import OrderedDict

from network import Network

from network_utils import NetworkUtils

from torch import Tensor

from typing import List
from typing import Optional

from utils import Utils


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
        # TODO:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        actions = [agent.act(state=state, noise=noise).detach().numpy() for agent, state in zip(self._agents, states)]
        return [torch.tensor(action, dtype=torch.float).to(device=self._device) for action in actions]

    def target_act(self, states: List[Tensor], noise: bool) -> Tensor:
        # TODO:
        # Creating a tensor from a list of numpy.ndarrays is extremely slow.
        # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        actions = [agent.target_act(state=state, noise=noise).detach().numpy() for agent, state in zip(self._agents, states)]
        actions = numpy.array(actions)
        return torch.tensor(actions, dtype=torch.float).to(device=self._device)

    @staticmethod
    def transpose_to_tensor(input_list):
        return list(map(lambda x: torch.tensor(x, dtype=torch.float), zip(*input_list)))

    def update(self, global_state, global_action, global_reward, global_done, global_next_state):
        """update the critics and actors of all the agents """

        local_state = Utils.global_to_local(global_state, agents=len(self._agents))
        local_action = Utils.global_to_local(global_action, agents=len(self._agents))
        local_reward = Utils.global_to_local(global_reward, agents=len(self._agents))
        local_done = Utils.global_to_local(global_done, agents=len(self._agents))
        local_next_state = Utils.global_to_local(global_next_state, agents=len(self._agents))
        local_target_actions = [agent_action for agent_action in self.target_act(local_state, noise=False)]
        local_numeric_done = [torch.tensor(agent_done, dtype=torch.float) for agent_done in local_done]

        global_target_actions = Utils.local_to_global(local_target_actions, dim=1)
        global_numeric_done = torch.tensor(global_done, dtype=torch.float)

        for agent_index, agent in enumerate(self._agents):
            # --------------------------------------------- optimize critic --------------------------------------------
            agent.critic_optimizer.zero_grad()

            target_critic_input = torch.cat((global_state, global_target_actions), dim=1)
            critic_input = torch.cat((global_state, global_action), dim=1)

            with torch.no_grad():
                q_next = agent.target_critic(target_critic_input)

            q = agent.critic(critic_input)

            # TODO check if view is necessary

            a = local_reward[agent_index]
            b = self._discount * q_next
            c = (1 - local_numeric_done[agent_index])
            y = local_reward[agent_index].view(-1, 1) + self._discount * q_next * (1 - local_numeric_done[agent_index].view(-1, 1))

            y = y.detach()

            huber_loss = torch.nn.SmoothL1Loss()
            critic_loss = huber_loss(q, y)
            critic_loss.backward()

            agent.critic_optimizer.step()
            # --------------------------------------------- optimize actor ---------------------------------------------
            agent.actor_optimizer.zero_grad()

            q_input = [self._agents[i].actor(s) if i == agent_index else self._agents[i].actor(s).detach() for i, s in enumerate(local_state)]
            q_input = torch.cat((q_input[0], q_input[1]), dim=1)
            q_input = torch.cat((global_state, q_input), dim=1)

            actor_loss = -agent.critic(q_input).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()
            # ----------------------------------------------------------------------------------------------------------

            self.update_target()

    def update_target(self) -> None:
        for agent in self._agents:
            NetworkUtils.soft_update(target_network=agent.target_actor, source_network=agent.actor, tau=self._tau)
            NetworkUtils.soft_update(target_network=agent.target_critic, source_network=agent.critic, tau=self._tau)
