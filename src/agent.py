#!/usr/bin/env python3

import torch
import torch.optim as optimizer

from network import Network
from network_utils import NetworkUtils
from noise import Noise
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import List
from typing import Optional


class Agent:

    def __init__(self,
                 device: torch.device,
                 state_size: int,
                 action_size: int,
                 actor_layers: List[int],
                 actor_activation_function: Module,
                 actor_output_function: Module,
                 actor_optimizer: optimizer,
                 actor_learning_rate: float,
                 critic_layers: List[int],
                 critic_activation_function: Module,
                 critic_output_function: Module,
                 critic_optimizer: optimizer,
                 critic_learning_rate: float):

        self._device: torch.device = device

        self.target_actor: Network = Network(state_size=state_size,
                                             action_size=action_size,
                                             layers=actor_layers,
                                             activation_function=actor_activation_function,
                                             output_function=actor_output_function).to(device=self._device)

        self.target_critic: Network = Network(state_size=state_size * 2 + action_size * 2,
                                              action_size=1,
                                              layers=critic_layers,
                                              activation_function=critic_activation_function,
                                              output_function=critic_output_function).to(device=self._device)

        self.actor: Network = Network(state_size=state_size,
                                      action_size=action_size,
                                      layers=actor_layers,
                                      activation_function=actor_activation_function,
                                      output_function=actor_output_function).to(device=self._device)

        self.critic: Network = Network(state_size=state_size * 2 + action_size * 2,
                                       action_size=1,
                                       layers=critic_layers,
                                       activation_function=critic_activation_function,
                                       output_function=critic_output_function).to(device=self._device)

        self.actor_optimizer: Optimizer = actor_optimizer(params=self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer: Optimizer = critic_optimizer(params=self.critic.parameters(), lr=critic_learning_rate)

        NetworkUtils.hard_update(target_network=self.target_actor, source_network=self.actor)
        NetworkUtils.hard_update(target_network=self.target_critic, source_network=self.critic)

        self._noise: Noise = Noise(action_size)

    def act(self, state: Tensor, noise: Optional[float]) -> Tensor:
        action = self.actor(state) if noise is None else self.actor(state) + noise * self._noise.noise()
        return action.to(device=self._device)

    def target_act(self, state: Tensor, noise: Optional[float]) -> Tensor:
        action = self.target_actor(state) if noise is None else self.actor(state) + noise * self._noise.noise()
        return action.to(device=self._device)
