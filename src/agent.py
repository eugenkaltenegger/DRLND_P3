#!/usr/bin/env python3

import torch
import torch.optim as optimizer

from src.network import Network
from src.network_utils import NetworkUtils
from src.noise import Noise
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import List
from typing import Optional


class Agent:

    def __init__(self,
                 device: torch.device,
                 noise_maximum: float,
                 noise_minimum: float,
                 noise_decay: float,
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

        self._noise: Noise = Noise(action_size=action_size,
                                   scale_maximum=noise_maximum,
                                   scale_minimum=noise_minimum,
                                   scale_decay=noise_decay)

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

    def act(self, state: Tensor, noise: bool) -> Tensor:
        state = state.to(device=self._device)
        action: Optional[Tensor] = None

        if noise:
            noise_tensor = self._noise.get_action_noise().to(device=self._device)
            action = self.actor(state) + noise_tensor

        if not noise:
            action = self.actor(state)

        return action.to(device=self._device)

    def target_act(self, state: Tensor, noise: bool) -> Tensor:
        state = state.to(device=self._device)
        action: Optional[Tensor] = None

        if noise:
            noise_tensor = self._noise.get_action_noise().to(device=self._device)
            action = self.target_actor(state) + noise_tensor

        if not noise:
            action = self.target_actor(state)

        return action.to(device=self._device)
