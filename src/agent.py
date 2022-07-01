#!/usr/bin/env python3

import torch
import torch.optim as optimizer
import typing

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict
from typing import List

from src.networks.actor import Actor
from src.networks.critic import Critic
from src.networks.network import Network
from src.noise import Noise


# required for typehint Self
Self = typing.TypeVar("Self", bound="Agent")


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

        global_view_size = state_size * 2 + action_size * 2

        self._noise: Noise = Noise(action_size=action_size,
                                   scale_maximum=noise_maximum,
                                   scale_minimum=noise_minimum,
                                   scale_decay=noise_decay)

        assert(len(actor_layers) == 2)
        assert(len(critic_layers) == 2)

        self.target_actor: Network = Actor(seed=10,
                                           state_size=state_size,
                                           action_size=action_size,
                                           layers=actor_layers,
                                           activation_function=actor_activation_function,
                                           output_function=actor_output_function).to(device=self._device)
        self.actor: Network = Actor(seed=10,
                                    state_size=state_size,
                                    action_size=action_size,
                                    layers=actor_layers,
                                    activation_function=actor_activation_function,
                                    output_function=actor_output_function).to(device=self._device)
        self.target_critic: Network = Critic(seed=10,
                                             state_size=global_view_size,
                                             action_size=1,
                                             layers=critic_layers,
                                             activation_function=critic_activation_function,
                                             output_function=critic_output_function).to(device=self._device)
        self.critic: Network = Critic(seed=10,
                                      state_size=global_view_size,
                                      action_size=1,
                                      layers=critic_layers,
                                      activation_function=critic_activation_function,
                                      output_function=critic_output_function).to(device=self._device)

        self.actor_optimizer: Optimizer = actor_optimizer(params=self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer: Optimizer = critic_optimizer(params=self.critic.parameters(), lr=critic_learning_rate)

        Network.hard_update(source_network=self.actor, target_network=self.target_actor)
        Network.hard_update(source_network=self.critic, target_network=self.target_critic)

    def act(self, state: Tensor, add_noise: bool = True) -> Tensor:
        return self._apply_network(network=self.actor,
                                   state=state,
                                   add_noise=add_noise,
                                   output_min=-1.0,
                                   output_max=+1.0)

    def target_act(self, state: Tensor, add_noise: bool = True) -> Tensor:
        return self._apply_network(network=self.target_actor,
                                   state=state,
                                   add_noise=add_noise,
                                   output_min=-1.0,
                                   output_max=+1.0)

    def _apply_network(self,
                       network: Network,
                       state: Tensor,
                       add_noise: bool,
                       output_min: float,
                       output_max: float) -> Tensor:
        state: Tensor = state.to(device=self._device)

        network.eval()
        with torch.no_grad():
            action: Tensor = network(state)
        network.train()

        if add_noise:
            noise_tensor = self._noise.get_action_noise_scale().to(device=self._device)
            action = action + noise_tensor

        action = torch.clip(action, output_min, output_max)
        return action.to(device=self._device)

    def to_checkpoint_dict(self, filename: str = "checkpoint.pth") -> Dict:
        pass

    @staticmethod
    def from_checkpoint_dict(filename: str = "checkpoint.pth") -> Self:
        pass
