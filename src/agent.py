#!/usr/bin/env python3

import torch
import torch.optim as optimizer
import typing

from torch import device
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing import Dict
from typing import List
from typing import Optional

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
                 actor_layers: List[int] = None,
                 actor_activation_function: Module = None,
                 actor_output_function: Module = None,
                 actor_optimizer: optimizer = None,
                 actor_learning_rate: float = None,
                 critic_layers: List[int] = None,
                 critic_activation_function: Module = None,
                 critic_output_function: Module = None,
                 critic_optimizer: optimizer = None,
                 critic_learning_rate: float = None):

        self._device: torch.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.noise_maximum = noise_maximum
        self.noise_minimum = noise_minimum
        self.noise_decay = noise_decay

        self._noise: Noise = Noise(action_size=action_size,
                                   scale_maximum=noise_maximum,
                                   scale_minimum=noise_minimum,
                                   scale_decay=noise_decay)

        # agent created from hyperparameters (output functions may be None)
        if actor_layers is not None and \
                actor_activation_function is not None and \
                actor_optimizer is not None and \
                actor_learning_rate is not None and \
                critic_layers is not None and \
                critic_activation_function is not None and \
                critic_optimizer is not None and \
                critic_learning_rate is not None:

            assert (len(actor_layers) == 2)
            assert (len(critic_layers) == 2)

            global_view_size = state_size * 2 + action_size * 2

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
            self.critic_optimizer: Optimizer = critic_optimizer(params=self.critic.parameters(),
                                                                lr=critic_learning_rate)

            Network.hard_update(source_network=self.actor, target_network=self.target_actor)
            Network.hard_update(source_network=self.critic, target_network=self.target_critic)

        # agent created from checkpoint
        else:
            self.target_actor: Optional[Network] = None
            self.actor: Optional[Network] = None
            self.target_critic: Optional[Network] = None
            self.critic: Optional[Network] = None

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

        action = torch.clamp(action, output_min, output_max)
        return action.to(device=self._device)

    def to_checkpoint_dict(self) -> Dict:
        return {"state_size": self.state_size,
                "action_size": self.action_size,
                "noise_maximum": self.noise_maximum,
                "noise_minimum": self.noise_minimum,
                "noise_decay": self.noise_decay,
                "target_actor": self.target_actor.to_checkpoint_dict(),
                "actor": self.actor.to_checkpoint_dict(),
                "target_critic": self.target_critic.to_checkpoint_dict(),
                "critic": self.critic.to_checkpoint_dict()}

    @staticmethod
    def from_checkpoint_dict(checkpoint: Dict, device: device) -> Self:
        agent = Agent(device=device,
                      state_size=checkpoint["state_size"],
                      action_size=checkpoint["action_size"],
                      noise_maximum=checkpoint["noise_maximum"],
                      noise_minimum=checkpoint["noise_minimum"],
                      noise_decay=checkpoint["noise_decay"])

        agent.target_actor = Actor.from_checkpoint_dict(checkpoint=checkpoint["target_actor"]).to(device=device)
        agent.actor = Actor.from_checkpoint_dict(checkpoint=checkpoint["actor"]).to(device=device)
        agent.target_critic = Critic.from_checkpoint_dict(checkpoint=checkpoint["target_critic"]).to(device=device)
        agent.critic = Critic.from_checkpoint_dict(checkpoint=checkpoint["critic"]).to(device=device)

        return agent
