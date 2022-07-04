#!/usr/bin/env python3

import numpy
import torch
import typing

from typing import Dict
from typing import List
from typing import Tuple

from src.networks.network import Network

# required for typehint Self
Self = typing.TypeVar("Self", bound="Actor")


class Actor(Network):

    def __init__(self,
                 seed: int,
                 state_size: int,
                 action_size: int,
                 layers: List[int],
                 activation_function: torch.nn.Module,
                 output_function: torch.nn.Module) -> None:
        super(Actor, self).__init__()
        self.seed_string = seed
        self.seed = torch.manual_seed(self.seed_string)

        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers
        self.activation_function_name = activation_function
        self.output_function_name = output_function

        self.activation_function = activation_function()
        self.output_function = output_function() if output_function is not None else None

        self.fc1 = torch.nn.Linear(state_size, layers[0])
        self.fc2 = torch.nn.Linear(layers[0], layers[1])
        self.fc3 = torch.nn.Linear(layers[1], action_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = state
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)

        if self.output_function is None:
            return x

        if self.output_function is not None:
            x = self.output_function(x)
            return x

    @staticmethod
    def hidden_init(layer: torch.nn.Linear) -> Tuple[float, float]:
        fan_in = layer.weight.data.size()[0]
        lim = 1. / numpy.sqrt(fan_in)
        return -lim, lim

    def to_checkpoint_dict(self) -> Dict:
        """
        function to export a network to the network parameters (as dict)
        :return: the network parameters (as dict)
        """
        return {"seed": self.seed_string,
                "input_size": self.state_size,
                "output_size": self.action_size,
                "hidden_layers": self.layers,
                "activation_function": self.activation_function_name,
                "output_function": self.output_function_name,
                "state_dict": self.state_dict()}

    @staticmethod
    def from_checkpoint_dict(checkpoint: Dict) -> Self:
        """
        function to import a network from network parameters (as dict)
        :param checkpoint: the network parameters (as dict)
        :return: the network
        """
        actor = Actor(seed=checkpoint["seed"],
                      state_size=checkpoint["input_size"],
                      action_size=checkpoint["output_size"],
                      layers=checkpoint["hidden_layers"],
                      activation_function=checkpoint["activation_function"],
                      output_function=checkpoint["output_function"])
        actor.load_state_dict(state_dict=checkpoint["state_dict"])
        return actor
