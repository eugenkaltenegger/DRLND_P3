#!/usr/bin/env python3

import numpy
import torch
import typing

from collections import OrderedDict
from itertools import cycle
from torch import Tensor
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from typing import Dict
from typing import List

# required for typehint Self
Self = typing.TypeVar("Self", bound="Network")


class Network(Module):
    """
    class to hold a model for an agent
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 layers: List[int],
                 activation_function: Module,
                 output_function: Module,
                 seed: int = 0):
        """
        initializer for model class
            this constructor requires the setup function afterwards
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self._state_size = state_size
        self._action_size = action_size
        self._layers = layers
        self._activation_function_name = activation_function
        self._activation_function = activation_function()

        self._output_function_name = output_function if output_function is not None else None
        self._output_function = output_function() if output_function is not None else None

        self._model_sequential = self._create_model_sequential()
        self._reset_parameters()

    def forward(self, state: Tensor) -> Tensor:
        """
        function to process a state
        :param state: state to process
        :return: model output
        """
        if self._output_function is None:
            return self._model_sequential(state)
        if self._output_function is not None:
            return self._output_function(self._model_sequential(state))

    def to_checkpoint_dict(self) -> Dict:
        """
        function to export a network to the network parameters (as dict)
        :return: the network parameters (as dict)
        """
        return {"input_size": self._state_size,
                "output_size": self._action_size,
                "hidden_layers": self._layers,
                "activation_function": self._activation_function_name,
                "output_function": self._output_function_name,
                "state_dict": self.state_dict()}

    @staticmethod
    def from_checkpoint_dict(checkpoint: Dict) -> Self:
        network = Network(state_size=checkpoint["input_size"],
                          action_size=checkpoint["output_size"],
                          layers=checkpoint["hidden_layers"],
                          activation_function=checkpoint["activation_function"],
                          output_function=checkpoint["output_function"])
        network.load_state_dict(state_dict=checkpoint["state_dict"])
        return network

    @staticmethod
    def hidden_init(layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / numpy.sqrt(fan_in)
        return -lim, lim

    def _reset_parameters(self):
        layers = []
        for layer in self._model_sequential:
            if type(layer) == Linear:
                layers.append(layer)

        for layer in layers[:-1]:
            layer.weight.data.uniform_(*self.hidden_init(layer))

        layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def _create_model_sequential(self) -> Sequential:
        input_size = self._state_size
        output_size = self._action_size

        # nl stands for network layer
        nl_names = ["fc{}".format(counter) for counter in range(0, len(self._layers) + 1)]
        nl_sizes = []
        nl_sizes += [(input_size, self._layers[0])]
        nl_sizes += [(self._layers[i - 1], self._layers[i]) for i in range(1, len(self._layers))]
        nl_sizes += [(self._layers[-1], output_size)]
        nl_objects = [Linear(layer_size[0], layer_size[1]) for layer_size in nl_sizes]

        layers_dict: Dict[str, Module] = OrderedDict(zip(nl_names, nl_objects))

        # af stands for activation function
        af_names = ["af{}".format(counter) for counter in range(0, len(self._layers))]
        af_objects = [self._activation_function for _ in range(0, len(self._layers))]
        activation_function_dict: Dict[str, Module] = OrderedDict(zip(af_names, af_objects))

        key_iterators = [iter(layers_dict.keys()), iter(activation_function_dict.keys())]
        values_iterators = [iter(layers_dict.values()), iter(activation_function_dict.values())]

        key_list = list(iterator.__next__() for iterator in cycle(key_iterators))
        value_list = list(iterator.__next__() for iterator in cycle(values_iterators))

        model_dict: OrderedDict[str, Module] = OrderedDict(zip(key_list, value_list))
        return Sequential(model_dict)
