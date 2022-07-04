#!/usr/bin/env python3

import torch
import typing

from abc import ABC

# required for typehint Self
Self = typing.TypeVar("Self", bound="Network")


class Network(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super(Network, self).__init__()

    @staticmethod
    def hard_update(source_network: Self, target_network: Self) -> None:
        for target_parameter, source_parameter in zip(target_network.parameters(), source_network.parameters()):
            target_parameter.data.copy_(source_parameter.data)

    @staticmethod
    def soft_update(source_network: Self, target_network: Self, tau: float) -> None:
        for target_parameter, source_parameter in zip(target_network.parameters(), source_network.parameters()):
            target_parameter.data.copy_((1.0 - tau) * target_parameter.data + tau * source_parameter.data)
