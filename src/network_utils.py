#!/usr/bin/env python3

import logging

from network import Network


class NetworkUtils:

    @staticmethod
    def hard_update(target_network: Network, source_network: Network) -> None:
        for target_parameter, source_parameter in zip(target_network.parameters(), source_network.parameters()):
            target_parameter.data.copy_(source_parameter.data)

    @staticmethod
    def soft_update(target_network: Network, source_network: Network, tau: float) -> None:
        if not 0 < tau < 1:
            logging.warning("\rInvalid parameter TAU")

        for target_parameter, source_parameter in zip(target_network.parameters(), source_network.parameters()):
            target_parameter.data.copy_(target_parameter.data * (1.0 - tau) + source_parameter.data * tau)
