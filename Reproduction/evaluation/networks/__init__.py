"""
Neural Networks for A2C Critic

This module contains neural network implementations for the A2C Critic:
- StateEncoder: Encodes world states into fixed-dimensional vectors
- ValueNetwork: Estimates state values V(s)
"""

from .state_encoder import StateEncoder
from .value_network import ValueNetwork

__all__ = [
    'StateEncoder',
    'ValueNetwork',
]
