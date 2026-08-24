"""
Utility functions for A2C Critic

Includes GAE calculator and buffer management.
"""

from .gae_calculator import compute_gae

__all__ = [
    'compute_gae',
]
