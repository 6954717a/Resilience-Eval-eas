"""
LLM-based Evaluation for A2C Critic

This module provides LLM-enhanced evaluation capabilities:
- LLMRewardShaper: Uses LLM to provide semantic reward signals
- LLMOfflineAnalyzer: Analyzes trajectories offline for debugging and improvement
"""

from .reward_shaper import LLMRewardShaper
from .offline_analyzer import LLMOfflineAnalyzer

__all__ = [
    'LLMRewardShaper',
    'LLMOfflineAnalyzer',
]
