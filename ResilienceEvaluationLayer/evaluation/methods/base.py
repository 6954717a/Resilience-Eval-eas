#!/usr/bin/env python3

"""
Base Classes for Implementation Methods

Defines common interfaces and base classes for all implementation methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class MethodConfig:
    """Base configuration for methods."""
    enabled: bool = False
    name: str = ""
    version: str = "1.0"


class MethodBase(ABC):
    """Base class for all implementation methods."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.name = self.__class__.__name__

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the method."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the method state."""
        pass

    def is_enabled(self) -> bool:
        """Check if method is enabled."""
        return self.enabled


class MetricsProvider(ABC):
    """Interface for methods that provide metrics."""

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from this method."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics summary."""
        pass


class RecoveryMethod(MethodBase):
    """Base class for recovery-focused methods (CoPAL, Rebound)."""

    @abstractmethod
    def handle_failure(self, failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a failure event."""
        pass

    @abstractmethod
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery-specific metrics."""
        pass


class StabilityMethod(MethodBase):
    """Base class for stability-focused methods (SayCan)."""

    @abstractmethod
    def compute_stability_score(self) -> float:
        """Compute stability score."""
        pass

    @abstractmethod
    def get_stability_metrics(self) -> Dict[str, Any]:
        """Get stability-specific metrics."""
        pass


class ContinualLearningMethod(MethodBase):
    """Base class for continual learning methods (CLARE)."""

    @abstractmethod
    def adapt(self, task_info: Dict[str, Any]) -> None:
        """Adapt to new task."""
        pass

    @abstractmethod
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation metrics."""
        pass


__all__ = [
    "MethodConfig",
    "MethodBase",
    "MetricsProvider",
    "RecoveryMethod",
    "StabilityMethod",
    "ContinualLearningMethod",
]
