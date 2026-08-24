"""
Value Network for A2C Critic

Implements V(s) - State Value Function using a Multi-Layer Perceptron (MLP).
"""

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ValueNetwork(nn.Module):
    """
    Value Network that estimates V(s) - the expected return from state s.

    Architecture:
        state_vector → MLP layers → V(s) (scalar)

    Uses:
    - ReLU activations
    - LayerNorm for stability
    - Dropout for regularization
    """

    def __init__(
        self,
        state_dim: int = 128,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize Value Network.

        Args:
            state_dim: Dimension of input state vector
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dims = list(hidden_dims or [256, 128, 64])
        self.device = device

        # Build MLP layers
        layers = []
        input_dim = state_dim

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        # Output layer: single scalar V(s)
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)
        self.network.to(device)

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Initialized ValueNetwork: state_dim={state_dim}, "
                   f"hidden_dims={self.hidden_dims}, device={device}")

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate V(s).

        Args:
            state_vector: State vector(s) of shape [state_dim] or [batch, state_dim]

        Returns:
            value: V(s) estimate(s) of shape [] (scalar) or [batch]
        """
        value, _ = self.forward_with_features(state_vector)
        return value

    def extract_features(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Return the penultimate representation before the scalar value head.

        The critic's beta-stability trace uses this representation to estimate
        trajectory oscillation alongside V(s), TD residuals, and GAE. The method
        preserves the caller's shape convention: a single input returns
        ``[feature_dim]`` and a batch returns ``[batch, feature_dim]``.
        """
        single_input = state_vector.dim() == 1
        hidden = state_vector.unsqueeze(0) if single_input else state_vector

        if len(self.network) == 1:
            features = hidden
        else:
            features = self.network[:-1](hidden)

        return features.squeeze(0) if single_input else features

    def forward_with_features(
        self, state_vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns the penultimate hidden representation.

        Returns:
            A tuple ``(value, features)``. ``value`` has shape ``[]`` for a
            single state and ``[batch]`` for batched states; ``features`` follows
            the same single/batch convention as :meth:`extract_features`.
        """
        single_input = state_vector.dim() == 1
        features = self.extract_features(state_vector)

        if single_input:
            value = self.network[-1](features.unsqueeze(0)).squeeze(-1).squeeze(0)
        else:
            value = self.network[-1](features).squeeze(-1)

        return value, features

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.
        """
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

