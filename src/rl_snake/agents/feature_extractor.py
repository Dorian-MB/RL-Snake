"""Feature extractors and model evaluation utilities for RL agents."""

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LinearQNet(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Snake environment.

    This network processes the observation space and extracts features
    for the RL agent. It uses a simple linear architecture with ReLU
    activations.
    """

    def __init__(self, observation_space, features_dim=64, n_layers=3):
        """
        Initialize the feature extractor.

        Args:
            observation_space: The observation space of the environment
            features_dim: Dimension of each hidden layer (default: 64 for Snake)
            n_layers: Number of hidden layers (default: 3 for better abstraction)
        """
        super(LinearQNet, self).__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()

        # Calculate input dimension by flattening a sample observation
        with torch.no_grad():
            n_flatten = self.flatten(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        layers = []
        # Build dynamic architecture: n_layers of features_dim neurons each
        # First layer: input -> features_dim
        # Hidden layers: features_dim -> features_dim
        for i in range(n_layers):
            in_dim = features_dim if i > 0 else n_flatten
            layers.extend(
                [
                    nn.Linear(in_dim, features_dim),
                    #    nn.LayerNorm(features_dim),
                    nn.ReLU(),
                    #    nn.Dropout(0.1),
                ]
            )

        self.linear = nn.Sequential(*layers)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input tensor

        Returns:
            Extracted features
        """
        flat = self.flatten(X)
        out = self.linear(flat)
        return out
