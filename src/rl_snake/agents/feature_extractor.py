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
    
    def __init__(self, observation_space, features_dim=32):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: The observation space of the environment
            features_dim: Dimension of the output features
        """
        super(LinearQNet, self).__init__(observation_space, features_dim)
        self.flatten = nn.Flatten()
        
        # Calculate input dimension by flattening a sample observation
        with torch.no_grad():
            n_flatten = self.flatten(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        # Define the neural network architecture
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

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

    