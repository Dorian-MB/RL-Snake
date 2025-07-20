"""Feature extractors and model evaluation utilities for RL agents."""

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

    
def evaluate_model(model, eval_env, num_episodes=10):
    """
    Evaluate a trained model's performance.
    
    Args:
        model: Trained RL model
        eval_env: Environment for evaluation
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward across episodes
    """
    all_rewards = []
    
    for episode in range(num_episodes):
        obs = eval_env.reset()
        # Handle different environment return formats
        if isinstance(obs, tuple):
            obs = obs[0]
        
        terminated = False
        total_rewards = 0
        
        # Limit steps to prevent infinite loops
        for _ in range(1000):
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step in environment
            step_result = eval_env.step(action)
            
            # Handle different return formats (gym vs gymnasium)
            if len(step_result) == 5:  # Gymnasium format
                obs, reward, terminated, truncated, info = step_result
                terminated = terminated or truncated
            else:  # Gym format
                obs, reward, terminated, info = step_result
                
            total_rewards += reward
            
            # Check if episode is done
            if (isinstance(terminated, bool) and terminated) or \
               (hasattr(terminated, '__iter__') and all(terminated)):
                break
            
        all_rewards.append(total_rewards)
    
    average_reward = sum(all_rewards) / num_episodes
    return average_reward
