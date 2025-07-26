"""
RL-Snake: Reinforcement Learning Snake Game

A Python package for training and playing Snake using reinforcement learning algorithms.
"""

__version__ = "1.0.0"
__author__ = "Dorian"

# Import main components
try:
    from .config.constants import GameConstants
    from .game.snake import SnakeGame, Snake, SnakeCell, Food
    from .game.fast_snake import FastSnakeGame
    from .environment.snake_env import SnakeEnv
    from .agents.trainer import ModelTrainer
    from .agents.feature_extractor import LinearQNet
    from .environment.utils import ModelLoader, ModelRenderer
except ImportError:
    # Handle import errors gracefully during package installation
    pass

__all__ = [
    "GameConstants",
    "SnakeGame", 
    "Snake",
    "SnakeCell",
    "Food",
    "FastSnakeGame",
    "SnakeEnv",
    "ModelTrainer",
    "LinearQNet",
    "ModelLoader",
    "ModelRenderer",
]
