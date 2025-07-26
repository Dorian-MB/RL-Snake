"""
RL-Snake: Reinforcement Learning Snake Game

A Python package for training and playing Snake using reinforcement learning algorithms.
"""

__version__ = "1.0.0"
__author__ = "Dorian"

# Import main components
try:
    from .environment.snake_env import SnakeEnv
    from .agents.trainer import ModelTrainer
    from .agents.feature_extractor import LinearQNet
    from .environment.utils import ModelLoader, ModelRenderer
except ImportError:
    # Handle import errors gracefully during package installation
    pass

__all__ = [
    "SnakeEnv",
    "ModelTrainer",
    "LinearQNet",
    "ModelLoader",
    "ModelRenderer",
]
