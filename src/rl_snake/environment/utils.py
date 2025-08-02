"""Utility functions for environment management and model operations."""

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .snake_env import SnakeEnv

from pathlib import Path
import numpy as np
import time


def get_env(n_envs: int = 5, 
            use_frame_stack: bool = False, 
            n_stack: int = 4, 
            game_size: int = 30, 
            fast_game: bool = True):
    """
    Create a vectorized environment for training.
    
    Args:
        n_envs: Number of parallel environments
        use_frame_stack: Whether to use frame stacking
        n_stack: Number of frames to stack
        game_size: Size of the game grid
        fast_game: Whether to use fast game implementation
        
    Returns:
        Vectorized environment
    """
    # make_vec_env handles the multiprocessing details
    env = make_vec_env(
        lambda: SnakeEnv(game_size=game_size, fast_game=fast_game),
        n_envs=n_envs,  
        seed=42    
    )
    if use_frame_stack:
        env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
    return env


class ModelLoader:
    """Base class for loading trained models."""
    
    def __init__(self, name: str, use_frame_stack: bool = False, game_size: int = 30, 
                 n_stack: int = 4, fast_game: bool = True, n_envs: int = 5):
        """
        Initialize model loader.
        
        Args:
            name: Model name/filename
            use_frame_stack: Whether model uses frame stacking
            game_size: Size of the game grid
            n_stack: Number of frames to stack
            fast_game: Whether to use fast game implementation
        """
        name = name if name.endswith(".zip") else f"{name}.zip"
        # Updated path to use models folder
        path = Path().cwd() / "models" / name
        if not path.exists():
            raise FileNotFoundError(
                f"Model file {path} does not exist. Please train the model first."
            )
        
        self.n_stack = n_stack if use_frame_stack else 1
        self.use_frame_stack = use_frame_stack
        self.n_envs = n_envs
        self.name = name
        self.game_size = game_size
        self.fast_game = fast_game

        env = get_env(
            use_frame_stack=use_frame_stack, 
            game_size=game_size, 
            n_stack=n_stack, 
            fast_game=fast_game,
            n_envs=n_envs
            
        )

        if "PPO" in name:
            self.model = PPO.load(path, env=env)  
        elif "DQN" in name:
            self.model = DQN.load(path, env=env)  
        else:
            raise ValueError(f"Model {name} is not supported for rendering.")


class ModelRenderer(ModelLoader):
    """Class for rendering trained models in action."""
    
    def __init__(self, name: str, use_frame_stack: bool = False, game_size: int = 30, 
                 n_stack: int = 4, fast_game: bool = False, verbose: bool = True):
        """
        Initialize model renderer.
        
        Args:
            name: Model name/filename
            use_frame_stack: Whether model uses frame stacking
            game_size: Size of the game grid
            n_stack: Number of frames to stack
            fast_game: Whether to use fast game (usually False for rendering)
        """
        super().__init__(name, use_frame_stack, game_size, n_stack, fast_game=fast_game)
        self.env = SnakeEnv(game_size=game_size, fast_game=fast_game)
        self.verbose = verbose

    def render(self):
        """Render the model playing the game."""
        obs, _ = self.env.reset()
        terminated = False
        self.env.render()
        stacked_obs = [np.zeros_like(obs)] * self.n_stack 
        step = 0
        Total_reward = 0
        
        while not terminated:
            if self.use_frame_stack:
                # Note: Frame stacking order might need adjustment
                stacked_obs.pop(0)  # Remove the oldest frame
                stacked_obs.append(obs)  # Add the current observation
                obs = np.concatenate([obs] + stacked_obs).reshape((-1, 1)).flatten()
                
            action, _info = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, *_ = self.env.step(action)
            step += 1
            Total_reward += reward
            if self.verbose:
                print("-"*20)
                print(f"Action taken: {action}")
                print(f"Reward received: {reward}")
                print("Distance to food:", obs.take(5))
                print(f"Step: {step}")
            self.env.render()
            time.sleep(0.1)

        if self.verbose:
            print("\n"+"-"*20)
            print("Game over!")
            print(f"Total steps taken: {step}")
            print(f"Total reward: {Total_reward}")
            print(f"Final score: {self.env.snake_game.score}")
        self.env.close()
