"""Utility functions for environment management and model operations."""

import time
from pathlib import Path

import imageio
import numpy as np
import pygame
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from .snake_env import BaseSnakeEnv, SnakeEnv


def get_env(
    n_envs: int = 5,
    use_frame_stack: bool = False,
    n_stack: int = 4,
    game_size: int = 30,
    fast_game: bool = True,
    Env: BaseSnakeEnv = SnakeEnv,
):
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
        lambda: Env(game_size=game_size, fast_game=fast_game), n_envs=n_envs, seed=42
    )
    if use_frame_stack:
        env = VecFrameStack(env, n_stack=n_stack, channels_order="first")
    return env


class ModelLoader:
    """Base class for loading trained models."""

    def __init__(
        self,
        name: str,
        use_frame_stack: bool = False,
        game_size: int = 30,
        n_stack: int = 4,
        fast_game: bool = True,
        n_envs: int = 5,
    ):
        """
        Initialize model loader.

        Args:
            name: Model name/filename
            use_frame_stack: Whether model uses frame stacking
            game_size: Size of the game grid
            n_stack: Number of frames to stack
            fast_game: Whether to use fast game implementation
        """
        # Remove .zip extension if present
        base_name = name.replace(".zip", "")

        # Try new structure first: models/model_name/model_name.zip
        model_dir = Path().cwd() / "models" / base_name
        path = model_dir / f"{base_name}.zip"

        # Fallback to old structure: models/model_name.zip
        if not path.exists():
            path = Path().cwd() / "models" / f"{base_name}.zip"
            model_dir = path.parent

        if not path.exists():
            raise FileNotFoundError(
                f"Model file {path} does not exist. Please train the model first."
            )

        self.n_stack = n_stack if use_frame_stack else 1
        self.use_frame_stack = use_frame_stack
        self.n_envs = n_envs
        self.name = base_name
        self.game_size = game_size
        self.fast_game = fast_game

        env = get_env(
            use_frame_stack=use_frame_stack,
            game_size=game_size,
            n_stack=n_stack,
            fast_game=fast_game,
            n_envs=n_envs,
        )

        # Try to load custom feature extractor class
        custom_objects = None
        class_path = model_dir / "feature_extractor.dill"

        if class_path.exists():
            try:
                import json

                import dill

                # Load the class
                with open(class_path, "rb") as f:
                    extractor_class = dill.load(f)

                # Load the kwargs if they exist
                kwargs_path = model_dir / "feature_extractor_kwargs.json"
                extractor_kwargs = {}
                if kwargs_path.exists():
                    with open(kwargs_path, "r") as f:
                        extractor_kwargs = json.load(f)
                    print(f"✅ Loaded feature extractor kwargs: {extractor_kwargs}")

                custom_objects = {
                    "policy_kwargs": {
                        "features_extractor_class": extractor_class,
                        "features_extractor_kwargs": extractor_kwargs,
                    }
                }
                print(f"✅ Loaded custom feature extractor from: {class_path}")
            except ImportError:
                print(f"⚠️  dill not installed, cannot load custom feature extractor")
            except Exception as e:
                print(f"⚠️  Error loading feature extractor: {e}")

        # Load the model
        try:
            if "PPO" in base_name:
                self.model = PPO.load(path, env=env, custom_objects=custom_objects)
            elif "DQN" in base_name:
                self.model = DQN.load(path, env=env, custom_objects=custom_objects)
            else:
                raise ValueError(f"Model {base_name} is not supported for rendering.")
        except (RuntimeError, ValueError) as e:
            if custom_objects is not None and (
                "state_dict" in str(e) or "parameter group" in str(e)
            ):
                # Architecture mismatch between saved files
                error_msg = f"""
                    ❌ INCOMPATIBILITY ERROR ❌
                    There is a mismatch between the saved model files in: {model_dir}

                    The .zip model expects a different architecture than what's defined in:
                    - feature_extractor.dill (the saved class)
                    - feature_extractor_kwargs.json (the parameters)

                    Possible solutions:
                    1. Delete feature_extractor.dill and re-create it with your current LinearQNet class
                    2. Re-train the model to regenerate all files consistently
                    3. Use a notebook to recreate the .dill file with the correct class version

                    Original error: {str(e)}
                    """
                raise RuntimeError(error_msg) from e
            else:
                raise


class ModelRenderer(ModelLoader):
    """Class for rendering trained models in action."""

    def __init__(
        self,
        name: str,
        use_frame_stack: bool = False,
        game_size: int = 30,
        n_stack: int = 4,
        fast_game: bool = False,
        verbose: bool = True,
        save_gif: bool = False,
        gif_path: str = None,
    ):
        """
        Initialize model renderer.

        Args:
            name: Model name/filename
            use_frame_stack: Whether model uses frame stacking
            game_size: Size of the game grid
            n_stack: Number of frames to stack
            fast_game: Whether to use fast game (usually False for rendering)
            verbose: Whether to print game information
            save_gif: Whether to save the gameplay as a GIF
            gif_path: Path to save the GIF (auto-generated if None)
        """
        super().__init__(name, use_frame_stack, game_size, n_stack, fast_game=fast_game)
        self.env = SnakeEnv(game_size=game_size, fast_game=fast_game)
        self.verbose = verbose
        self.save_gif = save_gif

        if save_gif:
            base_path = Path().cwd() / "gifs"
            base_path.mkdir(exist_ok=True, parents=True)
            if gif_path is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                gif_path = f"gameplay_{name.replace('.zip', '')}_{timestamp}.gif"
            self.gif_path = base_path / gif_path
            self.frames = []
        else:
            self.gif_path = None
            self.frames = None

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
                print("-" * 20)
                print(f"Action taken: {action}")
                print(f"Reward received: {reward}")
                print("Distance to food:", obs.take(4))
                print(f"Step: {step}")
            self.env.render()

            # Capture frame for GIF if enabled
            if self.save_gif and self.env.snake_game.is_render_mode:
                frame = pygame.surfarray.array3d(self.env.snake_game.display)
                # Convert from (width, height, channels) to (height, width, channels)
                frame = frame.swapaxes(0, 1)
                self.frames.append(frame)

            time.sleep(0.1)

        if self.verbose:
            print("\n" + "-" * 20)
            print("Game over!")
            print(f"Total steps taken: {step}")
            print(f"Total reward: {Total_reward}")
            print(f"Final score: {self.env.snake_game.score}")

        # Save GIF if enabled
        if self.save_gif and self.frames:
            self._save_gif()

        self.env.close()

    def _save_gif(self):
        """Save captured frames as a GIF file."""
        if not self.frames:
            print("Warning: No frames captured for GIF creation.")
            return

        try:
            print(f"Saving GIF with {len(self.frames)} frames to: {self.gif_path}")
            imageio.mimsave(self.gif_path, self.frames, fps=10, duration=0.1)
            print(f"GIF saved successfully: {self.gif_path}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
