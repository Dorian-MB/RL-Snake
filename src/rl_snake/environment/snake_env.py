"""Snake environment for reinforcement learning training."""

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch

OBS_SHAPE = 7
SHAPE = (OBS_SHAPE,)


class BaseSnakeEnv(gym.Env, ABC):
    """
    Base class for Snake environment.

    This class provides the common interface and functionality
    for different Snake environment implementations.
    """

    def __init__(self, game_size, fast_game, seed=42):
        """
        Initialize the base Snake environment.

        Args:
            game_size: Size of the game grid
            fast_game: Whether to use the fast game implementation
        """
        super(BaseSnakeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions
        self.observation_space = gym.spaces.Box(
            low=-5, high=5, shape=SHAPE, dtype=np.float64
        )  # -pi; pi for angle , [0, 1, 2, 3] for neighbors, [0, 2] distance
        self.render_mode = "human"  # Default render mode
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = {"render_modes": ["human"]}
        self.game_size = game_size
        self.fast_game = fast_game
        self.seed(seed)

        if fast_game:
            from ..game.fast_snake import FastSnakeGame

            self.SnakeGameHandler = FastSnakeGame
        else:
            from ..game.snake import SnakeGame

            self.SnakeGameHandler = SnakeGame

        self.snake_game = self.SnakeGameHandler(self.game_size)

    def set_game_size(self, new_size):
        """
        Set a new game size.

        Args:
            new_size: New size for the game grid (NxN)
        """
        self.game_size = new_size
        self.snake_game.set_game_size(new_size)

    @property
    def food_position(self):
        """Get the position of the food."""
        return np.array(self.snake_game.food)

    @property
    def snake_positions(self):
        """Get all positions of the snake."""
        return np.array(self.snake_game.snake)

    @property
    def head_position(self):
        """Get the position of the snake's head."""
        return np.array(self.snake_game.snake[0])

    @property
    def raw_obs(self):
        """Get the raw observation from the game."""
        return self.snake_game.raw_obs

    def close(self):
        """Clean up environment resources."""
        self.snake_game.quit()

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            self.snake_game.render()

    def seed(self, seed=42):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        # torch.manual_seed(seed)

    @abstractmethod
    def step(self, action):
        """Take a step in the environment (to be implemented by subclasses)."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @abstractmethod
    def reset(self, seed=42):
        """Reset the environment (to be implemented by subclasses)."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class SnakeEnv(BaseSnakeEnv):
    """
    Main Snake environment for reinforcement learning.

    This environment provides:
    - step(action): Execute action and return (observation, reward, done, info)
    - reset(): Reset environment to initial state
    - render(): Visualize the current state
    - close(): Clean up resources
    """

    def __init__(self, game_size: int = 30, fast_game: bool = True, seed: int = 42):
        """
        Initialize the Snake environment.

        Args:
            game_size: Size of the game grid (NxN)
            fast_game: Whether to use fast game implementation (no pygame)
        """
        super(SnakeEnv, self).__init__(game_size=game_size, fast_game=fast_game, seed=seed)
        food = np.array(self.snake_game.food)
        head = np.array(self.snake_game.snake[0])
        self._last_distance = self.manhattan_distance(
            head=head, food=food
        )  # update each step
        self.distance_to_food = self.manhattan_distance(
            head=head, food=food
        )  # update each time snake eats food
        self.previous_score = 0
        self.food_steps = 0  # Track number of steps taken to eat food

    def euclidean_distance(self, head=None, food=None):
        """
        Calculate the Euclidean distance between snake head and food.

        Args:
            head: Snake head position (optional)
            food: Food position (optional)

        Returns:
            Normalized Euclidean distance
        """
        head = head if head is not None else self.snake_positions[0]
        food = food if food is not None else self.food_position
        new_distance = np.linalg.norm(head - food)
        return new_distance / self.game_size  # Normalize by game size

    def manhattan_distance(self, head=None, food=None):
        """
        Calculate the Manhattan distance between snake head and food.

        Args:
            head: Snake head position (optional)
            food: Food position (optional)

        Returns:
            Normalized Manhattan distance
        """
        head = head if head is not None else self.snake_positions[0]
        food = food if food is not None else self.food_position
        new_distance = np.abs(head[0] - food[0]) + np.abs(head[1] - food[1])
        return new_distance / self.game_size  # normalize

    def get_neighbors(self, grid, head, out_of_bounds_value=3):
        """
        Get the neighboring cells around the snake's head.

        Args:
            grid: Game grid
            head: Snake head position
            out_of_bounds_value: Value for out-of-bounds cells

        Returns:
            Array of neighbor values [up, left, down, right]
        """
        # Add padding to handle borders
        padded = np.pad(grid, 1, mode="constant", constant_values=out_of_bounds_value)
        # Adjust indices for padded array
        i, j = head
        pi, pj = i + 1, j + 1
        # Get neighbors
        neighbors = np.array(
            [
                padded[pi - 1, pj],  # up
                padded[pi, pj - 1],  # left
                padded[pi + 1, pj],  # down
                padded[pi, pj + 1],  # right
            ]
        )
        return neighbors

    def angle_between_snake_head_and_food(self, head, food):
        """
        Calculate the angle between the snake's head and the food.

        Args:
            head: Snake head position
            food: Food position

        Returns:
            Angle in radians
        """
        delta_x = food[0] - head[0]
        delta_y = food[1] - head[1]
        self.angle = np.array([np.arctan2(delta_y, delta_x)])
        return self.angle

    def feature_gen(self, raw_obs, info):
        """
        Generate feature vector from raw observation.

        Args:
            raw_obs: Raw game observation

        Returns:
            Feature vector combining neighbors, distance, and angle
        """
        new_distance = self.manhattan_distance()
        distance = np.array([new_distance])
        head = self.head_position
        food = self.food_position
        # Get direct neighbors of snake head
        neighbors = self.get_neighbors(raw_obs, head)
        angle = self.angle_between_snake_head_and_food(head, food)
        time_to_food = np.array(
            [
                min(info["n_steps"] - self.food_steps, 2 * self.distance_to_food)
                / self.distance_to_food
            ]
        )  # normalize to [0, 2]

        obs = np.concatenate([neighbors, distance, angle, time_to_food])
        return obs

    def get_reward(self, score, done, info):
        """
        Calculate reward based on game state.

        Args:
            score: Current game score
            done: Whether game is finished

        Returns:
            Reward value
        """
        # Calculate distance between the snake and the food
        new_distance = self.manhattan_distance()

        # Check if the snake has eaten food and update the reward
        if self.previous_score != score:
            reward = 10 + 50 / max(info["n_steps"] - self.food_steps, 1)
            self.food_steps = info["n_steps"]
            self.previous_score = score
            self.distance_to_food = new_distance
        elif done:
            reward = -30
        else:
            reward = 0.5 if new_distance <= self._last_distance else -0.6

        self._last_distance = new_distance
        return reward

    @property
    def obs(self):
        """Get current observation."""
        raw_obs = self.snake_game.raw_obs
        info = self.snake_game._get_info()
        return self.feature_gen(raw_obs, info)

    def _get_info(self, info={}, obs=None):
        """
        Get additional information about the game state.
        And dict obs type
        Returns:
            Dictionary with game info
        """
        info = info or self.snake_game._get_info()
        obs = obs if obs is not None else self.obs
        neighbors = obs[:4]  # First 4 values are neighbors
        distance = obs[4]  # Next value is distance
        angle = obs[5]  # Next value is angle
        time_to_food = obs[6]  # Last value is time to food
        return {
            **info,
            "obs": {
                "neighbors": neighbors,
                "distance": distance,
                "angle": angle,
                "time_to_food": time_to_food,
            },
        }

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0-3)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        raw_obs, score, done, info = self.snake_game.step(action)
        obs = self.feature_gen(raw_obs, info)
        reward = self.get_reward(score, done, info)
        terminated = done  # done = self.snake_game.game_over
        truncated = False  # No time limit in this case
        return obs, reward, terminated, truncated, self._get_info(info, obs)

    def reset(self, seed=42):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (initial_observation, info)
        """
        self.seed(seed)
        self.__init__(self.game_size, self.fast_game, seed=seed)
        return self.obs, self._get_info()
