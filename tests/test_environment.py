"""Test cases for the RL environment."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rl_snake.environment.snake_env import SnakeEnv
import gymnasium as gym


class TestSnakeEnvironment(unittest.TestCase):
    """Test cases for the Snake RL environment."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.env = SnakeEnv(game_size=10, fast_game=True)
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        self.assertIsInstance(self.env, gym.Env)
        self.assertEqual(self.env.game_size, 10)
        self.assertTrue(self.env.fast_game)
    
    def test_action_space(self):
        """Test action space is correctly defined."""
        self.assertIsInstance(self.env.action_space, gym.spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)  # 4 possible actions
    
    def test_observation_space(self):
        """Test observation space is correctly defined."""
        self.assertIsInstance(self.env.observation_space, gym.spaces.Box)
        expected_shape = (6,)  # Based on OBS_SHAPE constant
        self.assertEqual(self.env.observation_space.shape, expected_shape)
    
    def test_reset_function(self):
        """Test that reset function works correctly."""
        result = self.env.reset()
        self.assertEqual(len(result), 2)  # obs, info
        obs, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape, (6,))
    
    def test_step_function(self):
        """Test that step function returns correct format."""
        self.env.reset()
        result = self.env.step(action=0)
        self.assertEqual(len(result), 5)  # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = result
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_feature_generation(self):
        """Test feature generation from raw observation."""
        obs = self.env.obs
        self.assertEqual(len(obs), 6)  # 4 neighbors + 1 distance + 1 angle
        self.assertIsInstance(obs, np.ndarray)
    
    def test_reward_calculation(self):
        """Test reward calculation logic."""
        # Test initial state
        initial_reward = self.env.get_reward(score=0, done=False)
        self.assertIsInstance(initial_reward, (int, float))
        
        # Test food eaten reward
        food_reward = self.env.get_reward(score=1, done=False)
        self.assertEqual(food_reward, 10)  # Food eaten should give +10
        
        # Test game over penalty
        game_over_reward = self.env.get_reward(score=0, done=True)
        self.assertEqual(game_over_reward, -10)  # Game over should give -10
    
    def test_distance_calculation(self):
        """Test Euclidean distance calculation."""
        head = np.array([0, 0])
        food = np.array([3, 4])
        distance = self.env.euclidean_distance(head=head, food=food)
        expected_distance = 5.0 / self.env.game_size  # Normalized by game size
        self.assertAlmostEqual(distance, expected_distance, places=5)


if __name__ == "__main__":
    unittest.main()
