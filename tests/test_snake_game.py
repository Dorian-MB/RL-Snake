"""Test cases for the Snake game components."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rl_snake.game.snake import SnakeGame, Snake, Food
from rl_snake.game.fast_snake import FastSnakeGame


class TestSnakeGame(unittest.TestCase):
    """Test cases for the main Snake game."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.game = SnakeGame(game_size=10)
    
    def test_game_initialization(self):
        """Test that game initializes correctly."""
        self.assertEqual(self.game.game_size, 10)
        self.assertEqual(self.game.score, 0)
        self.assertFalse(self.game.done)
        self.assertFalse(self.game.game_over)
    
    def test_observation_shape(self):
        """Test that observations have correct shape."""
        obs = self.game.get_raw_observation()
        expected_shape = (self.game.game_size, self.game_size)
        self.assertEqual(obs.shape, expected_shape)
    
    def test_observation_content(self):
        """Test that observations contain expected values."""
        obs = self.game.get_raw_observation()
        unique_values = np.unique(obs)
        # Should contain 0 (empty), 1 (snake), 2 (food)
        self.assertTrue(all(val in [0, 1, 2] for val in unique_values))
    
    def test_step_function(self):
        """Test that step function returns correct format."""
        result = self.game.step(action=0)  # Move up
        self.assertEqual(len(result), 4)  # obs, reward, done, info
        obs, reward, done, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


class TestFastSnakeGame(unittest.TestCase):
    """Test cases for the fast Snake game."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.game = FastSnakeGame(game_size=10)
    
    def test_game_initialization(self):
        """Test that fast game initializes correctly."""
        self.assertEqual(self.game.game_size, 10)
        self.assertEqual(self.game.score, 0)
        self.assertFalse(self.game.game_over)
        self.assertEqual(len(self.game.snake), 1)  # Start with one segment
    
    def test_step_function(self):
        """Test that step function works correctly."""
        initial_pos = self.game.snake[0]
        result = self.game.step(action=0)  # Move up
        self.assertEqual(len(result), 4)
        
        # Check that snake moved
        new_pos = self.game.snake[0]
        self.assertNotEqual(initial_pos, new_pos)
    
    def test_food_placement(self):
        """Test that food is placed correctly."""
        self.assertIsNotNone(self.game.food)
        x, y = self.game.food
        self.assertTrue(0 <= x < self.game.game_size)
        self.assertTrue(0 <= y < self.game.game_size)
    
    def test_collision_detection(self):
        """Test collision detection."""
        # Force snake into wall
        self.game.snake = [(0, 0)]  # Place at top-left corner
        result = self.game.step(action=0)  # Try to move up (into wall)
        _, _, done, _ = result
        self.assertTrue(done)


class TestSnakeComponents(unittest.TestCase):
    """Test individual Snake game components."""
    
    def test_food_generation(self):
        """Test food generation."""
        food = Food(game_size=10)
        x, y = food.coordinates
        self.assertTrue(0 <= x < 10)
        self.assertTrue(0 <= y < 10)


if __name__ == "__main__":
    unittest.main()
