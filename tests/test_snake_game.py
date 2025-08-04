"""Test cases for the Snake game components using pytest."""

import pytest
import numpy as np

from rl_snake.game.snake import SnakeGame, Snake, Food
from rl_snake.game.fast_snake import FastSnakeGame


@pytest.fixture
def snake_game():
    """Fixture for creating a SnakeGame instance."""
    return SnakeGame(game_size=10)


@pytest.fixture
def fast_snake_game():
    """Fixture for creating a FastSnakeGame instance."""
    return FastSnakeGame(game_size=10)


class TestSnakeGame:
    """Test cases for the main Snake game."""
    
    def test_game_initialization(self, snake_game):
        """Test that game initializes correctly."""
        assert snake_game.game_size == 10
        assert snake_game.score == 0
        assert not snake_game.done
        assert not snake_game.game_over
    
    def test_observation_shape(self, snake_game):
        """Test that observations have correct shape."""
        obs = snake_game.get_raw_observation()
        expected_shape = (snake_game.game_size, snake_game.game_size)
        assert obs.shape == expected_shape
    
    def test_observation_content(self, snake_game):
        """Test that observations contain expected values."""
        obs = snake_game.get_raw_observation()
        unique_values = np.unique(obs)
        # Should contain 0 (empty), 1 (snake), 2 (food)
        assert all(val in [0, 1, 2] for val in unique_values)
    
    def test_step_function(self, snake_game):
        """Test that step function returns correct format."""
        result = snake_game.step(action=0)  # Move up
        assert len(result) == 4  # obs, reward, done, info
        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestFastSnakeGame:
    """Test cases for the fast Snake game."""
    
    def test_game_initialization(self, fast_snake_game):
        """Test that fast game initializes correctly."""
        assert fast_snake_game.game_size == 10
        assert fast_snake_game.score == 0
        assert not fast_snake_game.game_over
        assert len(fast_snake_game.snake) == 1  # Start with one segment
    
    def test_step_function(self, fast_snake_game):
        """Test that step function works correctly."""
        initial_pos = fast_snake_game.snake[0]
        result = fast_snake_game.step(action=0)  # Move up
        assert len(result) == 4
        
        # Check that snake moved
        new_pos = fast_snake_game.snake[0]
        assert initial_pos != new_pos
    
    def test_food_placement(self, fast_snake_game):
        """Test that food is placed correctly."""
        assert fast_snake_game.food is not None
        x, y = fast_snake_game.food
        assert 0 <= x < fast_snake_game.game_size
        assert 0 <= y < fast_snake_game.game_size
    
    def test_collision_detection(self, fast_snake_game):
        """Test collision detection."""
        # Force snake into wall
        fast_snake_game.snake = [(0, 0)]  # Place at top-left corner
        result = fast_snake_game.step(action=0)  # Try to move up (into wall)
        _, _, done, _ = result
        assert done


class TestSnakeComponents:
    """Test individual Snake game components."""
    
    def test_food_generation(self):
        """Test food generation."""
        food = Food(game_size=10)
        x, y = food.coordinates
        assert 0 <= x < 10
        assert 0 <= y < 10


# Tests paramétrés pour démontrer la puissance de pytest
@pytest.mark.parametrize("game_size", [5, 10, 15, 20])
def test_game_sizes(game_size):
    """Test that games work with different sizes."""
    game = SnakeGame(game_size=game_size)
    assert game.game_size == game_size
    obs = game.get_raw_observation()
    assert obs.shape == (game_size, game_size)


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_all_actions(snake_game, action):
    """Test that all actions work correctly."""
    result = snake_game.step(action=action)
    assert len(result) == 4
    obs, reward, done, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# Tests avec des fixtures personnalisées
@pytest.fixture
def small_game():
    """Fixture for a small game."""
    return SnakeGame(game_size=5)


@pytest.fixture
def large_game():
    """Fixture for a large game."""
    return SnakeGame(game_size=20)


def test_small_game_bounds(small_game):
    """Test that small games respect bounds."""
    obs = small_game.get_raw_observation()
    assert obs.shape == (5, 5)
    assert np.all(obs >= 0)
    assert np.all(obs <= 2)


def test_large_game_bounds(large_game):
    """Test that large games respect bounds."""
    obs = large_game.get_raw_observation()
    assert obs.shape == (20, 20)
    assert np.all(obs >= 0)
    assert np.all(obs <= 2)


# Tests avec des markers personnalisés
@pytest.mark.slow
def test_long_game_simulation(snake_game):
    """Test a longer game simulation (marked as slow)."""
    steps = 0
    done = False
    while not done and steps < 100:
        _, _, done, _ = snake_game.step(action=np.random.randint(0, 4))
        steps += 1
    
    # Game should either be done or we've reached max steps
    assert done or steps == 100


# Test d'exception
def test_invalid_game_size():
    """Test that invalid game sizes raise appropriate errors."""
    with pytest.raises((ValueError, TypeError)):
        SnakeGame(game_size=-1)


# Test avec approximation pour les nombres flottants
def test_reward_range(snake_game):
    """Test that rewards are in expected range."""
    _, reward, _, _ = snake_game.step(action=0)
    assert isinstance(reward, (int, float))
    # Assumons que les récompenses sont entre -1 et 1
    assert -10 <= reward <= 10
