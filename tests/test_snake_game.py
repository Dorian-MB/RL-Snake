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

@pytest.fixture(params=[SnakeGame, FastSnakeGame])
def any_game(request):
    """Fixture qui retourne soit SnakeGame soit FastSnakeGame."""
    GameClass = request.param
    return GameClass(game_size=10)

class TestSnakeGame:
    """Test cases for the main Snake game."""
    
    def test_game_initialization(self, any_game):
        """Test that game initializes correctly."""
        assert any_game.game_size == 10
        assert any_game.score == 0
        assert not any_game.done
    
    def test_observation_shape(self, any_game):
        """Test that observations have correct shape."""
        obs = any_game.get_raw_observation()
        expected_shape = (any_game.game_size, any_game.game_size)
        assert obs.shape == expected_shape
    
    def test_observation_content(self, any_game):
        """Test that observations contain expected values."""
        obs = any_game.get_raw_observation()
        unique_values = np.unique(obs)
        # Should contain 0 (empty), 1 (snake), 2 (food)
        assert all(val in [0, 1, 2] for val in unique_values)
    
    def test_step_function(self, any_game):
        """Test that step function returns correct format."""
        result = any_game.step(action=0)  # Move up
        assert len(result) == 4  # obs, reward, done, info
        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, int)
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestSnakeComponents:
    """Test individual Snake game components."""
    
    def test_food_generation(self):
        """Test food generation."""
        food = Food(game_size=10)
        x, y = food.coordinates
        assert 0 <= x < 10
        assert 0 <= y < 10


# Tests paramétrés pour démontrer la puissance de pytest
@pytest.mark.parametrize("game_size", [5, 20, 30])
def test_game_sizes(any_game, game_size):
    """Test that games work with different sizes."""
    any_game.set_game_size(new_size=game_size)
    assert any_game.game_size == game_size
    obs = any_game.get_raw_observation()
    assert obs.shape == (game_size, game_size)


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_all_actions(any_game, action):
    """Test that all actions work correctly."""
    result = any_game.step(action=action)
    assert len(result) == 4
    obs, reward, done, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)

# Tests avec des markers personnalisés
@pytest.mark.slow
def test_long_game_simulation(any_game):
    """Test a longer game simulation (marked as slow)."""
    steps = 0
    done = False
    while not done and steps < 100:
        _, _, done, _ = any_game.step(action=np.random.randint(0, 4))
        steps += 1
    
    # Game should either be done or we've reached max steps
    assert done or steps == 100


# Test d'exception
@pytest.mark.parametrize("game_size", [-1, 0, 1, 2, 3])
def test_invalid_game_size(any_game, game_size):
    """Test that invalid game sizes raise appropriate errors."""
    with pytest.raises((ValueError, TypeError)):
        any_game(game_size=game_size)


# Test avec approximation pour les nombres flottants
def test_reward_range(any_game):
    """Test that rewards are in expected range."""
    _, reward, _, _ = any_game.step(action=0)
    assert isinstance(reward, (int, float))
    # Assumons que les récompenses sont entre -1 et 1
    assert -10 <= reward <= 10

class TestGameLogic:
    """Test cases for game logic."""

    def test_simple_going_up_live_and_death(self, any_game):
        for _ in range(5):
            any_game.step(action=0)  # Move up
            assert not any_game.done
        any_game.step(action=0)  # Move up
        assert any_game.done
    
    # def test_die_from_itself(self, any_game):
    #     """Test that the snake dies when it runs into itself."""
    #     any_game.snake = [(1, 1), (1, 2), (1, 3), (2, 3), (2, 2), (2, 1)] # Need refacto on pygame game
    #     any_game.food = (9, 9)
    #     any_game.step(action=2)  # Move down
    #     assert any_game.done
