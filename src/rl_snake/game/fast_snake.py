"""Fast Snake game implementation without pygame dependencies."""

import random
import numpy as np

DEFAULT_GAME_SIZE = 30


class FastSnakeGame:
    """
    Lightweight Snake game implementation for fast training.
    
    This version doesn't use pygame and focuses on pure game logic
    for faster reinforcement learning training.
    """
    
    def __init__(self, game_size=DEFAULT_GAME_SIZE):
        """
        Initialize the fast Snake game.
        
        Args:
            game_size: Size of the game grid (NxN)
        """
        self.game_size = game_size
        self.reset()
    
    def set_game_size(self, new_size):
        """
        Set a new game size.
        
        Args:
            new_size: New size for the game grid (NxN)
        """
        self.game_size = new_size
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.snake = [(self.game_size // 2, self.game_size // 2)]
        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False
        self.n_steps = 0

    def _place_food(self):
        """Place food at a random position not occupied by the snake."""
        while self.food is None or self.food in self.snake:
            self.food = (
                random.randint(0, self.game_size - 1), 
                random.randint(0, self.game_size - 1)
            )

    def step(self, action):
        """
        Execute one game step.
        
        Args:
            action: Action to take (0-Up, 1-Left, 2-Down, 3-Right)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.game_over:
            return self.raw_obs, self.score, self.game_over, self._get_info()
        self.n_steps += 1

        # Directions: 0-Up, 1-Left, 2-Down, 3-Right
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        direction = directions[action]
        new_head = (
            self.snake[0][0] + direction[0], 
            self.snake[0][1] + direction[1]
        )
        
        # Check for game over conditions
        if (new_head in self.snake or 
            new_head[0] < 0 or new_head[0] >= self.game_size or 
            new_head[1] < 0 or new_head[1] >= self.game_size):
            self.game_over = True
            return self.raw_obs, self.score, self.game_over, self._get_info()

        self.snake.insert(0, new_head)

        # Check if snake eats food
        if new_head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return self.raw_obs, self.score, self.game_over, self._get_info()

    def _get_info(self):
        """Get additional game info."""
        return {
            "n_steps": self.n_steps,
            "snake_length": len(self.snake),
            "score": self.score,
            "game_over": self.game_over,
            "food_position": self.food,
            "head_position": self.snake[0],
        }

    @property
    def raw_obs(self):
        """Get the current game observation."""
        return self.get_raw_observation()
    
    def get_raw_observation(self):
        """
        Get the current game state as a numpy array.
        
        Returns:
            numpy array where 0=empty, 1=snake, 2=food
        """
        raw_obs = np.zeros((self.game_size, self.game_size), dtype=np.int32)
        coords = np.array(self.snake)
        raw_obs[coords[:, 0], coords[:, 1]] = 1
        x, y = self.food
        raw_obs[x, y] = 2
        return raw_obs
    
    def render(self):
        """Render the game state to console."""
        for line in self.raw_obs:
            print(line, end="\n")
        print()
        
    def play(self, action):
        """
        Play one step and render the result.
        
        Args:
            action: Action to take
        """
        obs, score, done, _ = self.step(action)
        self.render()
        if done:
            print(f"Game Over! Final Score: {score}")
        
    def quit(self):
        """Clean up resources (no-op for this implementation)."""
        pass
