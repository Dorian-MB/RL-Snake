
import random
import numpy as np

N = 30

class FastSnakeGame:
    def __init__(self, size=N):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.score = 0
        self.food = None
        self._place_food()
        self.game_over = False

    def _place_food(self):
        while self.food is None or self.food in self.snake:
            self.food = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))

    def step(self, action):
        if self.game_over:
            return self.raw_obs, self.score, self.game_over, {}

        # Directions: 0-Up, 1-Left, 2-Down,  3-Right
        direction = [(-1, 0), (0, -1), (1, 0),  (0, 1)][action]
        new_head = (self.snake[0][0] + direction[0], self.snake[0][1] + direction[1])
        # Check for game over conditions
        if (new_head in self.snake) or new_head[0] < 0 or new_head[0] >= self.size or new_head[1] < 0 or new_head[1] >= self.size:
            self.game_over = True
            return self.raw_obs, self.score, self.game_over, {}

        self.snake.insert(0, new_head)

        # Check if snake eats food
        if new_head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        return self.raw_obs, self.score, self.game_over, {}

    @property
    def raw_obs(self):
        return self.get_raw_observation()
    
    def get_raw_observation(self):
        raw_obs = np.zeros((self.size, self.size))
        coords = np.array(self.snake)
        raw_obs[coords[:, 0], coords[:, 1]] = 1
        x, y = self.food
        raw_obs[x,y] = 2
        return raw_obs
    
    def render(self):
        for line in self.raw_obs :
            print(line, end="\n")
        print()
        
    def play(self, action):
        obs, score, done, _ = self.step(action)
        self.render()
        if done:
            print(f"Game Over! Final Score: {score}")
        
    def quit(self):
        pass