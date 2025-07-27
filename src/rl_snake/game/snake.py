"""Snake game engine and components."""

import random
import numpy as np
import pygame 
from ..config.constants import GameConstants


class SnakeCell(pygame.Rect):
    """Individual cell of the snake body."""
    
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        
    def collide_food(self, food):
        """Check collision with food."""
        return self.colliderect(food.hit_box)
    
    def collide_body(self, body_part):
        """Check collision with another body part."""
        return self.colliderect(body_part)


class Snake:
    """Snake game entity."""
    
    def __init__(self, x, y, width, height, game_size: int):
        """
        Initialize snake.
        
        Args:
            x, y: Initial position
            width, height: Snake cell dimensions
            game_size: Size of the game grid
        """
        head = SnakeCell(x, y, width, height)
        self.current_direction = None
        self.body = [[head, self.current_direction]]
        self.cte = GameConstants(game_size)
             
    @property
    def head(self):
        """Get the head of the snake."""
        return self.body[0][0]
    
    @property
    def coordinates(self):
        """Get coordinates of all body parts."""
        return [(cell.x, cell.y) for cell, _ in self.body]
    
    def draw(self, display):
        """Draw the snake on the display."""
        for snake_cell, _ in self.body:
            pygame.draw.rect(display, self.cte.BLACK, snake_cell)
        
    def check_collision(self):
        """Check if snake collides with itself."""
        for snake_cell, _ in self.body[1:]:
            if self.head.collide_body(snake_cell):
                self.current_direction = None  
                return False  
        return True
    
    def check_border_collision(self, border):
        """Check if snake collides with borders."""
        if not border.contains(self.head):
            self.current_direction = None
            return False
        return True

    def move(self):
        """Move the snake in the current direction."""
        for snake_cell, direction in self.body:
            snake_cell.move_ip(*direction)
            
        for i in range(len(self.body)-1, 0, -1):
            self.body[i][1] = self.body[i-1][1]
    
    def grow(self, former_tail_x, former_tail_y, tail_direction):
        """Add a new segment to the snake."""
        snake_tail = SnakeCell(former_tail_x, former_tail_y, self.cte.SNAKE_SIZE, self.cte.SNAKE_SIZE)
        self.body.append([snake_tail, tail_direction])


class Food:
    """Food entity in the game."""
    
    def __init__(self, game_size: int):
        """
        Initialize food.
        
        Args:
            game_size: Size of the game grid
        """
        self.game_size = game_size
        self.cte = GameConstants(game_size)
        self.new_food()
    
    @property
    def coordinates(self):
        """Get food coordinates."""
        return [self.x, self.y]
    
    def draw(self, display):
        """Draw the food on the display."""
        pygame.draw.circle(display, self.cte.BLUE, self.position, self.radius)
        
    def new_food(self):
        """Generate new food at random position."""
        self.x = random.randint(0, self.game_size-1)
        self.y = random.randint(0, self.game_size-1)
        SNAKE_SIZE = self.cte.SNAKE_SIZE
        X, Y = self.cte.X, self.cte.Y
        self.position = X+(self.x + 1/2) * SNAKE_SIZE, Y+(self.y + 1/2) * SNAKE_SIZE
        self.radius = self.cte.RADIUS
        self.hit_box = pygame.Rect(X+self.x*SNAKE_SIZE, Y+self.y*SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE)


class SnakeGame:
    """Main Snake game engine."""
    
    def __init__(self, game_size: int = 15):
        """
        Initialize the Snake game.
        
        Args:
            game_size: Size of the game grid (NxN)
        """
        self.n_steps = 0
        self.game_size = game_size
        self.cte = GameConstants(game_size)
        self.border = pygame.Rect(*self.cte.GAME_SIZE)
        self._snake = Snake(*self.cte.INIT_SNAKE_COO, game_size=game_size)
        self._food = Food(game_size)
        self.score = 0
        self.pause = False
        self.snake_grow = False
        self.game_over = False  # Real game over from pygame
        self.done = False  # Game over from the agent
        self.is_render_mode = False
        self.raw_obs = self.get_raw_observation()
    
    def reset(self):
        self.__init__(self.game_size)

    def set_game_size(self, new_size):
        """
        Set a new game size.
        
        Args:
            new_size: New size for the game grid (NxN)
        """
        if new_size < 5:
            raise ValueError("Game size must be greater than 5.")
        self.__init__(new_size)

    @property
    def snake(self):
        """Get snake coordinates."""
        #todo reverse coordinates to match env observation format. todo: serach why in 'get_raw_obs'
        return [self._get_snake_coo(*coord)[::-1] for coord in self._snake.coordinates]

    @property
    def food(self):
        """Get food coordinates."""
        return self._food.coordinates[::-1] # same

    def init_board(self, display):
        """Initialize the game board display."""
        display.fill(self.cte.WHITE)
        background = pygame.Rect(*self.cte.GAME_SIZE)
        pygame.draw.rect(display, self.cte.GRAY, background)
        border_display = pygame.Rect(*self.cte.BORDER_SIZE)
        pygame.draw.rect(display, self.cte.BLACK, border_display, self.cte.THICKNESS)
        
    def init_pygame(self):
        """Initialize pygame components."""
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(self.cte.DISPLAY_RES, pygame.RESIZABLE)
        self.font = pygame.font.SysFont("verdana", self.cte.FONT_SIZE)
        pygame.display.update()
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.is_render_mode = True

    def render(self):
        """Render the game."""
        self.draw()

    def console_render(self):
        for line in self.raw_obs:
            print(" ".join(str(int(cell)) for cell in line))
        
    def play(self):
        """Start the interactive game loop."""
        self.init_pygame()
        while not self.game_over:
            self.init_board(self.display)
            self.get_input_user()
            self.step(not_playing=False)
            self.clock.tick(self.cte.SNAKE_SPEED + 1/2*self.score)
        self.quit()

    def get_input_user(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self._snake = Snake(*self.cte.INIT_SNAKE_COO, game_size=self.game_size)
                    self._food.new_food()
                    self.done = False
                    self.score = 0

                if event.key == pygame.K_SPACE:
                    if not self.pause:
                        self.pause = True
                        self.old_direction = self._snake.current_direction
                        self._snake.current_direction = None
                    else:
                        self.pause = False
                        self._snake.current_direction = self.old_direction
                        
                if event.key in self.cte.DIRECTION: 
                    self._snake.current_direction = self.cte.DIRECTION[event.key]
                    self._snake.body[0][1] = self._snake.current_direction
                
    def step(self, action=None, not_playing=True):
        """
        Execute one game step.
        
        Args:
            action: Action to take (for RL agent)
            not_playing: Whether this is called from RL training
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            return self.raw_obs, self.score, self.done, self._get_info()
        self.n_steps += 1
        
        if action is not None:
            self._snake.current_direction = list(self.cte.DIRECTION.values())[action]
            self._snake.body[0][1] = self._snake.current_direction
        
        border_collision = self._snake.check_border_collision(self.border)
        snake_collision = self._snake.check_collision()
        if not (snake_collision and border_collision):
            self.done = True
            return self.raw_obs, self.score, self.done, self._get_info()

        tail, tail_direction = self._snake.body[-1]
        tail_x, tail_y = tail.x, tail.y
        
        if self._snake.current_direction is not None:
            self._snake.move()

        if self._snake.head.collide_food(self._food):
            self._food.new_food()
            self.score += 1
            self.snake_grow = True
            
        if self.is_render_mode:
            self.draw()
        
        if self.snake_grow:  # Will be drawn next iteration
            self._snake.grow(tail_x, tail_y, tail_direction)
            self.snake_grow = False 
        
        if not_playing:
            self.done = False if snake_collision and border_collision else True
            if not self.done:
                self.raw_obs = self.get_raw_observation()
            return self.raw_obs, self.score, self.done, self._get_info()

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

    def _get_snake_coo(self, x, y):
        """Get x, y coordinates within the game grid."""
        x, y = (x-self.cte.X)//self.cte.SNAKE_SIZE, (y-self.cte.Y)//self.cte.SNAKE_SIZE
        return x, y

    def get_raw_observation(self):
        """Get the current game state as a numpy array."""
        raw_obs = np.zeros((self.game_size, self.game_size), dtype=np.int32)
        clipped = lambda x: max(0, min(self.game_size-1, x))  # Ensure within bounds
        
        for cell, _ in self._snake.body:
            x, y = self._get_snake_coo(cell.x, cell.y)
            x, y = clipped(x), clipped(y)  # Ensure within bounds
            raw_obs[y, x] = 1
            
        x, y = clipped(self._food.x), clipped(self._food.y)
        raw_obs[y, x] = 2
        return raw_obs
    
    def draw_score(self, display, font, score):
        """Draw the current score on the display."""
        text = font.render(f'Score: {score}', True, self.cte.BLACK)
        score_position = (self.cte.SCORE_COO[0], self.cte.SCORE_COO[1]-text.get_height()//2)
        display.blit(text, score_position)

    def draw(self):
        """Draw the complete game state."""
        if not self.is_render_mode:  # Always init pygame before drawing
            self.init_pygame()
            self.is_render_mode = True
            
        self.init_board(self.display)  
        self._snake.draw(self.display)
        self._food.draw(self.display)
        self.draw_score(self.display, self.font, self.score)
        pygame.display.flip()
    
    def quit(self):
        """Clean up pygame resources."""
        pygame.quit() 


def test_speed():
    """Performance test for the game engine."""
    from time import time
    game = SnakeGame()
    action = 1
    i = 0
    start_time = time()
    for i in range(100_000):
        action = 0 if i % 2 == 0 else 2
        game.step(action)  # up/down
    return time() - start_time
    
def main():
    """Main function to run the game."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the Snake game engine.")
    parser.add_argument("--game_size", type=int, default=30, help="Size of the game grid (NxN).")
    args = parser.parse_args()
    
    game = SnakeGame(game_size=args.game_size)
    game.play()

if __name__ == "__main__":
    # print("\ntest_speed: ", test_speed(), "\n")
    main()
