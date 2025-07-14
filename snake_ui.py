import random
import numpy as np
import pygame 
from constant import GameConstants

class Snake_cell(pygame.Rect):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        
    def collidefood(self, food):
        return self.colliderect(food.hit_box)
    
    def collidebody(self, boddy):
        return self.colliderect(boddy)

class Snake:
    def __init__(self, x, y, width, height, game_size:int):
        head = Snake_cell(x, y, width, height)
        self.current_direction = None
        self.body = [[head, self.current_direction]]
        self.cte = GameConstants(game_size)
             
    @property
    def head(self):
        return self.body[0][0]
    
    @property
    def coordinate(self):
        return [(cell.x, cell.y) for cell, _ in self.body]
    
    def draw(self, dis):
        for snake_cell, _ in self.body:
            pygame.draw.rect(dis, self.cte.BLACK, snake_cell)
        
    def check_collision(self):
        for snake_cell, _ in self.body[1:]:
            if self.head.collidebody(snake_cell):
                self.current_direction = None  
                return False  
        return True
    
    def check_border_collision(self, border):
        if not border.contains(self.head):
            self.current_direction = None
            return False
        return True

    def move(self):
        for snake_cell, direction in self.body:
            snake_cell.move_ip(*direction)
            
        for i in range(len(self.body)-1, 0, -1):
            self.body[i][1] = self.body[i-1][1]
    
    def grow(self, former_tail_x, former_tail_y, tail_direction):
        snake_tail = Snake_cell(former_tail_x, former_tail_y, self.cte.SNAKE_SIZE, self.cte.SNAKE_SIZE)
        self.body.append([snake_tail, tail_direction])

class Food:
    def __init__(self, game_size:int):
        self.game_size = game_size
        self.cte = GameConstants(game_size)
        self.new_food()
    
    @property
    def coordinate(self):
        return self.x, self.y
    
    def draw(self, dis):
        pygame.draw.circle(dis,self.cte.BLUE, self.coo, self.radius)
        
    def new_food(self):
        self.x = random.randint(0, self.game_size-1) # include
        self.y = random.randint(0, self.game_size-1)
        SNAKE_SIZE = self.cte.SNAKE_SIZE
        X, Y = self.cte.X, self.cte.Y
        self.coo = X+(self.x + 1/2) * SNAKE_SIZE, Y+(self.y + 1/2) * SNAKE_SIZE
        self.radius = self.cte.RADIUS
        self.hit_box = pygame.Rect(X+self.x*SNAKE_SIZE, Y+self.y*SNAKE_SIZE, SNAKE_SIZE, SNAKE_SIZE)

class SnakeGame:
    def __init__(self, game_size:int=15):
        self.game_size = game_size
        self.cte = GameConstants(game_size)
        self.border = pygame.Rect(*self.cte.GAME_SIZE)
        self._snake = Snake(*self.cte.INIT_SNAKE_COO, game_size=game_size)
        self._food = Food(game_size)
        self.score = 0
        self.pause = False
        self.snake_grow = False
        self.game_over = False # real game over from pygame
        self.done = False # game over from the agent
        self.is_render_mode = False
        self.raw_obs = self.get_raw_observation()

    @property
    def snake(self):
        return self._snake.coordinate
    
    @property
    def food(self):
        return self._food.coordinate

    def init_board(self, dis):
        dis.fill(self.cte.WHITE)
        background = pygame.Rect(*self.cte.GAME_SIZE)
        pygame.draw.rect(dis, self.cte.GRAY, background)
        border_dis = pygame.Rect(*self.cte.BORDER_SIZE)
        pygame.draw.rect(dis, self.cte.BLACK, border_dis, self.cte.THICKNESS)
        
    def init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.dis = pygame.display.set_mode(self.cte.DISPLAY_RES, pygame.RESIZABLE)
        self.font = pygame.font.SysFont("verdana", self.cte.FONT_SIZE)
        pygame.display.update()
        pygame.display.set_caption('Snake game')
        self.clock = pygame.time.Clock()
        self.is_render_mode = True

    def render(self):
        self.draw()
        
    def play(self):
        self.init_pygame()
        while not self.game_over:
            self.init_board(self.dis)
            self.get_input_user()
            self.step(not_playing=False)
            self.clock.tick(self.cte.SNAKE_SPEED + 1/2*self.score)
        self.quit()

    def get_input_user(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                
            if event.type == pygame.KEYDOWN :
                if event.key == pygame.K_RETURN:
                    self._snake = Snake(*self.cte.INIT_SNAKE_COO)
                    self._food.new_food()
                    self.done = False
                    self.score = 0

                if event.key == pygame.K_SPACE:
                    if not self.pause :
                        self.pause = True
                        self.old_direction = self._snake.current_direction
                        self._snake.current_direction = None
                    else :
                        self.pause = False
                        self._snake.current_direction = self.old_direction
                        
                if event.key in self.cte.DIRECTION: 
                    self._snake.current_direction = self.cte.DIRECTION[event.key]
                    self._snake.body[0][1] = self._snake.current_direction
                
    def step(self, action=None, not_playing=True):
        if self.done:
            return self.raw_obs, self.score, self.done, {}

        if action is not None:
            self._snake.current_direction = list(self.cte.DIRECTION.values())[action]
            self._snake.body[0][1] = self._snake.current_direction
        
        border_collision = self._snake.check_border_collision(self.border)
        snake_collision = self._snake.check_collision()
        if not (snake_collision and border_collision):
            self.done = True  
            # self.game_over = True
            return self.raw_obs, self.score, self.done, {}
        
        tail, tail_direction = self._snake.body[-1]
        tail_x, tail_y = tail.x, tail.y
        
        if self._snake.current_direction is not None:
            self._snake.move()

        
        if self._snake.head.collidefood(self._food):
            self._food.new_food()
            self.score += 1
            self.snake_grow = True
            
        if self.is_render_mode:
            self.draw()
        
        if self.snake_grow : # will be draw next iteration
            self._snake.grow(tail_x, tail_y, tail_direction)
            self.snake_grow = False 
        
        if not_playing:
            self.done = False if snake_collision and border_collision else True
            if not self.done :
                self.raw_obs = self.get_raw_observation()
            return self.raw_obs, self.score, self.done, {}

    def get_raw_observation(self):
        raw_obs = np.zeros((self.game_size, self.game_size), dtype=np.int32)
        cliped = lambda x: max(0, min(self.game_size-1, x))  # Ensure within bounds
        for cell, _ in self._snake.body:
            x, y = (cell.x-self.cte.X)//self.cte.SNAKE_SIZE, (cell.y-self.cte.Y)//self.cte.SNAKE_SIZE
            x, y = cliped(x), cliped(y)  # Ensure within bounds
            raw_obs[y, x] = 1
        x, y = cliped(self._food.x), cliped(self._food.y)
        raw_obs[y, x] = 2
        return raw_obs
    
    def draw_score(self, dis, font, score):
        text = font.render(f'Score: {score}', True, self.cte.BLACK)
        score_coo = (self.cte.SCORE_COO[0] , self.cte.SCORE_COO[1]-text.get_height()//2)
        dis.blit(text, score_coo)

    def draw(self):
        if not self.is_render_mode: # always init pygame before drawing
            self.init_pygame()
            self.is_render_mode = True
            
        self.init_board(self.dis)  
        self._snake.draw(self.dis)
        self._food.draw(self.dis)
        self.draw_score(self.dis, self.font, self.score)
        pygame.display.flip()
    
    def quit(self):
        pygame.quit() 
        
        
def test_speed():
    from time import time
    game = SnakeGame()
    action = 1
    i = 0
    T = time()
    for i in range(100_000):
        action = 0 if i % 2 == 0 else 2
        game.step(action) # up/down
    return time()-T
    
if __name__ == "__main__":
    # print("\ntest_speed: ", test_speed(), "\n")

    game = SnakeGame()
    game.play()