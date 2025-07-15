import gymnasium as gym
import numpy as np
from snake_game.snake_game import SnakeGame
from snake_game.fast_snake import FastSnakeGame

OBS_SHAPE = 6
SHAPE = (OBS_SHAPE, )

class SnakeEnv(gym.Env):
    """
    step(action): This method takes an action as input, updates the game state based on that action, returns the new state, the reward gained (or lost), whether the game is over (done), and additional info if necessary.
    reset(): This method resets the environment to an initial state and returns this initial state. It's used at the beginning of a new episode.
    render(): This method is for visualizing the state of the environment. Depending on how you want to view the game, this could simply update the game window.
    close(): This method performs any necessary cleanup, like closing the game window.
    """
    
    def __init__(self, game_size:int=0, fast_game:bool=True):
        super(SnakeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4) # Output
        self.observation_space = gym.spaces.Box(low=-4, high=5, shape=SHAPE, dtype=np.float64)
        self.render_mode = "human"  # Default render mode
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = {"renders_mode":["human"]}
        self.game_size = game_size
        self.SnakeGameHandler = SnakeGame if not fast_game else FastSnakeGame
        self._init()    
    
    def _init(self):
        self.snake_game = self.SnakeGameHandler(self.game_size)
        food = np.array(self.snake_game.food)
        head = np.array(self.snake_game.snake[0])
        self._last_distance = self.euclidean_distance(head=head, food=food)
        self.previous_score = 0
        
        # define during run
        self.food_position = None
        self.snake_positions = None
        self.angle_to_food = None
        
    def seed(self, seed=42): # needed with make_vec_env
        np.random.seed(seed)
    
    def set_snake_and_food_position(self, raw_obs):
        self.food_position = np.argwhere(raw_obs == 2).flatten()
        self.snake_positions = np.argwhere(raw_obs == 1).flatten()
        
    def euclidean_distance(self, head=None, food=None):
        """ Calculate the Euclidean distance between the centroid of the snake and the food position."""
        head = head if head is not None else self.snake_positions.take((0, 1))
        food = food if food is not None else self.food_position
        new_distance = np.linalg.norm(head - food)
        return new_distance/self.game_size  # Normalize by the size of the game grid

    def get_neighbors(self, grid, head, out_of_bounds_value=3):
        """Utilise le padding pour gérer les bordures"""
        # Ajouter un padding de 1 avec la valeur out_of_bounds
        padded = np.pad(grid, 1, mode='constant', constant_values=out_of_bounds_value)
        # Ajuster les indices pour le tableau paddé
        i, j = head
        pi, pj = i + 1, j + 1
        # Récupérer les voisins
        neighbors = np.array([
            padded[pi-1, pj],  # haut
            padded[pi, pj-1],  # gauche
            padded[pi+1, pj],  # bas
            padded[pi, pj+1],  # droite
        ])
        return neighbors

    def angle_between_snake_head_and_food(self, head, food):
        """Calculate the angle between the snake's head and the food."""
        delta_x = food[0] - head[0]
        delta_y = food[1] - head[1]
        self.angle = np.array([np.arctan2(delta_y, delta_x)])
        return self.angle
    
    def feature_gen(self, raw_obs):
        self.set_snake_and_food_position(raw_obs)
        new_distance = self.euclidean_distance()
        distance = np.array([new_distance])
        
        head = self.snake_positions.take((0, 1))
        # recuper les voisins direct de la tete du serpent
        neighbors = self.get_neighbors(raw_obs, head) 
        angle = self.angle_between_snake_head_and_food(head, self.food_position) 

        obs = np.concatenate([neighbors, distance, angle])
        return obs
        
    def get_reward(self, score, done):
        # Calculate the Euclidean distance between the snake and the food
        new_distance = self.euclidean_distance()
        # Check if the snake has eaten food and update the reward
        if self.previous_score != score:
            reward = 10
            self.previous_score = score
        elif done:
            reward = -10
        else:
            reward =  1 if new_distance <= self._last_distance else -1
        self._last_distance = new_distance
        return reward
        
    @property
    def obs(self):
        raw_obs = self.snake_game.raw_obs
        return self.feature_gen(raw_obs)

    def step(self, action):
        raw_obs, score, done, _ = self.snake_game.step(action)
        obs = self.feature_gen(raw_obs)
        reward = self.get_reward(score, done)
        terminated = done  # done = self.snake_game.game_over
        truncated = False  # In this case, we don't have a time limit, so no
        return obs, reward, terminated, truncated, _

    def reset(self, seed=42):
        self.seed(seed)
        self._init()
        return self.obs, {}

    def render(self, mode="human"):
        if mode == "human":
            self.snake_game.render()
            
    def close(self):
        self.snake_game.quit()

if __name__ == "__main__":
    from utils import ModelRenderer
    model = ModelRenderer(name="PPO__snake",game_size=16, use_frame_stack=False, fast_game=False)
    model.render()