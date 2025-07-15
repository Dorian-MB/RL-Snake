from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from snake_env import SnakeEnv
from pathlib import Path
import numpy as np
import time


def get_env(n_envs:int=5, use_frame_stack:bool=False, n_stack:int=4, game_size:int=10, fast_game:bool=True):
    # make_vec_env handle the multiprocessing details
    env = make_vec_env(
        lambda: SnakeEnv(game_size=game_size, fast_game=fast_game),  # Use a lambda to pass parameters to the environment
        n_envs=n_envs,  
        seed=42    
    )
    if use_frame_stack:
        env = VecFrameStack(env, n_stack=n_stack, channels_order='first')
    return env


class ModelLoader:
    def __init__(self, name:str, use_frame_stack:bool=False, game_size:int=30, n_stack:int=4, fast_game:bool=True):
        name = name if name.endswith(".zip") else f"{name}.zip"
        path = Path().cwd() / "model" / name
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} does not exist. Please train the model first.")
        self.n_stack = n_stack
        self.use_frame_stack = use_frame_stack
        self.name = name
        self.game_size = game_size
        self.fast_game = fast_game

        if "PPO" in name:
            self.model = PPO.load(path, env=get_env(use_frame_stack=use_frame_stack, game_size=game_size, n_stack=n_stack, fast_game=fast_game))  
        elif "DQN" in name:
            self.model = DQN.load(path, env=get_env(use_frame_stack=use_frame_stack, game_size=game_size, n_stack=n_stack, fast_game=fast_game))  
        else:
            raise ValueError(f"Model {name} is not supported for rendering.")

class ModelRenderer(ModelLoader):
    def __init__(self, name:str, use_frame_stack:bool=False, game_size:int=30, n_stack:int=4, fast_game:bool=True):
        super().__init__(name, use_frame_stack, game_size, n_stack, fast_game=fast_game)
        self.env = SnakeEnv(game_size=game_size, fast_game=fast_game)
        
    def render(self):
        obs, _ = self.env.reset()
        terminated = False
        self.env.render()
        stacked_obs = [np.zeros_like(obs)]*self.n_stack 
        step = 0
        while not terminated:
            if "PPO" in self.name and self.use_frame_stack:
                #? not sure if its in the right order
                stacked_obs.pop(0)  # Remove the oldest frame
                stacked_obs.append(obs)  # Add the current observation
                obs = np.concatenate([obs]+stacked_obs).reshape((-1, 1)).flatten()

                
            action, _info = self.model.predict(obs, deterministic=True)
            print(f"Action taken: {action}")
            obs, reward, terminated, truncated, _ = self.env.step(action)
            step += 1
            print(f"Reward received: {reward}")
            print("distance to food:", obs.take(-2))
            print(f"step :{step}")
            self.env.render()
            time.sleep(0.1)
        self.env.close()