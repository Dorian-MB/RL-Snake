from stable_baselines3 import PPO,DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from colorama import Fore

from utils import get_env
from snake_env import SnakeEnv
from RL_snake import LinearQNet, evaluate_model
from utils import ModelLoader

from pathlib import Path
import time

class ModelTrainer:
    def __init__(self, model_name:str, 
                load_model:bool=False,
                fast_game:bool=True,
                policy_kwargs=None, 
                game_size:int=30, 
                n_envs:int=5, 
                n_stack:int=4, use_frame_stack:bool=False,
                verbose:int=2,
                ):
        self.model_name = model_name
        self.fast_game = fast_game
        self.policy_kwargs = policy_kwargs
        self.game_size = game_size
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.use_frame_stack = use_frame_stack
        self.verbose = verbose
        self.train_env = get_env(use_frame_stack=use_frame_stack, n_envs=n_envs, n_stack=n_stack, game_size=game_size)
        if load_model:
            self.model = ModelLoader(name=model_name, use_frame_stack=use_frame_stack, game_size=game_size, n_stack=n_stack).model
        else:
            self.model = self.get_model(model_name, policy_kwargs=policy_kwargs)
    
    def get_model(self, model_name, policy_kwargs=None):
        if model_name == "PPO":
            model = PPO("MlpPolicy", self.train_env,
                        policy_kwargs=policy_kwargs, 
                        verbose=self.verbose,
                        learning_rate=get_schedule_fn(0.0003), 
                        n_steps=100)               
        elif model_name == "DQN":
            model = DQN("MlpPolicy", self.train_env, 
                        policy_kwargs=policy_kwargs,  # Enable custom features extractor
                        verbose=self.verbose,
                        learning_rate=1e-3,            # Fixed learning rate (no schedule needed for DQN)
                        buffer_size=10000,             # Size of replay buffer
                        learning_starts=1000,          # Start learning after this many steps
                        target_update_interval=500,    # Update target network every 500 steps
                        train_freq=4,                  # Train every 4 steps
                        gradient_steps=1,              # Number of gradient steps per training
                        exploration_fraction=0.3,      # Fraction of training for exploration
                        exploration_initial_eps=1.0,   # Initial exploration probability
                        exploration_final_eps=0.05)    # Final exploration probability
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        return model
    
    def train(self, multiplicator:float=10):
        new_logger = configure("save_logs", ["stdout", "tensorboard"])
        self.model.set_logger(new_logger) # Run TensorBoard in a terminal: tensorboard --logdir=save_logs
        # IMPORTANT: Utiliser le même type d'environnement pour l'évaluation que pour l'entraînement
        eval_env = get_env(use_frame_stack=self.use_frame_stack, game_size=self.game_size, n_stack=self.n_stack, n_envs=self.n_envs)

        total_timesteps = int(100_000 * multiplicator)
        eval_interval = 10_000   # Increased interval since DQN learns differently
        n_session = total_timesteps//eval_interval
        num_eval_episodes = 5

        # Training loop with periodic evaluation
        T = time.perf_counter()
        for _ in range(0, total_timesteps, eval_interval):
            self.model.learn(total_timesteps=eval_interval)
            avg_reward = evaluate_model(self.model, eval_env, num_episodes=num_eval_episodes)
            print(f"Evaluation average reward: {avg_reward}")
        print(f"{Fore.GREEN}===Time elapsed: {time.perf_counter() - T:.2f} seconds==={Fore.RESET}")

    def save(self, name=""):
        model_dir = Path().cwd() / "model"
        model_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        save_name = f"{self.model_name}_{name}_snake.zip"
        self.model.save(model_dir / save_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model for the Snake game.")
    parser.add_argument("-s", "--save-name", type=str, default="", help="Save name for the model.")
    parser.add_argument("-l", "--load-model", action='store_true', help="Load an existing model instead of training a new one.")
    parser.add_argument("-m", "--model", type=str, default="PPO", help="Model type to train (PPO or DQN).")
    parser.add_argument("-f", "--fast-game", action='store_true', help="Dont use the fast version of the Snake game.")
    parser.add_argument("-g", "--game_size", type=int, default=15, help="Size of the game grid (N x N).")
    parser.add_argument("-n", "--n-envs", type=int, default=5, help="Number of parallel environments.")
    parser.add_argument("--n_stack", type=int, default=4, help="Number of frames to stack for frame stacking.")
    parser.add_argument("--use-frame-stack", action='store_true', help="Whether to use frame stacking.")
    parser.add_argument("-u", "--use-policy-kwargs", action='store_true', help="Whether to use custom policy kwargs for the model.")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="Verbosity level for training output.")
    parser.add_argument("-x", "--multiplicator", type=float, default=5, help="Multiplicator for total timesteps.")
    args = parser.parse_args()

    trainer = ModelTrainer(
        model_name=args.model,
        game_size=args.game_size,
        fast_game=not args.fast_game,
        n_envs=args.n_envs,
        n_stack=args.n_stack,
        load_model=args.load_model,
        use_frame_stack=args.use_frame_stack,
        policy_kwargs=dict(features_extractor_class=LinearQNet) if args.use_policy_kwargs else None,
        verbose=args.verbose
    )
    
    trainer.train(multiplicator=args.multiplicator)  # Adjust multiplicator as needed
    trainer.save(name=args.save_name)  # Save the model with


"""
POO returns :
    - ep_len_mean :             Average number of steps per episode.
    - ep_rew_mean :             Average reward earned per episode.
    - fps :                     Number of environment steps processed per second (speed of the simulation).
    - iterations :              Number of batches of data processed.
    - time_elapsed:             Total training time in seconds.
    - total_timesteps :         Total number of environment steps experienced by the agent.
    - approx_kl :               Measure of policy change after an update.
    - clip_fraction :           Proportion of time clipping is used in PPO. 
    - clip_range :              Range for policy update clipping in PPO.
    - entropy_loss :            Indicates exploration level (higher is more exploration).
    - explained_variance :      How well the value function predicts return.
    - learning_rate :           Current learning rate for optimization.
    - loss :                    Total combined loss being optimized.
    - n_updates :               Number of updates to the model so far.
    - policy_gradient_loss :    Loss from the policy gradient update.
    - value_loss :              Loss in predicting expected returns.
"""




