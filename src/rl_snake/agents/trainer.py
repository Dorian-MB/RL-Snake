"""Model training utilities and trainer class for RL agents."""

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_schedule_fn
from colorama import Fore

from ..environment.utils import get_env
from ..environment.snake_env import SnakeEnv
from .feature_extractor import LinearQNet, evaluate_model
from ..environment.utils import ModelLoader

from pathlib import Path
import time


class ModelTrainer:
    """
    Trainer class for reinforcement learning models.
    
    This class handles model creation, training, evaluation, and saving
    for different RL algorithms (PPO, DQN, A2C).
    """
    
    def __init__(self, model_name: str, 
                 load_model: bool = False,
                 fast_game: bool = True,
                 policy_kwargs=None, 
                 game_size: int = 30, 
                 n_envs: int = 5, 
                 n_stack: int = 4, 
                 use_frame_stack: bool = False,
                 verbose: int = 2):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Name/type of the model (PPO, DQN, A2C), or model path to load (default in `saved/` directory).
            load_model: Whether to load existing model
            fast_game: Whether to use fast game implementation
            policy_kwargs: Custom policy arguments
            game_size: Size of the game grid
            n_envs: Number of parallel environments
            n_stack: Number of frames to stack
            use_frame_stack: Whether to use frame stacking
            verbose: Verbosity level for training output
        """
        self.model_name = model_name
        self.fast_game = fast_game
        self.policy_kwargs = policy_kwargs
        self.game_size = game_size
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.use_frame_stack = use_frame_stack
        self.verbose = verbose
        
        # Create training environment
        self.train_env = get_env(
            use_frame_stack=use_frame_stack, 
            n_envs=n_envs, 
            n_stack=n_stack, 
            game_size=game_size
        )
        
        if load_model:
            self.model = ModelLoader(
                name=model_name, 
                use_frame_stack=use_frame_stack, 
                game_size=game_size, 
                n_stack=n_stack
            ).model
        else:
            self.model = self._get_model(model_name, policy_kwargs=policy_kwargs)
    
    def _get_model(self, model_name, policy_kwargs=None):
        """
        Create and configure the RL model.
        
        Args:
            model_name: Type of model to create
            policy_kwargs: Custom policy arguments
            
        Returns:
            Configured RL model
        """
        if model_name == "PPO":
            model = PPO(
                "MlpPolicy", 
                self.train_env,
                policy_kwargs=policy_kwargs, 
                verbose=self.verbose,
                learning_rate=get_schedule_fn(0.0003), 
                n_steps=100
            )               
        elif model_name == "DQN":
            model = DQN(
                "MlpPolicy", 
                self.train_env, 
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=1000,
                target_update_interval=500,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05
            )
        elif model_name == "A2C":
            model = A2C(
                "MlpPolicy",
                self.train_env,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                learning_rate=0.0007,
                n_steps=5
            )
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        return model
    
    def train(self, multiplicator: float = 10):
        """
        Train the model with periodic evaluation.
        
        Args:
            multiplicator: Multiplier for total training timesteps
        """
        # Configure logging (TensorBoard)
        new_logger = configure("logs", ["stdout", "tensorboard"])
        self.model.set_logger(new_logger)
        
        # Create evaluation environment
        eval_env = get_env(
            use_frame_stack=self.use_frame_stack, 
            game_size=self.game_size, 
            n_stack=self.n_stack, 
            n_envs=self.n_envs,
            fast_game=self.fast_game
        )

        total_timesteps = int(100_000 * multiplicator)
        eval_interval = 10_000
        num_eval_episodes = 5

        print(f"{Fore.CYAN}Starting training with {total_timesteps:,} timesteps{Fore.RESET}")
        
        # Training loop with periodic evaluation
        start_time = time.perf_counter()
        for step in range(0, total_timesteps, eval_interval):
            print(f"{Fore.YELLOW}Training step {step//eval_interval + 1}/{total_timesteps//eval_interval}{Fore.RESET}")
            
            self.model.learn(total_timesteps=eval_interval)
            avg_reward = evaluate_model(self.model, eval_env, num_episodes=num_eval_episodes)
            
            print(f"{Fore.GREEN}Evaluation average reward: {avg_reward:.2f}{Fore.RESET}")
            
        elapsed_time = time.perf_counter() - start_time
        print(f"{Fore.GREEN}===Training completed in {elapsed_time:.2f} seconds==={Fore.RESET}")

    def save(self, name=""):
        """
        Save the trained model.
        
        Args:
            name: Additional name suffix for the saved model
        """
        # Use new models directory
        model_dir = Path().cwd() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        save_name = f"{self.model_name}_{name}_snake.zip" if name else f"{self.model_name}_snake.zip"
        save_path = model_dir / save_name
        
        self.model.save(save_path)
        print(f"{Fore.GREEN}Model saved to: {save_path}{Fore.RESET}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model for the Snake game."
    )
    parser.add_argument(
        "-s", "--save-name", type=str, default="", 
        help="Save name for the model."
    )
    parser.add_argument(
        "-l", "--load-model", action='store_true', 
        help="Load an existing model instead of training a new one."
    )
    parser.add_argument(
        "-m", "--model", type=str, default="PPO", 
        help="Model type to train (PPO, DQN, A2C)."
    )
    parser.add_argument(
        "-f", "--fast-game", action='store_true', 
        help="Don't use the fast version of the Snake game."
    )
    parser.add_argument(
        "-g", "--game_size", type=int, default=15, 
        help="Size of the game grid (N x N)."
    )
    parser.add_argument(
        "-n", "--n-envs", type=int, default=5, 
        help="Number of parallel environments."
    )
    parser.add_argument(
        "--n_stack", type=int, default=4, 
        help="Number of frames to stack for frame stacking."
    )
    parser.add_argument(
        "--use-frame-stack", action='store_true', 
        help="Whether to use frame stacking."
    )
    parser.add_argument(
        "-u", "--use-policy-kwargs", action='store_true', 
        help="Whether to use custom policy kwargs for the model."
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1, 
        help="Verbosity level for training output."
    )
    parser.add_argument(
        "-x", "--multiplicator", type=float, default=5, 
        help="Multiplicator for total timesteps."
    )
    
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
    
    trainer.train(multiplicator=args.multiplicator)
    trainer.save(name=args.save_name)


"""
PPO Training Metrics Explanation:
    - ep_len_mean:             Average number of steps per episode
    - ep_rew_mean:             Average reward earned per episode  
    - fps:                     Environment steps processed per second
    - iterations:              Number of batches of data processed
    - time_elapsed:            Total training time in seconds
    - total_timesteps:         Total environment steps experienced
    - approx_kl:               Measure of policy change after update
    - clip_fraction:           Proportion of time clipping is used in PPO
    - clip_range:              Range for policy update clipping in PPO
    - entropy_loss:            Exploration level (higher = more exploration)
    - explained_variance:      How well value function predicts return
    - learning_rate:           Current learning rate for optimization
    - loss:                    Total combined loss being optimized
    - n_updates:               Number of updates to the model so far
    - policy_gradient_loss:    Loss from the policy gradient update
    - value_loss:              Loss in predicting expected returns
"""
