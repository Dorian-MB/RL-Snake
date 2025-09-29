"""Model training utilities and trainer class for RL agents."""

import time
from pathlib import Path

import numpy as np
import torch
from colorama import Fore
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import FloatSchedule, LinearSchedule
from tqdm.auto import tqdm

from ..config.config import (
    Config,
    create_argument_parser,
    create_callbacks_from_config,
    load_config,
)
from .feature_extractor import LinearQNet
from .utils import Logger, ModelLoader, get_env


class ModelTrainer:
    """
    Trainer class for reinforcement learning models.

    This class handles model creation, training, evaluation, and saving
    for different RL algorithms (PPO, DQN, A2C).
    """

    def __init__(
        self,
        model_type: str,
        model_name: str = "PPO_snake.zip",
        load_model: bool = False,
        fast_game: bool = True,
        callback_list: list = [],
        policy_kwargs=None,
        game_size: int = 30,
        n_envs: int = 5,
        n_stack: int = 4,
        use_frame_stack: bool = False,
        progress_bar: bool = True,
        config: Config = None,
        device: str = "auto",
        verbose: int = 0,
    ):
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
        self.model_type = model_type
        self.model_name = model_name
        self.fast_game = fast_game
        self.policy_kwargs = policy_kwargs
        self.game_size = game_size
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.use_frame_stack = use_frame_stack
        self.load_model = load_model
        self.progress_bar = progress_bar
        self.verbose = verbose
        self.callback_list = callback_list
        self.config = config
        self.logger = Logger()

        # Create training environment
        self.train_env = get_env(
            use_frame_stack=use_frame_stack,
            n_envs=n_envs,
            n_stack=n_stack,
            game_size=game_size,
        )

        self.device = (
            self._select_device()
            if (device == "auto" or device is None)
            else torch.device(device)
        )
        if load_model:
            self.model = ModelLoader(
                name=model_name,
                use_frame_stack=use_frame_stack,
                game_size=game_size,
                n_stack=n_stack,
                n_envs=n_envs,
                fast_game=fast_game,
            ).model
        else:
            self.model = self._get_model(model_type, policy_kwargs=policy_kwargs)
        self._set_log_interval(model_type)

    def _select_device(self):
        # DirectML (AMD/Intel/Nvidia via DML) en priorité sur Windows
        try:
            import torch_directml

            return torch_directml.device()
        except Exception:
            pass
        # Sinon CUDA si dispo, sinon CPU
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_config(cls, config: Config) -> "ModelTrainer":
        """
        Create ModelTrainer from configuration object.

        Args:
            config: Complete configuration object

        Returns:
            Configured ModelTrainer instance
        """
        # Create callbacks from configuration
        callback_list = create_callbacks_from_config(config.callbacks)
        return cls(
            model_type=config.model.model_type,
            model_name=config.model.save_name,
            load_model=config.model.load_model,
            fast_game=config.environment.fast_game,
            callback_list=callback_list,
            # Improved policy_kwargs with optimal architecture for Snake
            policy_kwargs=dict(
                features_extractor_class=LinearQNet,
                features_extractor_kwargs=dict(
                    features_dim=64, n_layers=3
                ),  # 64->64->64 for better abstraction
            )
            if config.model.use_policy_kwargs
            else None,
            game_size=config.environment.game_size,
            n_envs=config.environment.n_envs,
            n_stack=config.environment.n_stack,
            use_frame_stack=config.environment.use_frame_stack,
            progress_bar=config.training.progress_bar,
            config=config,
            device=config.training.device,
            verbose=config.training.verbose,
        )

    def _get_model(self, model_type, policy_kwargs=None):
        """
        Create and configure the RL model.

        Args:
            model_name: Type of model to create
            policy_kwargs: Custom policy arguments

        Returns:
            Configured RL model
        """
        if model_type == "PPO":
            # Utilisation moderne avec FloatSchedule et LinearSchedule de SB3
            lr_schedule = FloatSchedule(
                LinearSchedule(start=0.0003, end=0.00001, end_fraction=1.0)
            )

            model = PPO(
                "MlpPolicy",
                self.train_env,
                device=self.device,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                learning_rate=lr_schedule,  # Approche moderne SB3
                n_steps=100,  # Number of steps per update
                batch_size=100,  # Mini-batch size, warning: n_steps*n_envs % batch_size must be divisible, otherwise training can be inconsistent
            )
        elif model_type == "DQN":
            # DQN avec les nouvelles classes SB3
            lr_schedule = FloatSchedule(
                LinearSchedule(start=1e-3, end=1e-5, end_fraction=1.0)
            )

            model = DQN(
                "MlpPolicy",
                self.train_env,
                device=self.device,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                learning_rate=lr_schedule,  # Moderne et propre
                buffer_size=10000,
                learning_starts=1000,
                target_update_interval=500,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
            )
        elif model_type == "A2C":
            # A2C avec FloatSchedule
            lr_schedule = FloatSchedule(
                LinearSchedule(start=0.0007, end=0.00007, end_fraction=1.0)
            )

            model = A2C(
                "MlpPolicy",
                self.train_env,
                device=self.device,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                learning_rate=lr_schedule,
                n_steps=5,
            )
        else:
            raise ValueError(f"Model {model_type} is not supported.")
        return model

    def _set_log_interval(self, model_type):
        if model_type == "PPO":
            self.log_interval = 1
        elif model_type == "DQN":
            self.log_interval = 4
        elif model_type == "A2C":
            self.log_interval = 100

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_interval: int = 10_000,
        multiplicator: float = 1,
    ):
        """
        Train the model with periodic evaluation.

        Args:
            total_timesteps: Total number of timesteps for training
            multiplicator: Multiplier for total training timesteps
        """
        logs_path = Path().cwd() / "logs" / self.model_name.split(".")[0]
        logs_path.mkdir(parents=True, exist_ok=True)
        # Configure logging (TensorBoard) access with: tensorboard --logdir logs
        new_logger = configure(
            str(logs_path),
            ["stdout", "tensorboard"] if self.verbose > 1 else ["tensorboard"],
        )
        self.model.set_logger(new_logger)

        if isinstance(self.config, Config):
            total_timesteps = self.config.training.total_timesteps
            eval_interval = self.config.training.eval_interval
            multiplicator = self.config.training.multiplicator
            if self.verbose >= 1:
                self.logger.info(f"{Fore.CYAN}{self.config}{Fore.RESET}")

        # Create evaluation environment
        eval_env = get_env(
            use_frame_stack=self.use_frame_stack,
            game_size=self.game_size,
            n_stack=self.n_stack,
            n_envs=self.n_envs,
            fast_game=self.fast_game,
        )

        total_timesteps = int(total_timesteps * multiplicator)
        eval_interval = int(eval_interval * multiplicator)
        num_eval_episodes = 5

        self.logger.info(
            f"{Fore.CYAN}Starting training with {total_timesteps:,} timesteps{Fore.RESET}"
        )

        # Training loop with periodic evaluation
        start_time = time.perf_counter()
        for step in range(0, total_timesteps, eval_interval):
            self.logger.info(
                f"{Fore.YELLOW}Training step {step // eval_interval + 1}/{total_timesteps // eval_interval}{Fore.RESET}"
            )
            self.model.learn(
                total_timesteps=eval_interval,
                reset_num_timesteps=False,
                log_interval=self.log_interval,
                progress_bar=self.progress_bar,
                callback=self.callback_list,
            )
            avg_reward = list(
                np.round(
                    self.evaluate_model(eval_env, num_episodes=num_eval_episodes), 2
                )
            )

            self.logger.info(
                f"{Fore.GREEN}Evaluation average reward: {avg_reward}{Fore.RESET}"
            )

        elapsed_time = time.perf_counter() - start_time
        self.logger.info(
            f"{Fore.GREEN}===Training completed in {elapsed_time:.2f} seconds==={Fore.RESET}"
        )

    def evaluate_model(self, eval_env, num_episodes=10):
        """
        Evaluate model's performance.

        Args:
            eval_env: Environment for evaluation
            num_episodes: Number of episodes to evaluate

        Returns:
            Average reward across episodes
        """
        all_rewards = []

        for _ in tqdm(range(num_episodes), desc="Evaluating", total=num_episodes):
            obs = eval_env.reset()
            # Handle different environment return formats
            if isinstance(obs, tuple):
                obs = obs[0]

            terminated = np.array([False])
            total_rewards = 0
            # TODO REFACTOR
            while not terminated.all():
                # Get action from model
                action, _states = self.model.predict(obs, deterministic=True)

                # Take step in environment
                step_result = eval_env.step(action)

                # Handle different return formats (gym vs gymnasium)
                if len(step_result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = step_result
                    terminated = terminated or truncated
                    # Ensure terminated is a list-like (list or np.ndarray)
                    if not isinstance(terminated, (list, np.ndarray)):
                        terminated = [terminated]
                else:  # Gym format
                    obs, reward, terminated, info = step_result
                    if not isinstance(terminated, (list, np.ndarray)):
                        terminated = (
                            [terminated]
                            if not isinstance(terminated, list)
                            else terminated
                        )

                total_rewards += reward

                # print(f"Step reward: {reward}, Total rewards: {total_rewards}")
                # print(f"Step terminated: {terminated}")

            total_rewards = (
                total_rewards
                if isinstance(total_rewards, np.ndarray)
                else np.array(total_rewards)
            )
            all_rewards.append(total_rewards)

        average_reward = np.sum(all_rewards, axis=1) / num_episodes
        return average_reward

    def save(self, save_name=""):
        """
        Save the trained model.

        Args:
            name: Additional name suffix for the saved model
        """
        # Use new models directory
        model_dir = Path().cwd() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        save_name = save_name or self.model_name
        save_name = save_name if save_name.endswith(".zip") else f"{save_name}.zip"
        save_path = model_dir / save_name

        self.model.save(save_path)
        self.logger.info(f"{Fore.GREEN}Model saved to: {save_path}{Fore.RESET}")


def main():
    """Main function to run the model trainer."""
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Load configuration (file + command line override)
    config = load_config(config_path="config/training_config.yaml", args=args)

    # Create trainer from configuration
    trainer = ModelTrainer.from_config(config)

    # Train and save model
    trainer.train()
    trainer.save(save_name=config.model.save_name)


if __name__ == "__main__":
    main()


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
