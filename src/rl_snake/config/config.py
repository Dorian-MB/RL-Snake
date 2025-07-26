"""Configuration management for RL Snake training."""

import yaml
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ModelConfig:
    """Model-related configuration."""
    name: str = "PPO"
    save_name: str = ""
    load_model: bool = False
    use_policy_kwargs: bool = False


@dataclass
class EnvironmentConfig:
    """Environment-related configuration."""
    game_size: int = 15
    fast_game: bool = True
    use_frame_stack: bool = False
    n_stack: int = 4
    n_envs: int = 5


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    multiplicator: float = 5.0
    verbose: int = 1
    progress_bar: bool = True


@dataclass
class CallbacksConfig:
    """Callbacks-related configuration."""
    enabled: bool = False
    use_progress: bool = False
    use_curriculum: bool = False
    use_metrics: bool = False
    use_save: bool = False
    curriculum_start: int = 10
    curriculum_end: int = 20
    save_freq: int = 50_000


@dataclass
class LoggingConfig:
    """Logging-related configuration."""
    log_dir: str = "logs"
    model_dir: str = "models"


@dataclass
class Config:
    """Complete configuration for RL Snake training."""
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def _resolve_progress_bar(self):
        if not self.callbacks.enabled:
            return
        if self.training.progress_bar and self.callbacks.use_progress:
            self.callbacks.use_progress = False

    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            environment=EnvironmentConfig(**data.get('environment', {})),
            training=TrainingConfig(**data.get('training', {})),
            callbacks=CallbacksConfig(**data.get('callbacks', {})),
            logging=LoggingConfig(**data.get('logging', {}))
        )
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create configuration from command line arguments."""
        return cls(
            model=ModelConfig(
                name=args.model,
                save_name=args.save_name,
                load_model=args.load_model,
                use_policy_kwargs=args.use_policy_kwargs
            ),
            environment=EnvironmentConfig(
                game_size=args.game_size,
                fast_game=not args.no_fast_game,
                use_frame_stack=args.use_frame_stack,
                n_stack=args.n_stack,
                n_envs=args.n_envs
            ),
            training=TrainingConfig(
                multiplicator=args.multiplicator,
                verbose=args.verbose,
                progress_bar=args.progress_bar
            ),
            callbacks=CallbacksConfig(
                enabled=not args.no_callbacks,
                use_progress=not args.no_progress_callback,
                use_curriculum=args.enable_curriculum,
                use_metrics=args.enable_metrics,
                use_save=not args.no_save_callback,
                curriculum_start=args.curriculum_start,
                curriculum_end=args.curriculum_end,
                save_freq=args.save_freq
            )
        )
    
    def merge_with_args(self, args: argparse.Namespace) -> 'Config':
        """Merge configuration with command line arguments (args override config)."""
        #! Problem: if provided arg is same as default values (in args parser), they will not be overridden
        
        # Only override if argument was explicitly provided
        # check if different from default
        if hasattr(args, 'model') and args.model != 'PPO':
            self.model.name = args.model
        if hasattr(args, 'save_name') and args.save_name:
            self.model.save_name = args.save_name
        if hasattr(args, 'load_model') and args.load_model:
            self.model.load_model = args.load_model
        if hasattr(args, 'use_policy_kwargs') and args.use_policy_kwargs:
            self.model.use_policy_kwargs = args.use_policy_kwargs
            
        if hasattr(args, 'game_size') and args.game_size != 15:
            self.environment.game_size = args.game_size
        if hasattr(args, 'no_fast_game') and args.no_fast_game:
            self.environment.fast_game = False
        if hasattr(args, 'use_frame_stack') and args.use_frame_stack:
            self.environment.use_frame_stack = args.use_frame_stack
        if hasattr(args, 'n_stack') and args.n_stack != 4:
            self.environment.n_stack = args.n_stack
        if hasattr(args, 'n_envs') and args.n_envs != 5:
            self.environment.n_envs = args.n_envs
            
        if hasattr(args, 'multiplicator') and args.multiplicator != 5:
            self.training.multiplicator = args.multiplicator
        if hasattr(args, 'verbose') and args.verbose != 2:
            self.training.verbose = args.verbose
        if hasattr(args, 'progress_bar') and args.progress_bar:
            self.training.progress_bar = True
            
        # Callbacks overrides
        if hasattr(args, 'no_callbacks') and args.no_callbacks:
            self.callbacks.enabled = False
        if hasattr(args, 'no_progress_callback') and args.no_progress_callback :  # default progress bar overrides custom one
            self.callbacks.use_progress = False
        if hasattr(args, 'enable_curriculum') and args.enable_curriculum:
            self.callbacks.use_curriculum = True
        if hasattr(args, 'enable_metrics') and args.enable_metrics:
            self.callbacks.use_metrics = True
        if hasattr(args, 'no_save_callback') and args.no_save_callback:
            self.callbacks.use_save = False
        if hasattr(args, 'save_freq') and args.save_freq != 50_000:
            self.callbacks.save_freq = args.save_freq
        if hasattr(args, 'curriculum_start') and args.curriculum_start != 10:
            self.callbacks.curriculum_start = args.curriculum_start
        if hasattr(args, 'curriculum_end') and args.curriculum_end != 20:
            self.callbacks.curriculum_end = args.curriculum_end
            
        return self


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with all training options."""
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model for the Snake game."
    )
    
    # Configuration file
    parser.add_argument(
        "-c", "--config", type=str, default="config/training_config.yaml",
        help="Path to configuration file."
    )
    
    # Model arguments
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
        "-u", "--use-policy-kwargs", action='store_true', 
        help="Whether to use custom policy kwargs for the model."
    )
    
    # Environment arguments
    parser.add_argument(
        "-f", "--no-fast-game", action='store_true', 
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
    
    # Training arguments
    parser.add_argument(
        "-p", "--progress-bar", action='store_true',
        help="Whether to show default SB3 progress bar during training."
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=2, 
        help="Verbosity level for training output."
    )
    parser.add_argument(
        "-x", "--multiplicator", type=float, default=5, 
        help="Multiplicator for total timesteps."
    )
    
    # Callbacks arguments
    parser.add_argument(
        "--no-callbacks", action='store_true',
        help="Disable all callbacks during training."
    )
    parser.add_argument(
        "--no-progress-callback", action='store_true',
        help="Disable custom progress callback."
    )
    parser.add_argument(
        "--enable-curriculum", action='store_true',
        help="Enable curriculum learning callback."
    )
    parser.add_argument(
        "--enable-metrics", action='store_true',
        help="Enable metrics logging callback."
    )
    parser.add_argument(
        "--no-save-callback", action='store_true',
        help="Disable model saving callback."
    )
    parser.add_argument(
        "--save-freq", type=int, default=50_000,
        help="Frequency for saving models (in timesteps)."
    )
    parser.add_argument(
        "--curriculum-start", type=int, default=10,
        help="Starting grid size for curriculum learning."
    )
    parser.add_argument(
        "--curriculum-end", type=int, default=20,
        help="Ending grid size for curriculum learning."
    )
    
    return parser


def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    """
    Load configuration from file and/or command line arguments.
    
    Args:
        config_path: Path to YAML configuration file
        args: Command line arguments (will override config file values)
        
    Returns:
        Complete configuration object
    """
    # Start with default config
    config = Config()
    
    # Load from file if provided
    if config_path and Path(config_path).exists():
        config = Config.from_yaml(config_path)
    
    # Override with command line arguments if provided
    if args:
        config = config.merge_with_args(args)
    
    config._resolve_progress_bar()  # Ensure progress bar is resolved correctly
    return config


def create_callbacks_from_config(callbacks_config: CallbacksConfig) -> List:
    """
    Create callbacks list based on configuration.
    
    Args:
        callbacks_config: Configuration for callbacks
        
    Returns:
        List of configured callbacks, or None if disabled
    """
    if not callbacks_config.enabled:
        return None
    
    # Import here to avoid circular imports
    from ..agents.callbacks import create_snake_callbacks
    
    return create_snake_callbacks(
        callbacks=[],
        use_progress=callbacks_config.use_progress,
        use_curriculum=callbacks_config.use_curriculum,
        use_metrics=callbacks_config.use_metrics,
        use_save=callbacks_config.use_save,
        curriculum_start=callbacks_config.curriculum_start,
        curriculum_end=callbacks_config.curriculum_end,
        save_freq=callbacks_config.save_freq
    )
