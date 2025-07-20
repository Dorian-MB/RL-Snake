#!/usr/bin/env python3
"""Training script for RL Snake models."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rl_snake.agents.trainer import ModelTrainer
from rl_snake.agents.feature_extractor import LinearQNet


def main():
    """Main training function."""
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
    
    print(f"Starting training with model: {args.model}")
    trainer.train(multiplicator=args.multiplicator)
    trainer.save(name=args.save_name)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
