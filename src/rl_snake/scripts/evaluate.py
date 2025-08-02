#!/usr/bin/env python3
"""Evaluation script for trained RL Snake models."""

from rl_snake.agents.feature_extractor import evaluate_model
from rl_snake.environment.utils import ModelLoader, get_env


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate a trained reinforcement learning model for the Snake game."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Name of the trained model to evaluate."
    )
    parser.add_argument(
        "-e", "--episodes", type=int, default=10,
        help="Number of episodes to evaluate."
    )
    parser.add_argument(
        "-g", "--game_size", type=int, default=16,
        help="Size of the game grid (N x N)."
    )
    parser.add_argument(
        "--use-frame-stack", action='store_true',
        help="Whether the model uses frame stacking."
    )
    parser.add_argument(
        "--n_stack", type=int, default=4,
        help="Number of frames to stack."
    )
    parser.add_argument(
        "-f", "--no-fast-game", action='store_true',
        help="Use fast game implementation."
    )
    
    args = parser.parse_args()
    
    try:
        # Load the trained model
        print(f"Loading model: {args.model}")
        model_loader = ModelLoader(
            name=args.model,
            use_frame_stack=args.use_frame_stack,
            game_size=args.game_size,
            n_stack=args.n_stack,
            fast_game=not args.no_fast_game
        )
        
        # Create evaluation environment
        eval_env = get_env(
            use_frame_stack=args.use_frame_stack,
            game_size=args.game_size,
            n_stack=args.n_stack,
            fast_game=not args.no_fast_game,
            n_envs=1  # Single environment for evaluation
        )
        
        # Evaluate the model
        print(f"Evaluating over {args.episodes} episodes...")
        avg_reward = evaluate_model(
            model_loader.model, 
            eval_env, 
            num_episodes=args.episodes
        )
        
        print(f"Average reward over {args.episodes} episodes: {avg_reward:.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        exit(1)


if __name__ == "__main__":
    main()
