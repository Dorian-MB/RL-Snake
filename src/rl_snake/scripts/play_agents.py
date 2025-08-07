#!/usr/bin/env python3
"""Play script to watch trained RL Snake models in action."""
from rl_snake.environment.utils import ModelRenderer


def main():
    """Main play function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Watch a trained reinforcement learning model play Snake."
    )
    parser.add_argument(
        "-m", "--model", type=str, default="PPO_4layers64",
        help="Name of then trained model to watch."
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
        "-f", "--fast-game", action='store_true',
        help="Use fast game implementation (not recommended for visual play)."
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Loading model: {args.model}")
        
        # Create model renderer (fast_game=False for visual display)
        renderer = ModelRenderer(
            name=args.model,
            use_frame_stack=args.use_frame_stack,
            game_size=args.game_size,
            n_stack=args.n_stack,
            fast_game=args.fast_game
        )
        
        print("Starting game visualization...")
        renderer.render()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the model file exists in the models/ directory.")
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
