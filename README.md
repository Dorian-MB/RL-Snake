# RL-Snake

A modern implementation of reinforcement learning models playing the Snake game using Stable Baselines3 and Pygame.

## Features

- **Multiple RL Algorithms**: Support for PPO, DQN, and A2C
- **Flexible Game Engine**: Both visual (Pygame) and fast (numpy-only) implementations
- **Modern Python Structure**: Follows PEP 8 and modern packaging standards
- **Comprehensive Testing**: Unit tests for game logic and RL environment
- **Easy-to-use Scripts**: Simple command-line interface for training and evaluation

## Project Structure

```
RL-Snake/
├── src/rl_snake/           # Main package
│   ├── config/             # Game configuration and constants
│   ├── game/               # Snake game implementations
│   ├── environment/        # RL environment and utilities
│   └── agents/             # RL agents and training utilities
├── scripts/                # Command-line scripts
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── models/                 # Trained models
├── logs/                   # Training logs
└── requirements.txt        # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dorian-MB/RL-Snake.git
cd RL-Snake
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training a Model

```bash
# Train a PPO model
python scripts/train.py -m PPO -g 15 -x 5

# Train a DQN model with custom settings
python scripts/train.py -m DQN -g 20 -n 8 -x 10
```

### Playing with a Trained Model

```bash
# Watch a trained model play
python scripts/play.py -m PPO_snake

# Use specific game settings
python scripts/play.py -m DQN_snake -g 20
```

### Evaluating Model Performance

```bash
# Evaluate over 100 episodes
python scripts/evaluate.py -m PPO_snake -e 100
```

## Training Options

- `-m, --model`: Model type (PPO, DQN, A2C)
- `-g, --game_size`: Game grid size (default: 15)
- `-n, --n-envs`: Number of parallel environments (default: 5)
- `-x, --multiplicator`: Training time multiplier (default: 5)
- `-f, --fast-game`: Use fast game implementation
- `--use-frame-stack`: Enable frame stacking
- `-u, --use-policy-kwargs`: Use custom neural network architecture

## Architecture

### Game Engine
- **SnakeGame**: Full pygame implementation with graphics
- **FastSnakeGame**: Lightweight numpy-only implementation for fast training

### RL Environment
- **SnakeEnv**: Gymnasium-compatible environment
- Custom feature extraction (neighbors, distance to food, angle)
- Configurable reward system

### Agents
- **ModelTrainer**: Handles model creation and training
- **LinearQNet**: Custom feature extractor
- Evaluation utilities and model management

## Monitoring Training

Training logs are saved to the `logs/` directory and can be visualized with TensorBoard:

```bash
tensorboard --logdir=logs
```

## License

MIT License - see LICENSE file for details.
