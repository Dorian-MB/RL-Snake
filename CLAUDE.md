# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-Snake is a reinforcement learning project implementing Snake game AI using Stable Baselines3. The project supports multiple RL algorithms (PPO, DQN, A2C) with both visual (Pygame) and fast (numpy-only) game implementations.

## Development Commands

### Testing
```bash
# Run all tests
make test
# or
python -m pytest tests/ -v

# Run tests with coverage
make test-coverage
# or
python -m pytest tests/ --cov=rl_snake --cov-report=html
```

### Code Quality
```bash
# Linting (multiple tools)
make lint

# Format code (Ruff + isort)
make ruff
# or
ruff format src/
isort src/
```

### Training Models
```bash
# Default PPO training
make train

# DQN training
make train-dqn

# Training with custom parameters
python src/rl_snake/scripts/train.py -m PPO -g 15 -x 5

# Training with configuration file
python -m src.rl_snake.agents.trainer -c config/training_config.yaml
```

### Playing and Evaluation
```bash
# Play with trained model
make play

# Evaluate model performance
make evaluate
# or
python src/rl_snake/scripts/evaluate.py -m PPO_snake -e 100
```

### Monitoring
```bash
# Start TensorBoard for training logs
make tensorboard
# or
tensorboard --logdir=logs
```

## Project Architecture

### Core Components

- **Game Engine**: Two implementations
  - `SnakeGame`: Full Pygame implementation with graphics
  - `FastSnakeGame`: Lightweight numpy-only for fast training

- **RL Environment**: `BaseSnakeEnv` and `SnakeEnv` (Gymnasium-compatible)
  - Custom feature extraction (neighbors, distance to food, angle)
  - Configurable reward system

- **Agents**: `ModelTrainer` handles model creation and training
  - Custom feature extractor: `LinearQNet`
  - Support for PPO, DQN, A2C algorithms

- **Configuration System**: YAML-based with CLI override support
  - Training configurations in `config/` directory
  - Callbacks system for enhanced monitoring

### Module Structure
```
src/rl_snake/
├── config/         # Configuration management & constants
├── game/          # Snake game implementations (visual + fast)
├── environment/   # RL environment and utilities
├── agents/        # RL agents, training, feature extraction
└── scripts/       # CLI entry points for training/playing
```

### Key Files
- [src/rl_snake/agents/trainer.py](src/rl_snake/agents/trainer.py): Main training logic
- [src/rl_snake/environment/snake_env.py](src/rl_snake/environment/snake_env.py): RL environment
- [src/rl_snake/game/](src/rl_snake/game/): Game implementations
- [config/training_config.yaml](config/training_config.yaml): Default training configuration

## Entry Points

The project provides several CLI scripts via pyproject.toml:
- `rl-snake-train`: Training models
- `rl-snake-play`: Playing with trained agents
- `rl-play-snake`: Manual snake playing
- `rl-snake-evaluate`: Model evaluation

## Development Setup

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
make install-dev
# or
pip install -e ".[dev]"
```

## Testing Strategy

- Unit tests for game logic and RL environment in `tests/`
- Coverage reports generated in `reports/coverage`
- Test markers: `slow`, `integration`, `unit`
- Run specific test types: `pytest -m "not slow"`

## Configuration

The project uses YAML configuration files with CLI override support. Key configuration areas:
- Model parameters (algorithm, architecture)
- Environment settings (game size, fast mode)
- Training parameters (timesteps, callbacks)
- Callback configuration (progress, curriculum learning, metrics)

Models are saved to `models/` directory, logs to `logs/` directory.