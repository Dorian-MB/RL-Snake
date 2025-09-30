# Entry Points and CLI Commands

RL-Snake provides multiple CLI entry points for training, playing, and evaluating models. This document covers all available commands and their options.

## Overview

The project provides four main entry points (defined in `pyproject.toml`):

| Command | Purpose | Script Location |
|---------|---------|----------------|
| `rl-snake-train` | Train RL models | `rl_snake.scripts.train:main` |
| `rl-snake-play` | Watch trained models play | `rl_snake.scripts.play_agents:main` |
| `rl-play-snake` | Play Snake manually | `rl_snake.scripts.play_snake:main` |
| `rl-snake-evaluate` | Evaluate model performance | `rl_snake.scripts.evaluate:main` |

## Installation

After installing the package with `pip install -e .`, all entry points become available as command-line tools:

```bash
# Entry points (recommended)
rl-snake-train --help
rl-snake-play --model PPO_snake

# Alternative: Direct Python module execution
python -m src.rl_snake.agents.trainer --help
python src/rl_snake/scripts/play_agents.py --model PPO_snake
```

## 1. Training Models: `rl-snake-train`

Train reinforcement learning models to play Snake.

### Basic Usage

```bash
# Train with YAML configuration
rl-snake-train -c config/training_config.yaml

# Alternative direct execution
python -m src.rl_snake.agents.trainer -c config/training_config.yaml
python src/rl_snake/scripts/train.py -m PPO -g 15 -x 5
```

### Command-Line Options

#### Configuration

- `-c, --config PATH`: Path to YAML configuration file (default: `config/training_config.yaml`)

#### Model Options

- `-m, --model TYPE`: Model algorithm: `PPO`, `DQN`, `A2C` (default: `PPO`)
- `-s, --save-name NAME`: Save name suffix for the model (default: `""`)
- `-l, --load-model`: Load existing model to continue training
- `-u, --use-policy-kwargs`: Use custom `LinearQNet` feature extractor

#### Environment Options

- `-g, --game_size N`: Grid size (NxN) (default: `15`)
- `-n, --n-envs N`: Number of parallel environments (default: `5`)
- `-f, --no-fast-game`: Use pygame implementation instead of fast numpy version
- `--use-frame-stack`: Enable frame stacking
- `--n_stack N`: Number of frames to stack (default: `4`)

#### Training Options

- `-t, --total-timesteps N`: Total training timesteps (default: `10000`)
- `-x, --multiplicator X`: Multiply total timesteps by X (default: `1`)
- `--eval-interval N`: Evaluation interval in timesteps (default: `10000`)
- `-v, --verbose LEVEL`: Verbosity (0=silent, 1=info, 2=debug) (default: `1`)
- `-p, --progress-bar`: Show default SB3 progress bar

#### Callback Options

- `--no-callbacks`: Disable all callbacks
- `--no-progress-callback`: Disable custom progress callback
- `--enable-curriculum`: Enable curriculum learning
- `--enable-metrics`: Enable advanced metrics logging
- `--no-save-callback`: Disable periodic model saving
- `--save-freq N`: Save frequency in timesteps (default: `50000`)
- `--curriculum-start N`: Starting grid size for curriculum (default: `10`)
- `--curriculum-end N`: Ending grid size for curriculum (default: `20`)

### Training Examples

```bash
# Quick training test
rl-snake-train -m PPO -g 10 -t 50000

# Production training with custom policy
rl-snake-train -c config/training_config.yaml \
  --use-policy-kwargs \
  --total-timesteps 1000000 \
  --n-envs 12

# DQN training with larger grid
rl-snake-train -m DQN -g 20 -x 10 --n-envs 8

# Training with curriculum learning
rl-snake-train --enable-curriculum \
  --curriculum-start 8 \
  --curriculum-end 20 \
  --total-timesteps 2000000

# Continue training from checkpoint
rl-snake-train --load-model \
  --save-name "PPO_checkpoint_v2" \
  --total-timesteps 500000

# Quick test without saving
rl-snake-train -t 10000 --no-save-callback

# Override YAML config parameters
rl-snake-train -c config/production_training.yaml \
  --game-size 30 \
  --n-envs 16 \
  --save-freq 100000
```

## 2. Playing with Trained Models: `rl-snake-play`

Watch trained RL agents play Snake with visual display.

### Basic Usage

```bash
# Play with default model
rl-snake-play

# Play specific model
rl-snake-play -m PPO_snake

# Alternative execution
python src/rl_snake/scripts/play_agents.py -m PPO_snake
```

### Command-Line Options

- `-m, --model NAME`: Model name (default: `PPO_1_snake`)
- `-g, --game_size N`: Grid size (default: `16`)
- `--use-frame-stack`: Enable frame stacking (must match training)
- `--n_stack N`: Number of frames to stack (default: `4`)
- `-f, --fast-game`: Use fast implementation (not recommended for visual play)

### Play Examples

```bash
# Watch PPO model on 16x16 grid
rl-snake-play -m PPO_snake -g 16

# Watch DQN model on larger grid
rl-snake-play -m DQN_large -g 30

# Play model trained with frame stacking
rl-snake-play -m PPO_framestack \
  --use-frame-stack \
  --n_stack 4 \
  -g 20

# Play model with custom architecture
rl-snake-play -m PPO_4layers64 -g 16
```

### Troubleshooting Play Mode

**Model not found:**
```bash
# Check model exists in models/ directory
ls models/PPO_snake/

# Expected structure:
models/PPO_snake/
├── PPO_snake.zip
├── feature_extractor.dill  # If custom policy used
└── feature_extractor_kwargs.json
```

**Game size mismatch:**
- Use the same `--game_size` as during training
- Check training logs for correct size

**Architecture mismatch:**
- See [docs/MODEL_STORAGE.md](MODEL_STORAGE.md) for `.dill` architecture details

## 3. Manual Play: `rl-play-snake`

Play Snake manually using keyboard controls.

### Basic Usage

```bash
# Play with default settings
rl-play-snake

# Play with custom grid size
rl-play-snake --game_size 30

# Alternative execution
python src/rl_snake/scripts/play_snake.py --game_size 20
```

### Command-Line Options

- `--game_size N`: Grid size (NxN) (default: `30`)

### Keyboard Controls

- **Arrow Keys**: Move snake (Up, Down, Left, Right)
- **Space**: Pause/Resume game
- **Enter**: Restart game after game over
- **Close Window**: Quit game

### Manual Play Examples

```bash
# Small grid for quick games
rl-play-snake --game_size 10

# Large grid for challenge
rl-play-snake --game_size 40

# Standard 30x30 grid
rl-play-snake
```

## 4. Evaluating Models: `rl-snake-evaluate`

Quantitatively evaluate trained model performance over multiple episodes.

### Basic Usage

```bash
# Evaluate with default settings
rl-snake-evaluate -m PPO_snake

# Evaluate over 100 episodes
rl-snake-evaluate -m PPO_snake -e 100

# Alternative execution
python src/rl_snake/scripts/evaluate.py -m PPO_snake -e 100
```

### Command-Line Options

- `-m, --model NAME`: Model name (required)
- `-e, --episodes N`: Number of evaluation episodes (default: `10`)
- `-g, --game_size N`: Grid size (default: `16`)
- `--use-frame-stack`: Enable frame stacking
- `--n_stack N`: Number of frames to stack (default: `4`)
- `-f, --no-fast-game`: Don't use fast game (slower evaluation)

### Evaluation Examples

```bash
# Quick evaluation (10 episodes)
rl-snake-evaluate -m PPO_snake -e 10

# Comprehensive evaluation (100 episodes)
rl-snake-evaluate -m PPO_snake -e 100

# Evaluate on different grid size than training
rl-snake-evaluate -m PPO_small -g 30 -e 50

# Evaluate model with frame stacking
rl-snake-evaluate -m PPO_framestack \
  --use-frame-stack \
  --n_stack 4 \
  -e 50

# Fast evaluation (uses FastSnakeGame)
rl-snake-evaluate -m PPO_snake -e 100

# Slow evaluation (uses pygame)
rl-snake-evaluate -m PPO_snake -e 100 --no-fast-game
```

### Understanding Evaluation Output

```bash
$ rl-snake-evaluate -m PPO_snake -e 100

Loading model: PPO_snake
Evaluating over 100 episodes...
Average reward over 100 episodes: 45.32
```

**Interpreting results:**
- Average reward correlates with game score (food eaten)
- Typical ranges:
  - Poor: < 0 (dying quickly)
  - Learning: 0-20
  - Good: 20-50
  - Excellent: 50-100
  - Expert: 100+

## Using with Makefile

The project includes convenient Make targets:

```bash
# Training
make train          # Train PPO with default config
make train-dqn      # Train DQN

# Playing
make play           # Watch trained model play

# Evaluation
make evaluate       # Evaluate trained model

# Monitoring
make tensorboard    # Start TensorBoard server

# Testing
make test           # Run test suite
make lint           # Code quality checks
```

See `Makefile` for all available targets.

## Common Workflows

### 1. Train → Play → Evaluate

```bash
# 1. Train a model
rl-snake-train -m PPO -g 16 -t 500000 -s "experiment_1"

# 2. Watch it play
rl-snake-play -m PPO_experiment_1_snake -g 16

# 3. Quantitative evaluation
rl-snake-evaluate -m PPO_experiment_1_snake -e 100 -g 16
```

### 2. Curriculum Learning Workflow

```bash
# Train with curriculum (grid grows during training)
rl-snake-train --enable-curriculum \
  --curriculum-start 8 \
  --curriculum-end 20 \
  --total-timesteps 2000000 \
  -s "curriculum_v1"

# Evaluate at different grid sizes
rl-snake-evaluate -m PPO_curriculum_v1_snake -g 10 -e 50
rl-snake-evaluate -m PPO_curriculum_v1_snake -g 15 -e 50
rl-snake-evaluate -m PPO_curriculum_v1_snake -g 20 -e 50
```

### 3. Hyperparameter Experimentation

```bash
# Create custom config
cp config/training_config.yaml config/my_experiment.yaml
# Edit my_experiment.yaml with your settings

# Train with custom config
rl-snake-train -c config/my_experiment.yaml

# Compare with baseline
tensorboard --logdir=logs
```

### 4. Continue Training from Checkpoint

```bash
# Initial training
rl-snake-train -m PPO -t 500000 -s "checkpoint_v1"

# Continue training
rl-snake-train --load-model \
  -s "checkpoint_v1" \
  -t 500000
```

## Python API Usage

All entry points can also be called directly from Python:

```python
from rl_snake.agents.trainer import ModelTrainer
from rl_snake.environment.utils import ModelRenderer, ModelLoader
from rl_snake.config.config import load_config

# Training
config = load_config("config/training_config.yaml")
trainer = ModelTrainer.from_config(config)
trainer.train(total_timesteps=100000)
trainer.save("my_model")

# Playing
renderer = ModelRenderer(name="PPO_snake", game_size=16, fast_game=False)
renderer.render()

# Evaluation
from rl_snake.agents.feature_extractor import evaluate_model
loader = ModelLoader(name="PPO_snake", game_size=16)
avg_reward = evaluate_model(loader.model, loader.env, num_episodes=100)
```

## See Also

- [Configuration System](../config/README.md): Detailed YAML configuration guide
- [Model Storage](MODEL_STORAGE.md): Model architecture persistence with `.dill`
- [Callbacks](CALLBACKS.md): Training callbacks and curriculum learning
- [Main README](../README.md): Project overview and quick start
