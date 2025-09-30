# RL-Snake

Implementation of reinforcement learning models playing the Snake game using Stable Baselines3 and Pygame.

![Snake Agent Demo](gifs/gameplay_PPO_best_20250930_201502.gif)

## Features

- **Multiple RL Algorithms**: Support for PPO, DQN, and A2C
- **Flexible Configuration System**: YAML-based configuration with CLI override support
- **Flexible Game Engine**: Both visual (Pygame) and fast (numpy-only) implementations
- **Modern Python Structure**: Follows PEP 8 and modern packaging standards
- **Comprehensive Testing**: Unit tests for game logic and RL environment
- **Easy-to-use Scripts**: Simple command-line interface for training and evaluation
- **Pre-configured Training Profiles**: Ready-to-use configurations for different scenarios
- **GIF Recording**: Record gameplay sessions as animated GIF files

## Project Structure

```
RL-Snake/
├── src/rl_snake/           # Main package
│   ├── config/              # Configuration management & constant
│   ├── game/               # Snake game implementations
│   ├── environment/        # RL environment and utilities
│   ├── agents/             # RL agents and training utilities
│   └── scripts/            # Command-line scripts
├── config/                 # Configuration files
│   ├── training_config.yaml    # Default training configuration
│   ├── quick_training.yaml     # Fast training for development
│   └── production_training.yaml # Production-ready training
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── models/                 # Trained models
├── logs/                   # Training logs
├── gifs/                   # Generated GIF recordings
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
rl-snake-train -m PPO

# Train a DQN model 
rl-snake-train -m DQN 
```

### Playing with a Trained Model

```bash
# Watch a trained model play
rl-snake-play -m PPO_4layers64

# Use specific game settings
rl-snake-play -m PPO_4layers64 -g 20

# Record gameplay as GIF
rl-snake-play -m PPO_4layers64 --save-gif

# Record with custom filename and path
rl-snake-play -m PPO_4layers64 --save-gif --gif-path="gifs/my_agent.gif"
```

### Evaluating Model Performance

```bash
# Evaluate over 100 episodes
rl-snake-evaluate -m PPO_4layers64 -e 100
```

## Available Commands (Entry Points)

The project provides four CLI commands defined as entry points in `pyproject.toml`. These commands are automatically installed when you run `pip install -e .`:

### 1. `rl-snake-train` - Train RL Models

Train a reinforcement learning agent to play Snake.

```bash
# Basic usage
rl-snake-train

# With custom configuration
rl-snake-train -c config/my_config.yaml

# Python module equivalent
python -m rl_snake.scripts.train
```

**Entry point definition:**
```toml
rl-snake-train = "rl_snake.scripts.train:main"
```

---

### 2. `rl-snake-play` - Watch Trained Agents

Visualize a trained model playing Snake.

```bash
# Watch a specific model
rl-snake-play -m PPO_snake

# With GIF recording
rl-snake-play -m PPO_snake --save-gif

# Python module equivalent
python -m rl_snake.scripts.play_agents -m PPO_snake
```

**Entry point definition:**
```toml
rl-snake-play = "rl_snake.scripts.play_agents:main"
```

---

### 3. `rl-play-snake` - Play Snake Manually

Play Snake yourself (human control) for testing game mechanics.

```bash
# Play on default grid
rl-play-snake

# Custom grid size
rl-play-snake -g 20

# Python module equivalent
python -m rl_snake.scripts.play_snake
```

**Entry point definition:**
```toml
rl-play-snake = "rl_snake.scripts.play_snake:main"
```

**Controls:**
- Arrow keys: Move snake
- RETURN: Restart game
- SPACE: Pause/Resume

---

### 4. `rl-snake-evaluate` - Evaluate Model Performance

Run statistical evaluation of a trained model over multiple episodes.

```bash
# Evaluate with 100 episodes
rl-snake-evaluate -m PPO_snake -e 100

# Custom settings
rl-snake-evaluate -m PPO_snake -e 50 -g 20

# Python module equivalent
python -m rl_snake.scripts.evaluate -m PPO_snake -e 100
```

**Entry point definition:**
```toml
rl-snake-evaluate = "rl_snake.scripts.evaluate:main"
```

---

### Entry Points vs Python Modules

Both approaches work identically:

| Entry Point | Python Module Equivalent |
|-------------|--------------------------|
| `rl-snake-train` | `python -m rl_snake.scripts.train` |
| `rl-snake-play -m MODEL` | `python -m rl_snake.scripts.play_agents -m MODEL` |
| `rl-play-snake` | `python -m rl_snake.scripts.play_snake` |
| `rl-snake-evaluate -m MODEL` | `python -m rl_snake.scripts.evaluate -m MODEL` |

**When to use which?**
- ✅ **Entry points** (`rl-snake-*`): Easier to type, cleaner, recommended for normal use
- ✅ **Python modules** (`python -m`): Useful for debugging, scripting, or when entry points aren't installed

## Configuration System

The project uses a flexible YAML-based configuration system that allows you to control all aspects of training without modifying code.

### Using the Default Configuration

The simplest way to train is using the default configuration file:

```bash
# Train with default config (config/training_config.yaml)
rl-snake-train
```

### Using a Custom Configuration File

Point to any custom YAML configuration file:

```bash
# Use a specific config file
rl-snake-train -c config/my_custom_config.yaml
rl-snake-train --config path/to/another_config.yaml

# Use quick training profile for testing
rl-snake-train -c config/quick_training.yaml
```

### Configuration File Structure

The YAML configuration file has the following structure:

```yaml
# Model Configuration
model:
  model_type: "PPO"           # Algorithm: PPO, DQN, or A2C
  save_name: "PPO.zip"        # Name for saved model
  load_model: false           # Load existing model
  use_policy_kwargs: true     # Use custom neural network

# Environment Configuration
environment:
  game_size: 16               # Grid size (NxN)
  fast_game: true             # Use fast numpy implementation
  use_frame_stack: false      # Stack frames for temporal info
  n_stack: 4                  # Number of frames to stack
  n_envs: 5                   # Parallel environments

# Training Configuration
training:
  total_timesteps: 100_000    # Total training steps
  eval_interval: 20_000       # Evaluation frequency
  multiplicator: 5            # Multiply total_timesteps
  verbose: 1                  # Logging verbosity (0-2)
  progress_bar: false         # Show progress bar
  device: "cpu"               # Device: "cpu", "cuda", "auto"

# Callbacks Configuration
callbacks:
  enabled: true               # Enable/disable callbacks
  use_progress: true          # Custom progress tracking
  use_curriculum: false       # Curriculum learning
  use_metrics: false          # Advanced metrics
  use_save: false             # Periodic checkpointing
  curriculum_start: 10        # Curriculum starting size
  curriculum_end: 20          # Curriculum ending size
  save_freq: 50000           # Save frequency (steps)

# Logging Configuration
logging:
  log_dir: "logs"             # TensorBoard logs directory
  model_dir: "models"         # Saved models directory
```

### Overriding Configuration via CLI

You can override any configuration parameter from the command line:

```bash
# Override specific parameters
rl-snake-train -c config/training_config.yaml -m DQN -g 20 -x 10

# Train longer with more environments
rl-snake-train --total-timesteps 500000 --n-envs 10

# Use GPU and enable curriculum learning
rl-snake-train --device cuda --enable-curriculum
```

### Example: Creating a Custom Configuration

Create a new file `config/my_experiment.yaml`:

```yaml
model:
  model_type: "PPO"
  save_name: "my_experiment.zip"
  use_policy_kwargs: true

environment:
  game_size: 20
  n_envs: 8

training:
  total_timesteps: 200_000
  multiplicator: 1
  device: "auto"
```

Then use it:

```bash
rl-snake-train -c config/my_experiment.yaml
```

## Training Options (CLI Arguments)

### Core Options
- `-c, --config PATH`: Path to YAML configuration file
- `-m, --model`: Model type (PPO, DQN, A2C)
- `-g, --game_size`: Game grid size (default: 15)
- `-n, --n-envs`: Number of parallel environments (default: 5)
- `-x, --multiplicator`: Training time multiplier (default: 5)

### Environment Options
- `-f, --fast-game`: Use fast game implementation
- `--use-frame-stack`: Enable frame stacking
- `--n-stack N`: Number of frames to stack (default: 4)

### Training Options
- `--total-timesteps N`: Total training timesteps
- `--eval-interval N`: Evaluation frequency
- `--device DEVICE`: Device to use (cpu/cuda/auto)
- `--verbose N`: Verbosity level (0-2)
- `--progress-bar`: Show training progress bar

### Model Options
- `-u, --use-policy-kwargs`: Use custom neural network architecture
- `--load-model`: Load existing model to continue training
- `--save-name NAME`: Custom name for saved model

## GIF Recording Options

When using `rl-snake-play`, you can record gameplay sessions as animated GIF files:

- `--save-gif`: Enable GIF recording
- `--gif-path PATH`: Specify custom output path (auto-generated if not provided)

### GIF Recording Examples

```bash
# Basic GIF recording (auto-generated filename)
rl-snake-play -m PPO_4layers64 --save-gif

# Custom filename with timestamp
rl-snake-play -m PPO_4layers64 --save-gif --gif-path="gifs/snake_$(date +%Y%m%d_%H%M%S).gif"

# Record larger game grid
rl-snake-play -m PPO_4layers64 -g 20 --save-gif --gif-path="gifs/snake_20x20.gif"
```

GIF files are saved with 10 FPS and will automatically loop when viewed in browsers or image viewers.

## Callbacks Configuration

The training system supports configurable callbacks for enhanced monitoring and control:

### Available Callbacks

- **Progress Callback**: Custom progress tracking with episode metrics
- **Curriculum Learning**: Gradually increase game difficulty during training
- **Metrics Logging**: Advanced metrics collection and logging
- **Model Saving**: Periodic model checkpointing during training

### Callbacks in Configuration Files

```yaml
callbacks:
  enabled: true # Enable/disable all callbacks
  use_progress: true # Custom progress tracking
  use_curriculum: false # Curriculum learning
  use_metrics: false # Advanced metrics logging
  use_save: true # Periodic model saving
  curriculum_start: 10 # Starting grid size for curriculum
  curriculum_end: 20 # Ending grid size for curriculum
  save_freq: 50000 # Save frequency (timesteps)
```

### Callbacks Command Line Options

- `--no-callbacks`: Disable all callbacks
- `--no-progress-callback`: Disable custom progress callback
- `--enable-curriculum`: Enable curriculum learning
- `--enable-metrics`: Enable metrics logging
- `--no-save-callback`: Disable model saving
- `--save-freq N`: Set save frequency (default: 50000)
- `--curriculum-start N`: Set curriculum starting size (default: 10)
- `--curriculum-end N`: Set curriculum ending size (default: 20)

### Training Examples with Callbacks

```bash
# Training with all callbacks enabled
python -m src.rl_snake.agents.trainer -c config/production_training.yaml

# Training without any callbacks
python -m src.rl_snake.agents.trainer -c config/minimal_training.yaml

# Training with custom callback settings
python -m src.rl_snake.agents.trainer --enable-curriculum --curriculum-start 8 --curriculum-end 15

# Quick training without saving
python -m src.rl_snake.agents.trainer -c config/quick_training.yaml --no-save-callback
```

## Architecture

### Game Engine

- **SnakeGame**: Full pygame implementation with graphics
- **FastSnakeGame**: Lightweight numpy-only implementation for fast training

### RL Environment

- **BaseSnakeEnv**: Base Gymnasium-compatible environment, for snake game.
- **SnakeEnv**: Simple snake env
  - Custom feature extraction (neighbors, distance to food, angle)
  - Custom configurable reward system

### Agents

- **ModelTrainer**: Handles model creation and training
- **LinearQNet**: Custom feature extractor
- Evaluation utilities and model management

## Model Storage

Trained models are saved with a structured directory format that preserves both the model weights and the neural network architecture:

```
models/
└── PPO_snake/                       # Model directory
    ├── PPO_snake.zip                # SB3 model (weights + hyperparameters)
    ├── feature_extractor.dill       # Serialized custom network class
    └── feature_extractor_kwargs.json # Network architecture parameters
```

### Why This Structure?

When using custom neural networks (`use_policy_kwargs: true`), the model needs to know the exact architecture used during training. This system:

1. **Saves the model weights** (`.zip` file from Stable Baselines3)
2. **Serializes the network class** (`.dill` file) - the Python class definition itself
3. **Stores the architecture params** (`.json` file) - dimensions, layers, etc.

This ensures that:
- ✅ Models can be loaded **independently of the current code**
- ✅ You can modify `LinearQNet` without breaking old models
- ✅ Model architecture is **version-controlled with the model**

### Loading Models

The system automatically loads the correct architecture:

```bash
# Simply specify the model name
rl-snake-play -m PPO_snake

# The loader will automatically find and use:
# - models/PPO_snake/PPO_snake.zip
# - models/PPO_snake/feature_extractor.dill
# - models/PPO_snake/feature_extractor_kwargs.json
```

### Compatibility Notes

- **Models without custom policy**: Only `.zip` file is saved (standard SB3)
- **Legacy models**: Automatically detected and loaded with fallback
- **Architecture mismatch**: Clear error message with troubleshooting steps

## Monitoring Training

Training logs are saved to the `logs/` directory and can be visualized with TensorBoard:

```bash
tensorboard --logdir=logs
```

## License

MIT License - see LICENSE file for details.
