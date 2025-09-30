# Configuration System

RL-Snake uses a YAML-based configuration system with CLI override support for flexible training setup.

## Configuration Files

The `config/` directory contains YAML configuration files:

- `training_config.yaml`: Default training configuration
- `quick_training.yaml`: Fast training for development/testing
- `production_training.yaml`: Production-ready training with all features

## Configuration Structure

### Model Configuration

```yaml
model:
  name: "PPO"              # Algorithm: PPO, DQN, A2C
  save_name: "0"           # Model save name suffix
  load_model: false        # Load existing model to continue training
  use_policy_kwargs: false # Use custom feature extractor (LinearQNet)
```

**Supported algorithms:**
- `PPO`: Proximal Policy Optimization (recommended)
- `DQN`: Deep Q-Network
- `A2C`: Advantage Actor-Critic

### Environment Configuration

```yaml
environment:
  game_size: 16           # Grid size (NxN)
  fast_game: true         # Use FastSnakeGame (numpy-only, faster)
  use_frame_stack: false  # Stack multiple frames as input
  n_stack: 4              # Number of frames to stack
  n_envs: 5               # Number of parallel environments
```

**Game implementations:**
- `fast_game: true`: Lightweight numpy implementation (5-10x faster training)
- `fast_game: false`: Full pygame implementation (slower, but visual)

**Performance tips:**
- Use `fast_game: true` for training
- Increase `n_envs` for faster training (CPU-bound)
- Typical range: 5-16 parallel environments

### Training Configuration

```yaml
training:
  total_timesteps: 100_000  # Total training steps
  eval_interval: 10_000     # Evaluation frequency
  verbose: 1                # Logging verbosity (0=none, 1=info, 2=debug)
  progress_bar: false       # Show SB3 default progress bar
  multiplicator: 5          # Timestep multiplier
  device: "cpu"             # Device: "cpu", "cuda", "auto"
```

**Training timesteps:**
- Quick test: 10,000 - 50,000
- Basic training: 100,000 - 500,000
- Production: 1,000,000+

**Device selection:**
- `cpu`: Recommended for PPO with small networks
- `cuda`: For GPU training (limited benefit for PPO)
- `auto`: Automatically select available device

**Note on GPU training:** PPO with MlpPolicy is CPU-optimized. GPU training may not provide speedup and can cause issues with DirectML backend.

### Callbacks Configuration

```yaml
callbacks:
  enabled: true              # Enable/disable all callbacks
  use_progress: true         # Custom progress tracking
  use_curriculum: false      # Curriculum learning
  use_metrics: false         # Advanced metrics logging
  use_save: true            # Periodic model saving
  curriculum_start: 10       # Starting grid size
  curriculum_end: 20         # Ending grid size
  save_freq: 50000          # Save frequency (timesteps)
```

See [docs/CALLBACKS.md](../docs/CALLBACKS.md) for detailed callback documentation.

### Logging Configuration

```yaml
logging:
  log_dir: "logs"     # TensorBoard log directory
  model_dir: "models" # Model save directory
```

## CLI Override System

Any configuration parameter can be overridden via command-line arguments:

### Basic Usage

```bash
# Override single parameters
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --game-size 20 \
  --n-envs 8 \
  --total-timesteps 500000

# Override model type
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --model DQN \
  --save-name "DQN_experiment_1"
```

### Common Overrides

```bash
# Quick training test
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --total-timesteps 10000 \
  --eval-interval 2000

# Production training with custom policy
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --use-policy-kwargs \
  --total-timesteps 1000000 \
  --n-envs 16

# Larger grid training
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --game-size 30 \
  --total-timesteps 2000000

# Continue training from existing model
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --load-model \
  --save-name "PPO_checkpoint_v2"
```

### Callback Overrides

```bash
# Enable curriculum learning
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --enable-curriculum \
  --curriculum-start 8 \
  --curriculum-end 16

# Disable all callbacks
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --no-callbacks

# Custom save frequency
python -m src.rl_snake.agents.trainer -c config/training_config.yaml \
  --save-freq 100000
```

## Configuration Best Practices

### Development/Testing

```yaml
environment:
  game_size: 10
  n_envs: 5
  fast_game: true

training:
  total_timesteps: 50_000
  eval_interval: 10_000

callbacks:
  enabled: true
  use_progress: true
  use_save: false
```

### Production Training

```yaml
environment:
  game_size: 16
  n_envs: 12
  fast_game: true

training:
  total_timesteps: 1_000_000
  eval_interval: 50_000
  device: "cpu"

callbacks:
  enabled: true
  use_progress: true
  use_save: true
  save_freq: 100000
```

### Curriculum Learning

```yaml
environment:
  game_size: 8  # Will grow during training
  n_envs: 8

callbacks:
  enabled: true
  use_curriculum: true
  curriculum_start: 8
  curriculum_end: 20

training:
  total_timesteps: 2_000_000
```

## Custom Policy Configuration

When using `use_policy_kwargs: true`, the project uses a custom `LinearQNet` feature extractor:

```python
# Feature extractor configuration
policy_kwargs = {
    'features_extractor_class': LinearQNet,
    'features_extractor_kwargs': {
        'features_dim': 64,
        'n_layers': 3
    }
}
```

This custom architecture is saved using `.dill` serialization (see [docs/MODEL_STORAGE.md](../docs/MODEL_STORAGE.md)).

## Environment Variables

```bash
# Set CUDA device (if using GPU)
export CUDA_VISIBLE_DEVICES=0

# TensorBoard port
tensorboard --logdir=logs --port=6006
```

## Troubleshooting

### Training too slow
- Enable `fast_game: true`
- Increase `n_envs` (e.g., 8-16)
- Reduce `game_size` (e.g., 10-15)

### Models not learning
- Check `total_timesteps` (may need 500k+)
- Verify reward function in `src/rl_snake/environment/snake_env.py:222`
- Try different algorithm (PPO usually works best)

### GPU errors (DirectML)
- Set `device: "cpu"` in config
- DirectML backend doesn't support all PyTorch operations

### Configuration not loading
- Check YAML syntax (indentation, colons, quotes)
- Use `--verbose 2` for detailed logging
- Verify file path to config file

## See Also

- [Entry Points Documentation](../docs/ENTRY_POINTS.md)
- [Model Storage Documentation](../docs/MODEL_STORAGE.md)
- [Callbacks Documentation](../docs/CALLBACKS.md)
