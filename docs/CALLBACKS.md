# Training Callbacks

RL-Snake provides custom callbacks for enhanced monitoring, curriculum learning, and metrics collection during training. This document covers all available callbacks and their configuration.

## Overview

Callbacks are functions that execute at specific points during training:

- **Training start**: Initialize progress bars, set initial difficulty
- **Each step**: Update metrics, adjust curriculum, save checkpoints
- **Training end**: Final summaries, close progress bars

## Available Callbacks

### 1. SnakeProgressCallback

Enhanced progress bar with Snake-specific metrics and colored output.

**Features:**
- Real-time training progress visualization
- Episode metrics (score, length, time)
- FPS monitoring
- Performance-based color coding
- SB3-style formatting

**File: `src/rl_snake/agents/callbacks.py:10-133`**

**Metrics displayed:**
- `score`: Average score over last 20 episodes
- `max`: Maximum score in recent episodes
- `steps`: Average episode length
- `ep_time`: Average episode duration
- `fps`: Training speed (frames per second)

**Color coding:**
- 🔴 Red: Poor performance (score < 5)
- 🟠 Orange: Medium performance (score < 20)
- 🟢 Green: Good performance (score < 50)
- 🔵 Blue: Excellent performance (score ≥ 50)

**Example output:**
```
🐍 Snake RL: 45%|████████▌         | 450k/1M [05:32<06:12, 1476steps/s, score=23.4, max=45, steps=156, ep_time=0.8s, fps=1476]
```

### 2. SnakeCurriculumCallback

Dynamic curriculum learning that randomly adjusts game difficulty during training.

**Features:**
- Random grid size changes
- Prevents overfitting to single difficulty
- Configurable size range
- Adjustable update frequency

**File: `src/rl_snake/agents/callbacks.py:136-173`**

**How it works:**
1. Training starts at random size between `min_size` and `max_size`
2. Every `update_freq` steps, randomly select new grid size
3. All parallel environments update to new size
4. Model learns to generalize across different grid sizes

**Example:**
```
Timestep    0: Grid size = 12
Timestep 1000: Grid size = 18
Timestep 2000: Grid size = 10
Timestep 3000: Grid size = 25
...
```

### 3. SnakeMetricsCallback

Log additional Snake-specific metrics to TensorBoard.

**Features:**
- Detailed episode statistics
- TensorBoard integration
- Periodic logging
- Last 100 episodes analysis

**File: `src/rl_snake/agents/callbacks.py:176-216`**

**Metrics logged:**
- `snake/avg_score`: Average score over last 100 episodes
- `snake/max_score`: Maximum score
- `snake/avg_length`: Average episode length
- `snake/max_length`: Maximum episode length

**Console output:**
```
📊 Snake Metrics (last 100 episodes):
   Avg Score: 28.45 | Max Score: 67
   Avg Length: 134.2 | Max Length: 287
```

### 4. SnakeSaveCallback

Automatically save best-performing models during training.

**Features:**
- Periodic checkpointing
- Best model tracking
- Performance-based saving
- Configurable save frequency

**File: `src/rl_snake/agents/callbacks.py:219-246`**

**Saving logic:**
1. Every `save_freq` timesteps, check recent performance
2. If mean reward improved, save new checkpoint
3. Filename includes timestep and score

**Example saves:**
```
models/snake_best_50000_score_12.3.zip
models/snake_best_150000_score_24.8.zip
models/snake_best_350000_score_41.2.zip
```

## Configuration

### YAML Configuration

**File: `config/training_config.yaml`**

```yaml
callbacks:
  enabled: true              # Master switch for all callbacks
  use_progress: true         # SnakeProgressCallback
  use_curriculum: true       # SnakeCurriculumCallback
  use_metrics: false         # SnakeMetricsCallback (disabled by default)
  use_save: false            # SnakeSaveCallback (disabled by default)
  curriculum_start: 10       # Min grid size for curriculum
  curriculum_end: 20         # Max grid size for curriculum
  save_freq: 50000           # Save frequency in timesteps
```

### CLI Overrides

```bash
# Disable all callbacks
rl-snake-train --no-callbacks

# Disable specific callback
rl-snake-train --no-progress-callback

# Enable curriculum learning
rl-snake-train --enable-curriculum --curriculum-start 8 --curriculum-end 20

# Enable metrics logging
rl-snake-train --enable-metrics

# Disable saving
rl-snake-train --no-save-callback

# Custom save frequency
rl-snake-train --save-freq 100000
```

## Usage Examples

### Example 1: Default Training (Progress Only)

```yaml
# config/training_config.yaml
callbacks:
  enabled: true
  use_progress: true
  use_curriculum: false
  use_metrics: false
  use_save: false
```

```bash
rl-snake-train -c config/training_config.yaml
```

**Output:**
```
🐍 Snake RL: 100%|██████████| 100k/100k [01:30<00:00, 1111steps/s, score=15.2, max=28, steps=98, ep_time=0.5s, fps=1111]
```

### Example 2: Training with Curriculum Learning

```yaml
# config/curriculum_training.yaml
callbacks:
  enabled: true
  use_progress: true
  use_curriculum: true
  curriculum_start: 8
  curriculum_end: 20
```

```bash
rl-snake-train -c config/curriculum_training.yaml --total-timesteps 1000000
```

**Benefits:**
- Model learns multiple grid sizes
- Better generalization
- Reduces overfitting

### Example 3: Full Monitoring Setup

```yaml
# config/production_training.yaml
callbacks:
  enabled: true
  use_progress: true
  use_curriculum: true
  use_metrics: true
  use_save: true
  curriculum_start: 10
  curriculum_end: 25
  save_freq: 100000
```

```bash
rl-snake-train -c config/production_training.yaml -t 2000000

# Monitor with TensorBoard
tensorboard --logdir=logs
```

**Features enabled:**
- ✅ Progress bar with metrics
- ✅ Random curriculum learning
- ✅ TensorBoard metrics logging
- ✅ Best model checkpointing

### Example 4: Quick Test (No Callbacks)

```bash
# Minimal training for testing
rl-snake-train --no-callbacks -t 10000 -v 2

# OR use progress bar for SB3 default
rl-snake-train --no-callbacks --progress-bar -t 10000
```

## Callback Implementation

### Creating Custom Callbacks

You can create custom callbacks by extending `BaseCallback`:

```python
from stable_baselines3.common.callbacks import BaseCallback

class MyCustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        print("Training started!")

    def _on_step(self) -> bool:
        """
        Called after each step.

        Returns:
            bool: If False, training stops
        """
        if self.num_timesteps % 1000 == 0:
            print(f"Timestep: {self.num_timesteps}")
        return True  # Continue training

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        print("Training complete!")
```

**Usage:**

```python
from rl_snake.agents.trainer import ModelTrainer
from my_callbacks import MyCustomCallback

trainer = ModelTrainer(...)
custom_callback = MyCustomCallback()
trainer.train(callbacks=[custom_callback])
```

### Combining Multiple Callbacks

**File: `src/rl_snake/agents/callbacks.py:249-291`**

```python
from rl_snake.agents.callbacks import create_snake_callbacks

callbacks = create_snake_callbacks(
    use_progress=True,
    use_curriculum=True,
    use_metrics=True,
    use_save=True,
    curriculum_start=10,
    curriculum_end=20,
    save_freq=50000
)

trainer.train(callbacks=callbacks)
```

## Advanced Usage

### 1. Early Stopping Based on Performance

```python
from stable_baselines3.common.callbacks import BaseCallback

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, target_score=50, check_freq=10000):
        super().__init__()
        self.target_score = target_score
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            if hasattr(self.model, 'ep_info_buffer'):
                recent_scores = [ep['r'] for ep in self.model.ep_info_buffer[-20:]]
                if recent_scores and np.mean(recent_scores) >= self.target_score:
                    print(f"\n✅ Target score {self.target_score} reached! Stopping.")
                    return False  # Stop training
        return True
```

### 2. Dynamic Learning Rate Adjustment

```python
class AdaptiveLRCallback(BaseCallback):
    def __init__(self, initial_lr=0.0003, check_freq=50000):
        super().__init__()
        self.initial_lr = initial_lr
        self.check_freq = check_freq
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            if hasattr(self.model, 'ep_info_buffer'):
                recent_scores = [ep['r'] for ep in self.model.ep_info_buffer[-50:]]
                current_score = np.mean(recent_scores) if recent_scores else 0

                if current_score <= self.best_score:
                    # Reduce learning rate if no improvement
                    new_lr = self.model.learning_rate * 0.9
                    self.model.learning_rate = new_lr
                    print(f"\n📉 Reduced learning rate to {new_lr:.6f}")
                else:
                    self.best_score = current_score
        return True
```

### 3. Logging to External Services

```python
import wandb

class WandBCallback(BaseCallback):
    def __init__(self, project_name="rl-snake"):
        super().__init__()
        wandb.init(project=project_name)

    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                recent_scores = [ep['r'] for ep in self.model.ep_info_buffer[-20:]]
                wandb.log({
                    "timesteps": self.num_timesteps,
                    "avg_score": np.mean(recent_scores)
                })
        return True
```

## Curriculum Learning Deep Dive

### Random Curriculum Strategy

The `SnakeCurriculumCallback` uses **random curriculum** instead of gradual increase:

**Why random?**
- Prevents overfitting to progression order
- Exposes model to diverse difficulties throughout training
- More robust generalization
- Simpler implementation

**Comparison:**

```python
# Gradual curriculum (not used)
size = min_size + (max_size - min_size) * (timestep / total_timesteps)
# Example: 10 → 11 → 12 → ... → 20

# Random curriculum (used in RL-Snake)
size = random.randint(min_size, max_size)
# Example: 10 → 18 → 12 → 25 → 10 → 15 → ...
```

### Curriculum Configuration Examples

```bash
# Small range for focused training
rl-snake-train --enable-curriculum --curriculum-start 12 --curriculum-end 16

# Wide range for diverse training
rl-snake-train --enable-curriculum --curriculum-start 8 --curriculum-end 30

# Frequent changes for aggressive curriculum
rl-snake-train --enable-curriculum --save-freq 500  # Changes every 500 steps
```

### Analyzing Curriculum Impact

```bash
# Train without curriculum
rl-snake-train -s "baseline" -g 16 -t 500000

# Train with curriculum
rl-snake-train -s "curriculum" --enable-curriculum \
  --curriculum-start 10 --curriculum-end 20 -t 500000

# Evaluate both on different sizes
for size in 10 12 14 16 18 20; do
    echo "=== Grid size: $size ==="
    rl-snake-evaluate -m PPO_baseline_snake -g $size -e 50
    rl-snake-evaluate -m PPO_curriculum_snake -g $size -e 50
done
```

## TensorBoard Integration

### Viewing Callback Metrics

```bash
# Start TensorBoard
tensorboard --logdir=logs

# Open browser to http://localhost:6006
```

**Metrics available:**
- `rollout/ep_rew_mean`: Average episode reward (SB3)
- `rollout/ep_len_mean`: Average episode length (SB3)
- `snake/avg_score`: Average score (if metrics callback enabled)
- `snake/max_score`: Maximum score
- `train/learning_rate`: Current learning rate
- `train/policy_loss`: Policy loss
- `train/value_loss`: Value loss

### Custom Logging from Callbacks

```python
class CustomMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            # Log custom metrics to TensorBoard
            self.model.logger.record('custom/my_metric', some_value)
            self.model.logger.dump(self.num_timesteps)
        return True
```

## Troubleshooting

### Issue: Progress bar not showing

**Cause:** SB3 progress bar conflicts with custom progress bar

**Solution:**
```bash
# Disable SB3 progress bar in config
rl-snake-train --no-progress-bar  # Or set progress_bar: false in YAML
```

### Issue: Curriculum learning not working

**Cause:** Environment doesn't support `set_game_size()`

**Solution:** Verify environment has the method:
```python
# In snake_env.py
def set_game_size(self, new_size):
    self.game_size = new_size
    self.snake_game.set_game_size(new_size)
```

### Issue: Metrics not appearing in TensorBoard

**Cause:** Metrics callback not enabled or wrong log directory

**Solution:**
```bash
# Enable metrics
rl-snake-train --enable-metrics

# Check log directory
tensorboard --logdir=logs --bind_all
```

### Issue: Models not being saved

**Cause:** Save callback not enabled

**Solution:**
```bash
# Enable save callback
rl-snake-train --enable-save --save-freq 50000

# Check models directory
ls models/
```

## Performance Impact

**Callback overhead:**

| Callback | Overhead | Impact |
|----------|----------|--------|
| SnakeProgressCallback | ~0.1% | Negligible |
| SnakeCurriculumCallback | ~0.5% | Very low |
| SnakeMetricsCallback | ~0.2% | Negligible |
| SnakeSaveCallback | ~5% during save | Periodic |

**Recommendations:**
- ✅ Always use `SnakeProgressCallback` (minimal overhead, great UX)
- ✅ Use `SnakeCurriculumCallback` for better generalization
- ⚠️ Use `SnakeMetricsCallback` only if detailed TensorBoard logging needed
- ⚠️ Use `SnakeSaveCallback` with reasonable `save_freq` (≥50k timesteps)

## See Also

- [Entry Points Documentation](ENTRY_POINTS.md): CLI commands
- [Configuration System](../config/README.md): YAML configuration guide
- [Main README](../README.md): Project overview
