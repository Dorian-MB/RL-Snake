# Model Storage and Architecture Persistence

RL-Snake uses a sophisticated model storage system that persists not just model weights, but also the neural network architecture using `.dill` serialization. This enables models to remain loadable even after code changes.

## Problem: Architecture Dependency

Traditional Stable Baselines3 model saving only stores:
- Model weights (`.zip` file)
- Policy hyperparameters

**This creates a problem:** If you modify the feature extractor class (e.g., change `LinearQNet` layer structure), old models fail to load:

```python
# Original training: LinearQNet with 3 layers
model.save("models/PPO_snake.zip")

# Later: You modify LinearQNet to have 4 layers
model = PPO.load("models/PPO_snake.zip")  # ❌ Fails! Architecture mismatch
```

**Error:**
```
RuntimeError: Error(s) in loading state_dict for MlpPolicy:
    Missing key(s) in state_dict: "features_extractor.linear.6.weight", ...
```

## Solution: `.dill` Architecture Persistence

RL-Snake solves this by saving the **Python class definition** itself using `dill` serialization:

```
models/PPO_snake/
├── PPO_snake.zip                       # SB3 model weights
├── feature_extractor.dill              # Serialized LinearQNet class
└── feature_extractor_kwargs.json      # Architecture hyperparameters
```

This ensures the **exact class used during training** is restored during loading, regardless of current code state.

## How It Works

### 1. During Training (Saving)

When you train with a custom policy (`--use-policy-kwargs`), the trainer saves three files:

**File: `src/rl_snake/agents/trainer.py:263-289`**

```python
def save(self, save_name=""):
    save_name = save_name.replace(".zip", "")

    # Create model directory
    model_dir = Path().cwd() / "models" / save_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save SB3 model weights
    model_path = model_dir / f"{save_name}.zip"
    self.model.save(model_path)

    # 2. Save feature extractor class with dill
    if self.policy_kwargs and 'features_extractor_class' in self.policy_kwargs:
        import dill, json

        class_path = model_dir / "feature_extractor.dill"
        with open(class_path, 'wb') as f:
            dill.dump(self.policy_kwargs['features_extractor_class'], f)

        # 3. Save architecture kwargs
        if 'features_extractor_kwargs' in self.policy_kwargs:
            kwargs_path = model_dir / "feature_extractor_kwargs.json"
            with open(kwargs_path, 'w') as f:
                json.dump(self.policy_kwargs['features_extractor_kwargs'], f, indent=2)
```

### 2. During Inference (Loading)

When loading a model, the system reconstructs the exact architecture:

**File: `src/rl_snake/environment/utils.py:43-88`**

```python
class ModelLoader:
    def __init__(self, name: str, ...):
        base_name = name.replace(".zip", "")
        model_dir = Path().cwd() / "models" / base_name
        path = model_dir / f"{base_name}.zip"

        # Check for .dill architecture file
        custom_objects = None
        class_path = model_dir / "feature_extractor.dill"

        if class_path.exists():
            import dill, json

            # Load the saved class definition
            with open(class_path, 'rb') as f:
                extractor_class = dill.load(f)

            # Load architecture kwargs
            kwargs_path = model_dir / "feature_extractor_kwargs.json"
            extractor_kwargs = {}
            if kwargs_path.exists():
                with open(kwargs_path, 'r') as f:
                    extractor_kwargs = json.load(f)

            # Reconstruct policy_kwargs
            custom_objects = {
                "policy_kwargs": {
                    "features_extractor_class": extractor_class,
                    "features_extractor_kwargs": extractor_kwargs
                }
            }

        # Load model with custom architecture
        if "PPO" in base_name:
            self.model = PPO.load(path, env=env, custom_objects=custom_objects)
        elif "DQN" in base_name:
            self.model = DQN.load(path, env=env, custom_objects=custom_objects)
```

## Model Directory Structure

### New Structure (Recommended)

```
models/
├── PPO_snake/
│   ├── PPO_snake.zip                   # SB3 weights
│   ├── feature_extractor.dill          # Class definition
│   └── feature_extractor_kwargs.json   # Architecture params
├── DQN_large/
│   ├── DQN_large.zip
│   ├── feature_extractor.dill
│   └── feature_extractor_kwargs.json
└── PPO_experiment_v2/
    └── PPO_experiment_v2.zip           # MlpPolicy (no custom arch)
```

### Legacy Structure (Still Supported)

```
models/
├── PPO_snake.zip                       # Old flat structure
├── DQN_snake.zip
└── A2C_snake.zip
```

The loader automatically detects and handles both structures with fallback support.

## Custom Policy Architecture

### Feature Extractor: `LinearQNet`

When using `--use-policy-kwargs`, models use the `LinearQNet` feature extractor:

**File: `src/rl_snake/agents/feature_extractor.py:12-49`**

```python
class LinearQNet(BaseFeaturesExtractor):
    """
    Custom feature extractor for RL Snake.

    Args:
        observation_space: Gym observation space
        features_dim: Output dimension (default: 64)
        n_layers: Number of hidden layers (default: 3)
    """

    def __init__(self, observation_space, features_dim=64, n_layers=3):
        super(LinearQNet, self).__init__(observation_space, features_dim)

        n_flatten = int(np.prod(observation_space.shape))

        # Build sequential network
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(observations)
```

**Saved JSON: `feature_extractor_kwargs.json`**

```json
{
  "features_dim": 64,
  "n_layers": 3
}
```

## Usage Examples

### Training with Custom Policy

```bash
# Train with LinearQNet feature extractor
rl-snake-train -m PPO --use-policy-kwargs -s "custom_arch"

# Results in:
# models/PPO_custom_arch_snake/
#   ├── PPO_custom_arch_snake.zip
#   ├── feature_extractor.dill
#   └── feature_extractor_kwargs.json
```

### Loading Saved Models

```bash
# Automatically loads .dill architecture
rl-snake-play -m PPO_custom_arch_snake

# Python API
from rl_snake.environment.utils import ModelLoader

loader = ModelLoader(name="PPO_custom_arch_snake", game_size=16)
model = loader.model  # Fully reconstructed with original architecture
```

### Handling Legacy Models

If you have old models trained before the `.dill` system:

#### Option 1: Manual `.dill` Creation

```python
import dill
import json
from pathlib import Path
from src.rl_snake.agents.feature_extractor import LinearQNet

# Model information
model_name = "PPO_old_model"
model_dir = Path("models") / model_name
model_dir.mkdir(exist_ok=True)

# Save class definition
with open(model_dir / "feature_extractor.dill", 'wb') as f:
    dill.dump(LinearQNet, f)

# Save kwargs (adjust if needed)
kwargs = {
    "features_dim": 64,
    "n_layers": 3
}
with open(model_dir / "feature_extractor_kwargs.json", 'w') as f:
    json.dump(kwargs, f, indent=2)

# Move old .zip into directory
old_path = Path("models") / f"{model_name}.zip"
new_path = model_dir / f"{model_name}.zip"
old_path.rename(new_path)
```

#### Option 2: Retrain with New System

```bash
# Simply retrain with --use-policy-kwargs
rl-snake-train -m PPO --use-policy-kwargs -s "my_model_v2"
```

## Architecture Mismatch Errors

When .dill files are present but architectures don't match, you get a helpful error:

```python
❌ INCOMPATIBILITY ERROR ❌

There is a mismatch between saved files in: models/PPO_snake/

The model architecture in the code has changed and is incompatible
with the saved model weights.

Possible solutions:
1. Use the correct model architecture matching the saved weights
2. Delete the incompatible saved model files and retrain
3. Check if LinearQNet class has been modified since training

Files found:
  - models/PPO_snake/PPO_snake.zip (weights)
  - models/PPO_snake/feature_extractor.dill (architecture)
  - models/PPO_snake/feature_extractor_kwargs.json (hyperparameters)
```

### Resolution Strategies

1. **Use correct architecture**: Restore the original `LinearQNet` class
2. **Delete and retrain**: Remove incompatible files and train fresh
3. **Version control**: Use git to restore old architecture:
   ```bash
   git checkout HEAD~10 -- src/rl_snake/agents/feature_extractor.py
   ```

## MlpPolicy vs Custom Policy

### MlpPolicy (Default)

```bash
# Training WITHOUT custom policy
rl-snake-train -m PPO -s "default_model"

# Creates: models/PPO_default_model_snake/PPO_default_model_snake.zip
# No .dill files needed - uses SB3's built-in MlpPolicy
```

**Advantages:**
- Simpler storage (single .zip file)
- Always compatible with SB3 updates
- No architecture versioning concerns

**Disadvantages:**
- Less control over architecture
- Cannot customize network structure

### Custom Policy (LinearQNet)

```bash
# Training WITH custom policy
rl-snake-train -m PPO --use-policy-kwargs -s "custom_model"

# Creates: models/PPO_custom_model_snake/
#   ├── PPO_custom_model_snake.zip
#   ├── feature_extractor.dill
#   └── feature_extractor_kwargs.json
```

**Advantages:**
- Full control over architecture
- Can customize layers, activations, dimensions
- Architecture persisted with `.dill`

**Disadvantages:**
- More complex storage
- Must maintain `.dill` files
- Architecture changes require care

## Best Practices

### 1. Consistent Naming Convention

```bash
# Include architecture info in model name
rl-snake-train --use-policy-kwargs -s "PPO_3layer64_v1"
rl-snake-train --use-policy-kwargs -s "PPO_4layer128_v2"
```

### 2. Document Architecture Changes

```python
# In feature_extractor.py, add comments:
class LinearQNet(BaseFeaturesExtractor):
    """
    Custom feature extractor for RL Snake.

    Architecture History:
    - v1 (2025-01): 3 layers, 64 units
    - v2 (2025-02): 4 layers, 64 units, added dropout
    - v3 (2025-03): 3 layers, 64 units, removed dropout (reverted)
    """
```

### 3. Use Version Control

```bash
# Tag model releases
git tag -a model_PPO_v1 -m "PPO with 3-layer LinearQNet"
git push --tags
```

### 4. Archive Old Models

```bash
# Create archive directory
mkdir -p models/archive

# Move old versions
mv models/PPO_old/ models/archive/PPO_old_2025-01/
```

### 5. Test After Architecture Changes

```bash
# After modifying LinearQNet, test loading:
python -c "
from rl_snake.environment.utils import ModelLoader
loader = ModelLoader('PPO_snake', game_size=16)
print('✓ Model loaded successfully')
"
```

## Troubleshooting

### Issue: "Missing key(s) in state_dict"

**Cause:** Architecture mismatch between saved weights and current code

**Solution:** Create `.dill` files for legacy models (see "Handling Legacy Models" above)

### Issue: "Model file does not exist"

**Cause:** Incorrect model name or path

**Solution:**
```bash
# Check available models
ls models/

# Use correct name (without .zip extension)
rl-snake-play -m PPO_snake
```

### Issue: "Cannot unpickle LinearQNet"

**Cause:** Module import path changed

**Solution:** Ensure `LinearQNet` is importable:
```python
from src.rl_snake.agents.feature_extractor import LinearQNet  # Must work
```

### Issue: ".dill file exists but model won't load"

**Cause:** Corrupted `.dill` file

**Solution:** Recreate `.dill` file:
```python
import dill
from pathlib import Path
from src.rl_snake.agents.feature_extractor import LinearQNet

model_dir = Path("models/PPO_snake")
with open(model_dir / "feature_extractor.dill", 'wb') as f:
    dill.dump(LinearQNet, f)
```

## Technical Details

### Why `dill` Instead of `pickle`?

- **`pickle`**: Cannot serialize classes, only instances
- **`dill`**: Can serialize Python classes, functions, and more complex objects

```python
import pickle
import dill

# pickle fails on classes
pickle.dumps(LinearQNet)  # ❌ TypeError: cannot pickle 'type' object

# dill succeeds
dill.dumps(LinearQNet)    # ✓ Works!
```

### Security Considerations

**Warning:** `.dill` files execute arbitrary Python code when loaded. Only load `.dill` files from trusted sources.

```python
# Safe: Your own trained models
loader = ModelLoader("PPO_snake")

# Unsafe: Unknown .dill files from internet
# DO NOT load untrusted .dill files
```

### Performance Impact

**.dill file size:** ~5-20 KB (negligible)
**Loading overhead:** ~10-50 ms (negligible)
**Storage benefit:** Enables model portability across code versions

## See Also

- [Entry Points Documentation](ENTRY_POINTS.md): CLI commands for model usage
- [Configuration System](../config/README.md): Training configuration guide
- [Main README](../README.md): Project overview
