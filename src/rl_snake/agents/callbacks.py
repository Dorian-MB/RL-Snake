"""Custom callbacks for Snake RL training."""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm


class SnakeProgressCallback(BaseCallback):
    """Progress bar with Snake-specific metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        
    def _on_training_start(self) -> None:
        """Initialize progress bar."""
        total_timesteps = self.model._total_timesteps
        self.pbar = tqdm(total=total_timesteps, desc="🐍 Training Snake RL")
        
    def _on_step(self) -> bool:
        """Update progress bar."""
        if self.pbar:
            self.pbar.update(1)
            
            # Update every 1000 steps
            if self.num_timesteps % 1000 == 0:
                ep_info_buffer = getattr(self.model, 'ep_info_buffer', None)
                if ep_info_buffer and len(ep_info_buffer) > 0:
                    recent_rewards = [ep['r'] for ep in ep_info_buffer[-10:]]
                    avg_reward = np.mean(recent_rewards)
                    max_length = max([ep.get('l', 0) for ep in ep_info_buffer[-10:]], default=0)
                    self.pbar.set_description(
                        f"🐍 Snake RL - Score: {avg_reward:.1f} | Max Length: {max_length}"
                    )
        return True
        
    def _on_training_end(self) -> None:
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()


class SnakeCurriculumCallback(BaseCallback):
    """Curriculum learning for Snake game difficulty."""
    
    def __init__(self, start_size=10, end_size=20, verbose=1):
        super().__init__(verbose)
        self.start_size = start_size
        self.end_size = end_size
        self.current_size = start_size
        
        # Define curriculum steps
        self.curriculum_steps = [
            (10_000, 12),   # After 10k steps, increase to 12x12
            (30_000, 15),   # After 30k steps, increase to 15x15
            (60_000, 18),   # After 60k steps, increase to 18x18
            (90_000, 20),   # After 90k steps, increase to 20x20
        ]
        
    def _on_training_start(self) -> None:
        """Set initial difficulty."""
        if self.verbose:
            print(f"🎮 Starting curriculum: {self.start_size}x{self.start_size} grid")
        self._update_game_size(self.start_size)
        
    def _on_step(self) -> bool:
        """Check for curriculum progression."""
        for timestep, new_size in self.curriculum_steps:
            if (self.num_timesteps == timestep and 
                new_size > self.current_size and 
                new_size <= self.end_size):
                
                if self.verbose:
                    print(f"\n🚀 Curriculum step: Increasing to {new_size}x{new_size} grid")
                    print(f"   Timestep: {self.num_timesteps:,}")
                
                self._update_game_size(new_size)
                self.current_size = new_size
                break
                
        return True
    
    def _update_game_size(self, new_size):
        """Update game size in all environments."""
        try:
            # For vectorized environments
            if hasattr(self.training_env, 'env_method'):
                self.training_env.env_method('set_game_size', new_size)
            elif hasattr(self.training_env, 'set_game_size'):
                self.training_env.set_game_size(new_size)
            else:
                if self.verbose:
                    print("⚠️  Environment doesn't support dynamic game size")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Could not update game size: {e}")


class SnakeMetricsCallback(BaseCallback):
    """Log additional Snake-specific metrics."""
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_scores = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Collect and log metrics."""
        # Collect episode info
        if hasattr(self.locals, 'infos'):
            for info in self.locals['infos']:
                if 'episode' in info:
                    ep_info = info['episode']
                    self.episode_scores.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
        
        # Log metrics periodically
        if self.num_timesteps % self.log_freq == 0 and self.episode_scores:
            # Calculate statistics
            avg_score = np.mean(self.episode_scores[-100:])  # Last 100 episodes
            max_score = np.max(self.episode_scores[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            max_length = np.max(self.episode_lengths[-100:])
            
            # Log to TensorBoard if available
            if hasattr(self.model, 'logger') and self.model.logger:
                self.model.logger.record('snake/avg_score', avg_score)
                self.model.logger.record('snake/max_score', max_score)
                self.model.logger.record('snake/avg_length', avg_length)
                self.model.logger.record('snake/max_length', max_length)
                self.model.logger.dump(self.num_timesteps)
            
            if self.verbose:
                print(f"\n📊 Snake Metrics (last 100 episodes):")
                print(f"   Avg Score: {avg_score:.2f} | Max Score: {max_score:.0f}")
                print(f"   Avg Length: {avg_length:.1f} | Max Length: {max_length:.0f}")
        
        return True


class SnakeSaveCallback(BaseCallback):
    """Save best models based on performance."""
    
    def __init__(self, save_freq=50_000, save_path="models/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Save model if performance improved."""
        if self.num_timesteps % self.save_freq == 0:
            # Get recent performance
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                recent_rewards = [ep['r'] for ep in self.model.ep_info_buffer[-50:]]
                if recent_rewards:
                    mean_reward = np.mean(recent_rewards)
                    
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        save_path = f"{self.save_path}snake_best_{self.num_timesteps}_score_{mean_reward:.1f}"
                        self.model.save(save_path)
                        
                        if self.verbose:
                            print(f"\n💾 New best model saved! Score: {mean_reward:.2f}")
                            print(f"   Path: {save_path}.zip")
        
        return True


def create_snake_callbacks(
    use_progress=True,
    use_curriculum=True,
    use_metrics=True,
    use_save=True,
    curriculum_start=10,
    curriculum_end=20,
    save_freq=50_000
):
    """
    Create a standard set of callbacks for Snake training.
    
    Args:
        use_progress: Whether to include progress bar
        use_curriculum: Whether to include curriculum learning
        use_metrics: Whether to include metrics logging
        use_save: Whether to include model saving
        curriculum_start: Starting grid size
        curriculum_end: Ending grid size
        save_freq: How often to save models
        
    Returns:
        List of configured callbacks
    """
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = []
    
    if use_progress:
        callbacks.append(SnakeProgressCallback())
        
    if use_curriculum:
        callbacks.append(SnakeCurriculumCallback(
            start_size=curriculum_start,
            end_size=curriculum_end
        ))
        
    if use_metrics:
        callbacks.append(SnakeMetricsCallback())
        
    if use_save:
        callbacks.append(SnakeSaveCallback(save_freq=save_freq))

        
    callbacks = CallbackList(callbacks)  # Use CallbackList for better management
    return callbacks
