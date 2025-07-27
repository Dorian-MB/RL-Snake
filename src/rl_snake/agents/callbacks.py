"""Custom callbacks for Snake RL training."""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm.auto import tqdm
from collections import deque
import time
import random

class SnakeProgressCallback(BaseCallback):
    """Enhanced progress bar with Snake-specific metrics and SB3 style."""
    
    def __init__(self, verbose=0, update_freq=100):
        super().__init__(verbose)
        self.pbar = None
        self.update_freq = update_freq
        self.start_time = None
        self.last_update = 0
        
    def _on_training_start(self) -> None:
        """Initialize progress bar with SB3-style formatting."""
        import time
        self.start_time = time.perf_counter()
        self.total_timesteps = self.model._total_timesteps - self.model.num_timesteps
        
        # Create progress bar with SB3-style formatting and colors
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="🐍 Snake RL",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
            ascii=False,  # Enable Unicode characters for smooth bars
            colour='#00ff41',  # Matrix green color
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            ncols=120,  # Width for better display
        )
        
        # Set custom bar style for smoother appearance
        if hasattr(self.pbar, 'bar_style'):
            self.pbar.bar_style = '█▉▊▋▌▍▎▏ '
    
    def _get_progress_color(self, avg_score):
        """Get color based on performance."""
        if avg_score < 5:
            return '#ff4444'  # Red for poor performance
        elif avg_score < 20:
            return '#ffaa00'  # Orange for medium performance  
        elif avg_score < 50:
            return '#00ff41'  # Green for good performance
        else:
            return '#00aaff'  # Blue for excellent performance
        
    def _get_postfix_dict(self, ep_info_buffer=None):
        """Generate postfix dictionary for progress bar."""
        postfix_dict = {}
        # Calculate FPS and time metrics
        elapsed_time = time.perf_counter() - self.start_time
        fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        if ep_info_buffer and len(ep_info_buffer) > 0:
            # Use last 10-50 episodes for stable metrics
            recent_episodes = list(deque(ep_info_buffer, maxlen=min(20, len(ep_info_buffer))))
            
            if recent_episodes:
                scores = [ep['r'] for ep in recent_episodes]
                lengths = [ep.get('l', 0) for ep in recent_episodes]
                ep_times = [ep.get('t', 0) for ep in recent_episodes]
                
                # Calculate statistics
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                avg_length = np.mean(lengths)
                avg_ep_time = np.mean(ep_times)
                
                # Create postfix
                postfix_dict.update({
                    'score': f'{avg_score:.1f}',
                    'max': f'{max_score:.0f}',
                    'steps': f'{avg_length:.0f}',
                    'ep_time': f'{avg_ep_time:.1f}s',
                    'fps': f'{fps:.0f}'
                })
            
            if not postfix_dict:
                postfix_dict['fps'] = f'{fps:.0f}'
                
        return postfix_dict
        
    def _on_step(self) -> bool:
        """Update progress bar with enhanced metrics."""
        if self.pbar and (self.num_timesteps - self.last_update >= self.update_freq or 
                         self.num_timesteps == self.model._total_timesteps):
            
            # Update progress
            steps_since_last = self.num_timesteps - self.last_update
            self.pbar.update(steps_since_last)
            self.last_update = self.num_timesteps
            
            # Calculate FPS and time metrics
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
            
            # Get episode statistics
            ep_info_buffer = getattr(self.model, 'ep_info_buffer', None)
            postfix_dict = self._get_postfix_dict(ep_info_buffer)
            
            # Update color based on performance
            if ep_info_buffer and len(ep_info_buffer) > 0:
                recent_episodes = list(deque(ep_info_buffer, maxlen=20))
                if recent_episodes:
                    avg_score = np.mean([ep['r'] for ep in recent_episodes])
                    new_color = self._get_progress_color(avg_score)
                    if hasattr(self.pbar, 'colour'):
                        self.pbar.colour = new_color
            
            self.pbar.set_postfix(postfix_dict)
            
        return True
        
    def _on_training_end(self) -> None:
        """Close progress bar and show final summary."""
        if self.pbar:
            # Final update to 100%
            remaining_steps = self.total_timesteps - self.pbar.n
            if remaining_steps > 0:
                self.pbar.update(remaining_steps)
                
            ep_info_buffer = getattr(self.model, 'ep_info_buffer', None)
            postfix_dict = self._get_postfix_dict(ep_info_buffer)
            self.pbar.set_postfix(postfix_dict)
            self.pbar.close()


class SnakeCurriculumCallback(BaseCallback):
    """Random curriculum learning for Snake game difficulty."""
    
    def __init__(self, min_size=10, max_size=50, update_freq=100, verbose=0):
        super().__init__(verbose)
        self.min_size = min_size
        self.max_size = max_size
        self.update_freq = update_freq
        self.current_size = min_size
        self.last_update = 0
        
    def _on_training_start(self) -> None:
        """Set initial difficulty."""
        self.current_size = random.randint(self.min_size, self.max_size)
        self._update_game_size(self.current_size)
        
    def _on_step(self) -> bool:
        """Randomly change grid size every update_freq steps."""
        if self.num_timesteps - self.last_update >= self.update_freq:
            # Generate new random size
            new_size = random.randint(self.min_size, self.max_size)
            
            if new_size != self.current_size:
                self._update_game_size(new_size)
                self.current_size = new_size
            
            self.last_update = self.num_timesteps
                
        return True
    
    def _update_game_size(self, new_size):
        """Update game size in all environments."""
        try:
            # For vectorized environments
            if hasattr(self.training_env, 'env_method'):
                self.training_env.env_method('set_game_size', new_size)
        except Exception as e:
            raise e


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
    callbacks:list=[],
    use_progress:bool=False,
    use_curriculum:bool=False,
    use_metrics:bool=False,
    use_save:bool=False,
    curriculum_start:int=10,
    curriculum_end:int=20,
    save_freq:int=50_000
):
    """
    Create a standard set of callbacks for Snake training.
    
    Args:
        callbacks: List of additional custom callbacks to include
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
    
    if use_progress:
        callbacks.append(SnakeProgressCallback())
        
    if use_curriculum:
        callbacks.append(SnakeCurriculumCallback())

    # if use_metrics:
    #     callbacks.append(SnakeMetricsCallback())
        
    # if use_save:
    #     callbacks.append(SnakeSaveCallback(save_freq=save_freq))

        
    callbacks = CallbackList(callbacks)  # Use CallbackList for better management
    return callbacks
