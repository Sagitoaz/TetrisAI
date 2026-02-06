"""
Training Infrastructure
Handles training loop, checkpointing, logging, and monitoring
"""

import os
import time
import csv
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm


class Trainer:
    """
    Trainer class for DQN Agent
    
    Manages:
    - Training loop
    - Checkpoint saving/loading
    - Logging (CSV, JSON)
    - Progress monitoring
    """
    
    def __init__(self, agent, environment, config=None):
        """
        Initialize Trainer
        
        Args:
            agent: DQN Agent instance
            environment: TetrisEnvironment instance
            config: Training configuration dictionary
        """
        self.agent = agent
        self.env = environment
        
        # Default configuration
        default_config = {
            'num_episodes': 10000,
            'max_steps_per_episode': 5000,
            'checkpoint_freq': 100,        # Save every N episodes
            'log_freq': 10,                # Log every N episodes
            'eval_freq': 50,               # Evaluate every N episodes
            'eval_episodes': 5,            # Number of episodes for evaluation
            'save_dir': 'models',
            'log_dir': 'logs',
            'experiment_name': f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Create directories
        self.experiment_dir = os.path.join(self.config['save_dir'], self.config['experiment_name'])
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.config['log_dir'], self.config['experiment_name'])
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(self.log_dir, 'training_log.csv')
        self.stats_file = os.path.join(self.log_dir, 'training_stats.json')
        self._init_log_file()
        
        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_scores': [],
            'episode_lines': [],
            'episode_steps': [],
            'losses': [],
            'eval_scores': [],
            'best_score': 0,
            'best_episode': 0
        }
        
        # Save configuration
        self._save_config()
    
    def _init_log_file(self):
        """Initialize CSV log file with headers"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'score', 'lines', 'level', 'steps',
                'total_reward', 'avg_loss', 'epsilon', 'memory_size',
                'duration', 'timestamp'
            ])
    
    def _save_config(self):
        """Save training configuration"""
        config_file = os.path.join(self.experiment_dir, 'config.json')
        
        config_data = {
            'trainer_config': self.config,
            'agent_config': self.agent.config,
            'environment': {
                'state_size': self.env.state_size,
                'action_size': self.env.action_size
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
    
    def train(self, resume_from=None):
        """
        Main training loop
        
        Args:
            resume_from: Path to checkpoint to resume from (optional)
        """
        start_episode = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            self.agent.load(resume_from)
            start_episode = self.agent.episode_count
            print(f"Resumed training from episode {start_episode}")
        
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.config['experiment_name']}")
        print(f"{'='*60}")
        print(f"Episodes: {self.config['num_episodes']}")
        print(f"Checkpoint frequency: {self.config['checkpoint_freq']}")
        print(f"Saving to: {self.experiment_dir}")
        print(f"{'='*60}\n")
        
        try:
            for episode in range(start_episode, self.config['num_episodes']):
                episode_start_time = time.time()
                
                # Run one episode
                stats = self._run_episode(episode)
                
                # Log progress
                if episode % self.config['log_freq'] == 0:
                    self._log_episode(episode, stats, time.time() - episode_start_time)
                
                # Save checkpoint
                if episode % self.config['checkpoint_freq'] == 0 and episode > 0:
                    self._save_checkpoint(episode)
                
                # Evaluate
                if episode % self.config['eval_freq'] == 0 and episode > 0:
                    self._evaluate(episode)
                
                # Update training statistics
                self._update_stats(stats)
                
                # Check for best model
                if stats['score'] > self.training_stats['best_score']:
                    self.training_stats['best_score'] = stats['score']
                    self.training_stats['best_episode'] = episode
                    self._save_best_model()
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
            print("Saving current state...")
            self._save_checkpoint(episode, prefix='interrupted')
        
        finally:
            # Save final statistics
            self._save_final_stats()
            print("\n\nTraining completed!")
            print(f"Best score: {self.training_stats['best_score']} (Episode {self.training_stats['best_episode']})")
            print(f"Results saved to: {self.experiment_dir}")
    
    def _run_episode(self, episode_num):
        """
        Run a single training episode
        
        Args:
            episode_num: Current episode number
            
        Returns:
            Dictionary with episode statistics
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0
        losses = []
        
        done = False
        while not done:
            # Agent chooses action
            action = self.agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.replay()
            if loss > 0:
                losses.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Break if max steps reached
            if steps >= self.config['max_steps_per_episode']:
                break
        
        # Episode end
        self.agent.on_episode_end()
        
        # Compile stats
        stats = {
            'episode': episode_num,
            'score': info['score'],
            'lines': info['lines'],
            'level': info['level'],
            'steps': steps,
            'total_reward': total_reward,
            'avg_loss': np.mean(losses) if losses else 0,
            'epsilon': self.agent.epsilon,
            'memory_size': len(self.agent.memory)
        }
        
        return stats
    
    def _log_episode(self, episode, stats, duration):
        """
        Log episode statistics
        
        Args:
            episode: Episode number
            stats: Episode statistics
            duration: Episode duration in seconds
        """
        # Print to console
        print(f"Episode {episode:5d} | "
              f"Score: {stats['score']:6d} | "
              f"Lines: {stats['lines']:3d} | "
              f"Steps: {stats['steps']:4d} | "
              f"Reward: {stats['total_reward']:7.1f} | "
              f"Loss: {stats['avg_loss']:6.4f} | "
              f"ε: {stats['epsilon']:.3f} | "
              f"Mem: {stats['memory_size']:6d} | "
              f"Time: {duration:.1f}s")
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                stats['score'],
                stats['lines'],
                stats['level'],
                stats['steps'],
                stats['total_reward'],
                stats['avg_loss'],
                stats['epsilon'],
                stats['memory_size'],
                duration,
                datetime.now().isoformat()
            ])
    
    def _save_checkpoint(self, episode, prefix='checkpoint'):
        """
        Save training checkpoint
        
        Args:
            episode: Current episode number
            prefix: Checkpoint filename prefix
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{prefix}_{episode:05d}')
        self.agent.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self):
        """Save the best performing model"""
        best_path = os.path.join(self.experiment_dir, 'best_model')
        self.agent.save(best_path)
        print(f"New best model saved! Score: {self.training_stats['best_score']}")
    
    def _evaluate(self, episode):
        """
        Evaluate current model performance
        
        Args:
            episode: Current episode number
        """
        print(f"\n--- Evaluation at Episode {episode} ---")
        
        eval_scores = []
        eval_lines = []
        
        for i in range(self.config['eval_episodes']):
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < self.config['max_steps_per_episode']:
                action = self.agent.act(state, training=False)  # No exploration
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                steps += 1
            
            eval_scores.append(info['score'])
            eval_lines.append(info['lines'])
        
        avg_score = np.mean(eval_scores)
        avg_lines = np.mean(eval_lines)
        
        print(f"Eval Results: Avg Score = {avg_score:.1f}, Avg Lines = {avg_lines:.1f}")
        print(f"-----------------------------------\n")
        
        self.training_stats['eval_scores'].append({
            'episode': episode,
            'avg_score': avg_score,
            'avg_lines': avg_lines,
            'scores': eval_scores
        })
    
    def _update_stats(self, stats):
        """
        Update training statistics
        
        Args:
            stats: Episode statistics
        """
        self.training_stats['episode_rewards'].append(stats['total_reward'])
        self.training_stats['episode_scores'].append(stats['score'])
        self.training_stats['episode_lines'].append(stats['lines'])
        self.training_stats['episode_steps'].append(stats['steps'])
        self.training_stats['losses'].append(stats['avg_loss'])
    
    def _save_final_stats(self):
        """Save final training statistics to JSON"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=4)
        
        # Calculate and save summary statistics
        summary = {
            'total_episodes': len(self.training_stats['episode_scores']),
            'best_score': self.training_stats['best_score'],
            'best_episode': self.training_stats['best_episode'],
            'avg_score': np.mean(self.training_stats['episode_scores'][-100:]),  # Last 100 episodes
            'avg_lines': np.mean(self.training_stats['episode_lines'][-100:]),
            'final_epsilon': self.agent.epsilon,
            'total_experiences': len(self.agent.memory)
        }
        
        summary_file = os.path.join(self.log_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("\nTraining Summary:")
        print(f"  Total Episodes: {summary['total_episodes']}")
        print(f"  Best Score: {summary['best_score']} (Episode {summary['best_episode']})")
        print(f"  Avg Score (last 100): {summary['avg_score']:.1f}")
        print(f"  Avg Lines (last 100): {summary['avg_lines']:.1f}")
        print(f"  Total Experiences: {summary['total_experiences']}")
