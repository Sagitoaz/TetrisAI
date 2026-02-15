"""
Training Script for Tetris AI
Simple and effective training loop
"""
import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent
from ai.environment import Tetris
from datetime import datetime
from statistics import mean
from tqdm import tqdm


def train():
    """Train Tetris AI with DQN"""
    
    # ====================
    # TRAINING CONFIGURATION
    # ====================
    episodes = 10000                 # Total episodes to train
    max_steps = None                # Max steps per episode (None = until game over)
    epsilon_stop_episode = 2000     # Stop exploration decay at this episode
    mem_size = 200000                # Replay memory size
    discount = 0.95                 # Discount factor (gamma)
    batch_size = 512                # Training batch size
    epochs = 1                      # Epochs per training step
    train_every = 1                 # Train every N episodes
    log_every = 50                  # Log stats every N episodes
    save_best_model = True          # Save best model
    n_neurons = [32, 32]            # Network architecture
    activations = ['relu', 'relu', 'linear']  # Activations
    replay_start_size = 2000        # Min replay size before training
    
    # ====================
    # SETUP
    # ====================
    print("="*60)
    print("TETRIS AI - DQN TRAINING")
    print("="*60)
    print(f"Episodes: {episodes}")
    print(f"Memory size: {mem_size}")
    print(f"Batch size: {batch_size}")
    print(f"Network: {n_neurons}")
    print(f"Replay start: {replay_start_size}")
    print("="*60)
    
    # Create environment and agent
    env = Tetris()
    agent = DQNAgent(
        state_size=env.get_state_size(),
        n_neurons=n_neurons,
        activations=activations,
        epsilon_stop_episode=epsilon_stop_episode,
        mem_size=mem_size,
        discount=discount,
        replay_start_size=replay_start_size
    )
    
    # Tracking
    scores = []
    lines_cleared_list = []
    best_score = 0
    best_lines = 0
    
    # Training loop
    print("\nStarting training...\n")
    
    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0
        
        # Play episode
        while not done and (not max_steps or steps < max_steps):
            # Get all possible next states
            next_states = env.get_next_states()
            
            # Convert to state vectors
            state_dict = {tuple(v): k for k, v in next_states.items()}
            
            # Agent selects best state
            best_state = agent.best_state(state_dict.keys())
            best_action = state_dict[best_state]
            
            # Execute action
            reward, done = env.play(best_action[0], best_action[1])
            
            # Store experience
            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            steps += 1
        
        # Track stats
        scores.append(env.get_game_score())
        lines_cleared_list.append(env.lines_cleared)
        
        # Train agent
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)
        
        # Log progress
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            avg_lines = mean(lines_cleared_list[-log_every:])
            
            print(f"\nEpisode {episode}/{episodes}")
            print(f"  Avg Score: {avg_score:.1f} (min={min_score}, max={max_score})")
            print(f"  Avg Lines: {avg_lines:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Memory: {len(agent.memory)}/{agent.mem_size}")
        
        # Save best model
        if save_best_model:
            if env.get_game_score() > best_score:
                best_score = env.get_game_score()
                os.makedirs('models', exist_ok=True)
                agent.save_model('models/best_score.keras')
                print(f"\n✓ New best score: {best_score} (episode {episode})")
            
            if env.lines_cleared > best_lines:
                best_lines = env.lines_cleared
                os.makedirs('models', exist_ok=True)
                agent.save_model('models/best_lines.keras')
                print(f"\n✓ New best lines: {best_lines} (episode {episode})")
    
    # Training complete
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Score: {best_score}")
    print(f"Best Lines: {best_lines}")
    print(f"Final Epsilon: {agent.epsilon:.3f}")
    print("="*60)
    
    # Save final model
    os.makedirs('models', exist_ok=True)
    agent.save_model('models/final.keras')
    print("\n✓ Saved final model to models/final.keras")


if __name__ == "__main__":
    train()
