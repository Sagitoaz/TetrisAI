"""
Play Tetris with Trained AI Model
Watch the AI play!
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent
from ai.environment import Tetris
import numpy as np
from time import sleep
from PIL import Image


def play_with_ai(model_path='models/best_lines.keras', episodes=5, render_delay=0.5):
    """
    Play Tetris with trained AI
    
    Args:
        model_path: Path to trained model
        episodes: Number of episodes to play
        render_delay: Delay between moves (seconds)
    """
    print("="*60)
    print("TETRIS AI - GAMEPLAY")
    print("="*60)
    print(f"Loading model: {model_path}")
    
    # Create environment and agent
    env = Tetris()
    agent = DQNAgent(
        state_size=env.get_state_size(),
        n_neurons=[32, 32],
        activations=['relu', 'relu', 'linear']
    )
    
    # Load trained model
    agent.load_model(model_path)
    agent.epsilon = 0  # No exploration during play
    
    print(f"✓ Model loaded successfully!")
    print(f"Playing {episodes} episodes...")
    print("="*60)
    
    # Play episodes
    for episode in range(episodes):
        current_state = env.reset()
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while not done:
            # Get possible states
            next_states = env.get_next_states()
            state_dict = {tuple(v): k for k, v in next_states.items()}
            
            # Agent selects best state
            best_state = agent.best_state(state_dict.keys())
            best_action = state_dict[best_state]
            
            # Render (optional - using simple console display)
            if steps % 10 == 0:
                print(f"  Step {steps}: Score={env.get_game_score()}, Lines={env.lines_cleared}")
            
            # Execute action
            reward, done = env.play(best_action[0], best_action[1])
            steps += 1
            
            # Delay for visualization
            sleep(render_delay)
        
        # Episode complete
        print(f"\n  GAME OVER!")
        print(f"  Final Score: {env.get_game_score()}")
        print(f"  Lines Cleared: {env.lines_cleared}")
        print(f"  Pieces Placed: {env.pieces_placed}")
        print("-"*60)
    
    print("\n" + "="*60)
    print("GAMEPLAY COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Tetris with trained AI')
    parser.add_argument('--model', type=str, default='models/best_lines.keras',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to play')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between moves (seconds)')
    
    args = parser.parse_args()
    
    play_with_ai(args.model, args.episodes, args.delay)
