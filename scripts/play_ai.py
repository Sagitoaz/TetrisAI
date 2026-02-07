"""
Play Script - Watch Trained AI Play Tetris
Load a trained model and watch it play
"""

import os
import sys
import argparse
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent
from ai.environment import TetrisEnvironment


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Watch trained Tetris AI play')
    
    parser.add_argument('model_path', type=str,
                       help='Path to trained model (prefix without extensions)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to play (default: 5)')
    parser.add_argument('--render', action='store_true',
                       help='Render the game visually')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between steps in seconds (default: 0.1)')
    parser.add_argument('--max-steps', type=int, default=5000,
                       help='Maximum steps per episode (default: 5000)')
    
    return parser.parse_args()


def play_episode(agent, env, render=False, delay=0.1, max_steps=5000):
    """
    Play one episode with the trained agent
    
    Args:
        agent: Trained DQN agent
        env: Tetris environment
        render: Whether to render visually
        delay: Delay between steps
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with episode statistics
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    print("\n▶️  Starting new episode...")
    
    while not done and steps < max_steps:
        # Agent chooses action (no exploration)
        action = agent.act(state, training=False)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Render if enabled
        if render:
            env.render()
            time.sleep(delay)
        
        # Update
        state = next_state
        total_reward += reward
        steps += 1
        
        # Print progress every 100 steps
        if steps % 100 == 0:
            print(f"   Steps: {steps}, Score: {info['score']}, Lines: {info['lines']}")
    
    # Episode statistics
    stats = {
        'score': info['score'],
        'lines': info['lines'],
        'level': info['level'],
        'steps': steps,
        'total_reward': total_reward
    }
    
    return stats


def main():
    """Main function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🎮 TETRIS AI - PLAY MODE")
    print("="*70)
    
    # Check if model exists
    model_file = f"{args.model_path}_model.weights.h5"
    if not os.path.exists(model_file):
        print(f"\n❌ Error: Model file not found: {model_file}")
        print("   Please provide the correct model path (prefix without extensions)")
        print("   Example: models/experiment_name/best_model")
        sys.exit(1)
    
    print(f"\n📦 Loading model from: {args.model_path}")
    
    # Create environment
    env = TetrisEnvironment(render=args.render)
    
    # Create agent
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size
    )
    
    # Load trained model
    agent.load(args.model_path)
    
    # Get agent stats
    stats = agent.get_stats()
    print(f"\n📊 Agent Statistics:")
    print(f"   Episodes trained: {stats['episode']}")
    print(f"   Training steps: {stats['training_step']}")
    print(f"   Epsilon: {stats['epsilon']:.4f}")
    print(f"   Memory size: {stats['memory_size']}")
    
    # Play episodes
    print(f"\n{'='*70}")
    print(f"🎯 Playing {args.episodes} episode(s)")
    print(f"{'='*70}")
    
    episode_stats = []
    
    for episode in range(args.episodes):
        print(f"\n📺 Episode {episode + 1}/{args.episodes}")
        
        stats = play_episode(
            agent=agent,
            env=env,
            render=args.render,
            delay=args.delay,
            max_steps=args.max_steps
        )
        
        episode_stats.append(stats)
        
        # Print episode results
        print(f"\n✅ Episode {episode + 1} Results:")
        print(f"   Score: {stats['score']}")
        print(f"   Lines cleared: {stats['lines']}")
        print(f"   Level reached: {stats['level']}")
        print(f"   Steps taken: {stats['steps']}")
        print(f"   Total reward: {stats['total_reward']:.1f}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("📈 SUMMARY")
    print(f"{'='*70}")
    
    avg_score = sum(s['score'] for s in episode_stats) / len(episode_stats)
    avg_lines = sum(s['lines'] for s in episode_stats) / len(episode_stats)
    max_score = max(s['score'] for s in episode_stats)
    max_lines = max(s['lines'] for s in episode_stats)
    
    print(f"   Episodes played: {len(episode_stats)}")
    print(f"   Average score: {avg_score:.1f}")
    print(f"   Average lines: {avg_lines:.1f}")
    print(f"   Best score: {max_score}")
    print(f"   Best lines: {max_lines}")
    print(f"{'='*70}\n")
    
    # Cleanup
    env.close()
    
    print("✅ Finished!")


if __name__ == '__main__':
    main()
