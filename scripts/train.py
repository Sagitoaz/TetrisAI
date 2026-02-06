"""
Training Script for Tetris AI
Main entry point for training the DQN agent
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent
from ai.environment import TetrisEnvironment
from ai.trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Tetris AI using DQN')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes (default: 10000)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    
    # Agent parameters
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                       help='Minimum exploration rate (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Exploration decay rate (default: 0.995)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--memory', type=int, default=100000,
                       help='Replay memory capacity (default: 100000)')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint-freq', type=int, default=100,
                       help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--eval-freq', type=int, default=50,
                       help='Evaluate every N episodes (default: 50)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    # Config presets
    parser.add_argument('--config', type=str, choices=['exploration', 'exploitation', 'balanced'],
                       help='Use predefined configuration preset')
    
    return parser.parse_args()


def get_config_preset(preset_name):
    """
    Get predefined configuration presets
    
    Args:
        preset_name: Name of preset ('exploration', 'exploitation', 'balanced')
        
    Returns:
        Configuration dictionary
    """
    presets = {
        'exploration': {
            'epsilon': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.9995,  # Slow decay - more exploration
            'learning_rate': 0.001,
            'gamma': 0.95,
            'batch_size': 64,
            'description': 'High exploration for diverse experience collection'
        },
        'exploitation': {
            'epsilon': 0.5,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,   # Fast decay - quick exploitation
            'learning_rate': 0.0005,
            'gamma': 0.98,
            'batch_size': 128,
            'description': 'Low exploration for refined learning'
        },
        'balanced': {
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.997,   # Balanced decay
            'learning_rate': 0.001,
            'gamma': 0.95,
            'batch_size': 64,
            'description': 'Balanced exploration and exploitation'
        }
    }
    
    return presets.get(preset_name, {})


def main():
    """Main training function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🎮 TETRIS AI - DEEP Q-NETWORK TRAINING")
    print("="*70)
    
    # Create environment
    print("\n📦 Creating environment...")
    env = TetrisEnvironment(render=False)
    print(f"   State size: {env.state_size}")
    print(f"   Action size: {env.action_size}")
    
    # Configure agent
    agent_config = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_min': args.epsilon_min,
        'epsilon_decay': args.epsilon_decay,
        'memory_capacity': args.memory,
        'batch_size': args.batch_size,
        'target_update_freq': 10
    }
    
    # Apply preset if specified
    if args.config:
        print(f"\n⚙️  Using '{args.config}' configuration preset")
        preset = get_config_preset(args.config)
        print(f"   {preset.get('description', '')}")
        
        # Update agent config with preset values
        agent_config.update({
            'learning_rate': preset.get('learning_rate', args.lr),
            'gamma': preset.get('gamma', args.gamma),
            'epsilon': preset.get('epsilon', args.epsilon),
            'epsilon_min': preset.get('epsilon_min', args.epsilon_min),
            'epsilon_decay': preset.get('epsilon_decay', args.epsilon_decay),
            'batch_size': preset.get('batch_size', args.batch_size)
        })
    
    # Create agent
    print("\n🤖 Creating DQN agent...")
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=agent_config
    )
    
    print(f"   Learning rate: {agent_config['learning_rate']}")
    print(f"   Gamma: {agent_config['gamma']}")
    print(f"   Epsilon: {agent_config['epsilon']} → {agent_config['epsilon_min']}")
    print(f"   Epsilon decay: {agent_config['epsilon_decay']}")
    print(f"   Batch size: {agent_config['batch_size']}")
    print(f"   Memory capacity: {agent_config['memory_capacity']}")
    
    # Configure trainer
    trainer_config = {
        'num_episodes': args.episodes,
        'checkpoint_freq': args.checkpoint_freq,
        'eval_freq': args.eval_freq,
        'experiment_name': args.name or f"{args.config or 'default'}_{args.episodes}ep"
    }
    
    print("\n🎯 Creating trainer...")
    trainer = Trainer(agent, env, trainer_config)
    print(f"   Episodes: {trainer_config['num_episodes']}")
    print(f"   Checkpoint frequency: {trainer_config['checkpoint_freq']}")
    print(f"   Evaluation frequency: {trainer_config['eval_freq']}")
    print(f"   Experiment name: {trainer_config['experiment_name']}")
    
    # Print model summary
    print("\n📊 Model Architecture:")
    agent.model.summary()
    
    # Start training
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print("   Press Ctrl+C to stop training and save progress")
    print("="*70 + "\n")
    
    trainer.train(resume_from=args.resume)
    
    # Cleanup
    env.close()
    
    print("\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
