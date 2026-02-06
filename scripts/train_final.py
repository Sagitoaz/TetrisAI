"""
Final Training Script - Train with Merged Data
Use this after merging replay memories from Person A and B
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent, ReplayMemory
from ai.environment import TetrisEnvironment
from ai.trainer import Trainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train final model with merged data')
    
    parser.add_argument('--memory-file', type=str, required=True,
                       help='Path to merged memory file (e.g., data/merged_memory.pkl)')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes (default: 3000)')
    parser.add_argument('--name', type=str, default='final_model',
                       help='Experiment name (default: final_model)')
    parser.add_argument('--config', type=str, choices=['balanced', 'exploitation'],
                       default='balanced',
                       help='Training configuration preset')
    
    # Agent parameters
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.98,
                       help='Discount factor (default: 0.98)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🎮 TETRIS AI - FINAL MODEL TRAINING")
    print("="*70)
    
    # Check if memory file exists
    if not os.path.exists(args.memory_file):
        print(f"\n❌ Error: Memory file not found: {args.memory_file}")
        print("\nPlease merge memories first:")
        print("  python ai\\utils.py merge memory1.pkl memory2.pkl data\\merged_memory.pkl")
        sys.exit(1)
    
    print(f"\n📦 Loading merged memory from: {args.memory_file}")
    
    # Create environment
    print("\n🎮 Creating environment...")
    env = TetrisEnvironment(render=False)
    
    # Configure agent
    agent_config = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'epsilon': 0.3,           # Lower epsilon for final training
        'epsilon_min': 0.01,
        'epsilon_decay': 0.996,    # Moderate decay
        'memory_capacity': 200000,  # Large capacity for merged memory
        'batch_size': args.batch_size,
        'target_update_freq': 10
    }
    
    # Create agent
    print("\n🤖 Creating DQN agent...")
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=agent_config
    )
    
    # Load merged memory
    print("\n💾 Loading merged replay memory...")
    agent.memory.load(args.memory_file)
    print(f"   Loaded {len(agent.memory)} experiences")
    
    # Verify memory has enough samples
    if len(agent.memory) < agent.batch_size:
        print(f"\n❌ Error: Not enough experiences in memory!")
        print(f"   Memory size: {len(agent.memory)}")
        print(f"   Required: {agent.batch_size}")
        sys.exit(1)
    
    print(f"\n⚙️  Agent Configuration:")
    print(f"   Learning rate: {agent_config['learning_rate']}")
    print(f"   Gamma: {agent_config['gamma']}")
    print(f"   Epsilon: {agent_config['epsilon']} → {agent_config['epsilon_min']}")
    print(f"   Batch size: {agent_config['batch_size']}")
    print(f"   Memory size: {len(agent.memory)}")
    
    # Pre-train on existing experiences
    print("\n🔥 Pre-training on merged experiences...")
    print("   This may take a while...")
    
    pretrain_steps = min(1000, len(agent.memory) // agent.batch_size)
    total_loss = 0
    
    for step in range(pretrain_steps):
        loss = agent.replay()
        total_loss += loss
        
        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"   Step {step + 1}/{pretrain_steps} | Avg Loss: {avg_loss:.4f}")
    
    print(f"\n✅ Pre-training completed!")
    print(f"   Average loss: {total_loss / pretrain_steps:.4f}")
    
    # Configure trainer
    trainer_config = {
        'num_episodes': args.episodes,
        'checkpoint_freq': 100,
        'eval_freq': 50,
        'experiment_name': args.name
    }
    
    print("\n🎯 Creating trainer...")
    trainer = Trainer(agent, env, trainer_config)
    
    # Start training
    print("\n" + "="*70)
    print("🚀 STARTING FINAL TRAINING")
    print("="*70)
    print(f"   Episodes: {trainer_config['num_episodes']}")
    print(f"   Starting with {len(agent.memory)} pre-loaded experiences")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Cleanup
    env.close()
    
    print("\n✅ Final training completed successfully!")
    print(f"\n📊 Results saved to:")
    print(f"   Models: models\\{args.name}")
    print(f"   Logs: logs\\{args.name}")


if __name__ == '__main__':
    main()
