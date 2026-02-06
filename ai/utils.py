"""
Utility Functions
Helper functions for data management, visualization, and analysis
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def merge_replay_memories(memory_files, output_file, max_size=200000):
    """
    Merge multiple replay memory files into one
    
    Args:
        memory_files: List of memory file paths
        output_file: Output file path
        max_size: Maximum size of merged memory
    """
    print("\n🔄 Merging replay memories...")
    
    all_experiences = []
    
    for memory_file in memory_files:
        if os.path.exists(memory_file):
            print(f"   Loading: {memory_file}")
            with open(memory_file, 'rb') as f:
                experiences = pickle.load(f)
                all_experiences.extend(experiences)
                print(f"      Added {len(experiences)} experiences")
        else:
            print(f"   ⚠️  File not found: {memory_file}")
    
    print(f"\n   Total experiences: {len(all_experiences)}")
    
    # Sample if too large
    if len(all_experiences) > max_size:
        print(f"   Sampling down to {max_size} experiences...")
        indices = np.random.choice(len(all_experiences), max_size, replace=False)
        all_experiences = [all_experiences[i] for i in indices]
    
    # Save merged memory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_experiences, f)
    
    print(f"\n✅ Merged memory saved to: {output_file}")
    print(f"   Final size: {len(all_experiences)} experiences")


def analyze_training_logs(log_file):
    """
    Analyze training logs and generate statistics
    
    Args:
        log_file: Path to training_log.csv
        
    Returns:
        Dictionary with analysis results
    """
    import pandas as pd
    
    print(f"\n📊 Analyzing training logs: {log_file}")
    
    # Read CSV
    df = pd.read_csv(log_file)
    
    # Calculate statistics
    stats = {
        'total_episodes': len(df),
        'best_score': df['score'].max(),
        'best_episode': df.loc[df['score'].idxmax(), 'episode'],
        'avg_score': df['score'].mean(),
        'avg_lines': df['lines'].mean(),
        'max_lines': df['lines'].max(),
        'final_epsilon': df['epsilon'].iloc[-1],
        'avg_steps': df['steps'].mean(),
        'total_steps': df['steps'].sum()
    }
    
    # Performance improvement
    first_100_avg = df['score'][:100].mean()
    last_100_avg = df['score'][-100:].mean()
    improvement = ((last_100_avg - first_100_avg) / first_100_avg * 100) if first_100_avg > 0 else 0
    
    stats['first_100_avg_score'] = first_100_avg
    stats['last_100_avg_score'] = last_100_avg
    stats['improvement_percent'] = improvement
    
    # Print results
    print("\n📈 Training Statistics:")
    print(f"   Total episodes: {stats['total_episodes']}")
    print(f"   Best score: {stats['best_score']} (episode {stats['best_episode']})")
    print(f"   Average score: {stats['avg_score']:.1f}")
    print(f"   Average lines: {stats['avg_lines']:.1f}")
    print(f"   Max lines: {stats['max_lines']}")
    print(f"   Final epsilon: {stats['final_epsilon']:.4f}")
    print(f"   Total training steps: {stats['total_steps']}")
    print(f"\n   Performance improvement:")
    print(f"      First 100 episodes avg: {stats['first_100_avg_score']:.1f}")
    print(f"      Last 100 episodes avg: {stats['last_100_avg_score']:.1f}")
    print(f"      Improvement: {stats['improvement_percent']:.1f}%")
    
    return stats


def plot_training_progress(log_file, save_path=None):
    """
    Plot training progress from log file
    
    Args:
        log_file: Path to training_log.csv
        save_path: Path to save plot (optional)
    """
    import pandas as pd
    
    print(f"\n📈 Plotting training progress...")
    
    # Read CSV
    df = pd.read_csv(log_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Score over time
    ax1 = axes[0, 0]
    ax1.plot(df['episode'], df['score'], alpha=0.3, label='Score')
    # Moving average
    window = 100
    moving_avg = df['score'].rolling(window=window).mean()
    ax1.plot(df['episode'], moving_avg, linewidth=2, label=f'{window}-episode MA')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Lines cleared
    ax2 = axes[0, 1]
    ax2.plot(df['episode'], df['lines'], alpha=0.3, label='Lines')
    moving_avg_lines = df['lines'].rolling(window=window).mean()
    ax2.plot(df['episode'], moving_avg_lines, linewidth=2, label=f'{window}-episode MA')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Lines Cleared')
    ax2.set_title('Lines Cleared Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    ax3 = axes[1, 0]
    ax3.plot(df['episode'], df['epsilon'], linewidth=2, color='green')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon) Decay')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss
    ax4 = axes[1, 1]
    ax4.plot(df['episode'], df['avg_loss'], alpha=0.5, label='Loss')
    moving_avg_loss = df['avg_loss'].rolling(window=window).mean()
    ax4.plot(df['episode'], moving_avg_loss, linewidth=2, label=f'{window}-episode MA')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Loss')
    ax4.set_title('Training Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_experiments(experiment_dirs, save_path=None):
    """
    Compare multiple training experiments
    
    Args:
        experiment_dirs: List of experiment log directories
        save_path: Path to save comparison plot
    """
    import pandas as pd
    
    print("\n🔍 Comparing experiments...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, exp_dir in enumerate(experiment_dirs):
        log_file = os.path.join(exp_dir, 'training_log.csv')
        
        if not os.path.exists(log_file):
            print(f"   ⚠️  Log file not found: {log_file}")
            continue
        
        df = pd.read_csv(log_file)
        exp_name = os.path.basename(exp_dir)
        color = colors[i % len(colors)]
        
        # Moving average
        window = 100
        moving_avg_score = df['score'].rolling(window=window).mean()
        moving_avg_lines = df['lines'].rolling(window=window).mean()
        
        # Plot scores
        axes[0].plot(df['episode'], moving_avg_score, 
                    linewidth=2, label=exp_name, color=color)
        
        # Plot lines
        axes[1].plot(df['episode'], moving_avg_lines,
                    linewidth=2, label=exp_name, color=color)
        
        print(f"   ✓ {exp_name}: {len(df)} episodes")
    
    # Configure plots
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score (100-episode MA)')
    axes[0].set_title('Score Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Lines (100-episode MA)')
    axes[1].set_title('Lines Cleared Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n   Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_experiment_report(experiment_dir):
    """
    Create a comprehensive report for an experiment
    
    Args:
        experiment_dir: Path to experiment directory
    """
    print(f"\n📝 Creating experiment report...")
    
    # Load configuration
    config_file = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Analyze logs
    log_file = os.path.join(experiment_dir.replace('models', 'logs'), 'training_log.csv')
    if os.path.exists(log_file):
        stats = analyze_training_logs(log_file)
    else:
        stats = {}
    
    # Create report
    report_file = os.path.join(experiment_dir, 'REPORT.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Experiment Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Configuration\n\n")
        if config:
            f.write(f"```json\n")
            f.write(json.dumps(config, indent=2))
            f.write(f"\n```\n\n")
        
        f.write(f"## Training Statistics\n\n")
        if stats:
            f.write(f"- **Total Episodes:** {stats['total_episodes']}\n")
            f.write(f"- **Best Score:** {stats['best_score']} (Episode {stats['best_episode']})\n")
            f.write(f"- **Average Score:** {stats['avg_score']:.1f}\n")
            f.write(f"- **Average Lines:** {stats['avg_lines']:.1f}\n")
            f.write(f"- **Max Lines:** {stats['max_lines']}\n")
            f.write(f"- **Performance Improvement:** {stats['improvement_percent']:.1f}%\n")
            f.write(f"  - First 100 episodes: {stats['first_100_avg_score']:.1f}\n")
            f.write(f"  - Last 100 episodes: {stats['last_100_avg_score']:.1f}\n\n")
        
        f.write(f"## Files\n\n")
        f.write(f"- Configuration: `config.json`\n")
        f.write(f"- Best Model: `best_model_*.h5`\n")
        f.write(f"- Checkpoints: `checkpoints/`\n")
        f.write(f"- Training Logs: `../logs/`\n")
    
    print(f"   Report saved to: {report_file}")


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ai/utils.py merge memory1.pkl memory2.pkl output.pkl")
        print("  python ai/utils.py analyze logs/experiment/training_log.csv")
        print("  python ai/utils.py plot logs/experiment/training_log.csv")
        print("  python ai/utils.py report models/experiment")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'merge':
        memory_files = sys.argv[2:-1]
        output_file = sys.argv[-1]
        merge_replay_memories(memory_files, output_file)
    
    elif command == 'analyze':
        log_file = sys.argv[2]
        analyze_training_logs(log_file)
    
    elif command == 'plot':
        log_file = sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        plot_training_progress(log_file, save_path)
    
    elif command == 'report':
        experiment_dir = sys.argv[2]
        create_experiment_report(experiment_dir)
    
    else:
        print(f"Unknown command: {command}")
