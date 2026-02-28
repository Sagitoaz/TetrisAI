"""
Training Script for Tetris AI
Simple and effective training loop with checkpoint / resume support.

Hỗ trợ nhiều người / nhiều máy train song song:
    # Máy A
    python scripts/train.py --name alice

    # Máy B
    python scripts/train.py --name bob

    # Sau khi copy models từ cả hai máy về một thư mục, gộp lại:
    python scripts/merge.py --models models/alice_best_score.keras models/bob_best_score.keras

Tên session mặc định là hostname của máy.
"""
import sys
import os
import signal
import json
import socket
import argparse
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent
from ai.environment import Tetris
from datetime import datetime
from statistics import mean
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Graceful interrupt handling
# ---------------------------------------------------------------------------
_interrupted = False

def _handle_interrupt(sig, frame):
    global _interrupted
    if not _interrupted:
        print("\n\n⚠  Training interrupted! Saving checkpoint before exit...")
        _interrupted = True
    # Second Ctrl+C → force quit immediately
    else:
        print("\nForce quitting.")
        sys.exit(1)

signal.signal(signal.SIGINT, _handle_interrupt)
signal.signal(signal.SIGTERM, _handle_interrupt)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_model(name): return f'models/{name}.keras'
def _ckpt_meta(name):  return f'models/{name}.json'
def _best_score_path(name): return f'models/{name}_best_score.keras'
def _best_lines_path(name): return f'models/{name}_best_lines.keras'


def save_checkpoint(agent, episode, best_score, best_lines, scores, lines_list, name='checkpoint'):
    """Persist model weights + training metadata to disk."""
    os.makedirs('models', exist_ok=True)
    agent.save_model(_ckpt_model(name))
    meta = {
        'episode':     episode,
        'epsilon':     float(agent.epsilon),
        'best_score':  best_score,
        'best_lines':  best_lines,
        'scores':      list(scores[-200:]),
        'lines_cleared': list(lines_list[-200:]),
        'session_name': name,
    }
    with open(_ckpt_meta(name), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Checkpoint saved at episode {episode}  (epsilon={agent.epsilon:.4f})"
          f"  →  models/{name}.keras + models/{name}.json")


def load_checkpoint(agent, name='checkpoint'):
    """
    Load checkpoint if available.
    Returns the saved metadata dict or None if no checkpoint exists.
    """
    if not (os.path.exists(_ckpt_meta(name)) and os.path.exists(_ckpt_model(name))):
        return None
    with open(_ckpt_meta(name)) as f:
        meta = json.load(f)
    agent.load_model(_ckpt_model(name))
    # Restore epsilon (clamp to agent's min)
    agent.epsilon = max(float(meta['epsilon']), agent.epsilon_min)
    return meta


def train(model_file=None, start_epsilon=None, name=None):
    """
    Train Tetris AI with DQN.

    Args:
        model_file:    Path to a .keras file to load weights from before training.
                       Overrides checkpoint resume.  e.g. 'models/best_score.keras'
        start_epsilon: Epsilon to use when loading model_file (default 0.0 –
                       the model is already trained, no need to explore much).
        name:          Session name used for all saved files.
                       Defaults to machine hostname.
                       Checkpoint  → models/{name}.keras + models/{name}.json
                       Best score  → models/{name}_best_score.keras
                       Best lines  → models/{name}_best_lines.keras
    """
    if name is None:
        name = socket.gethostname().split('.')[0]  # e.g. 'alice-pc'
    
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
    checkpoint_every = 500          # Save checkpoint every N episodes (for resume)
    n_neurons = [32, 32]            # Network architecture
    activations = ['relu', 'relu', 'linear']  # Activations
    replay_start_size = 2000        # Min replay size before training
    
    # ====================
    # SETUP
    # ====================
    print("="*60)
    print("TETRIS AI - DQN TRAINING")
    print("="*60)
    print(f"Session name : {name}")
    print(f"Checkpoint   : models/{name}.keras + models/{name}.json")
    print(f"Best score   : models/{name}_best_score.keras")
    print(f"Best lines   : models/{name}_best_lines.keras")
    print("-"*60)
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
    
    # ---------------------------------------------------------------------------
    # Resume from checkpoint (if available)
    # ---------------------------------------------------------------------------
    start_episode = 0
    scores = []
    lines_cleared_list = []
    best_score = 0
    best_lines = 0

    # ---------------------------------------------------------------------------
    # Resolve starting state: --model flag > checkpoint > fresh
    # ---------------------------------------------------------------------------
    if model_file:
        # Load weights from an arbitrary .keras file
        if not os.path.exists(model_file):
            print(f"ERROR: model file not found: {model_file}")
            sys.exit(1)
        agent.load_model(model_file)
        eps = start_epsilon if start_epsilon is not None else 0.0
        agent.epsilon = max(eps, agent.epsilon_min)
        print(f"✓ Loaded weights from '{model_file}'  (epsilon set to {agent.epsilon:.4f})")
        print("Starting fine-tune training from episode 0.")
    else:
        checkpoint = load_checkpoint(agent, name)
        if checkpoint:
            answer = input(
                f"\nCheckpoint found at episode {checkpoint['episode']}  "
                f"(epsilon={checkpoint['epsilon']:.4f}, "
                f"best_score={checkpoint['best_score']}, "
                f"best_lines={checkpoint['best_lines']}).\n"
                "Resume? [Y/n]: "
            ).strip().lower()
            if answer in ('', 'y', 'yes'):
                start_episode       = checkpoint['episode'] + 1
                best_score          = checkpoint['best_score']
                best_lines          = checkpoint['best_lines']
                scores              = checkpoint.get('scores', [])
                lines_cleared_list  = checkpoint.get('lines_cleared', [])
                print(f"✓ Resuming from episode {start_episode}  epsilon={agent.epsilon:.4f}")
            else:
                agent.epsilon = 1.0
                print("Starting fresh training.")
        else:
            print("No checkpoint found. Starting fresh training.")
    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    print("\nStarting training...\n")
    
    for episode in tqdm(range(start_episode, episodes)):
        
        if _interrupted:
            break

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
                agent.save_model(_best_score_path(name))
                print(f"\n✓ New best score: {best_score} (episode {episode})")
            
            if env.lines_cleared > best_lines:
                best_lines = env.lines_cleared
                os.makedirs('models', exist_ok=True)
                agent.save_model(_best_lines_path(name))
                print(f"\n✓ New best lines: {best_lines} (episode {episode})")
        
        # Periodic checkpoint
        if checkpoint_every and episode and episode % checkpoint_every == 0:
            save_checkpoint(agent, episode, best_score, best_lines,
                            scores, lines_cleared_list, name)
    
    # ---------------------------------------------------------------------------
    # After loop: save checkpoint regardless of how we exited
    # ---------------------------------------------------------------------------
    last_episode = episode  # last completed episode
    save_checkpoint(agent, last_episode, best_score, best_lines,
                    scores, lines_cleared_list, name)

    if _interrupted:
        print("\n" + "="*60)
        print("TRAINING PAUSED")
        print("="*60)
        print(f"Session name       : {name}")
        print(f"Stopped at episode : {last_episode}")
        print(f"Best Score so far  : {best_score}")
        print(f"Best Lines so far  : {best_lines}")
        print(f"Epsilon            : {agent.epsilon:.4f}")
        print(f"Checkpoint saved   : models/{name}.keras")
        print(f"Best score file    : models/{name}_best_score.keras")
        print()
        print(f"Tiếp tục:")
        print(f"  python scripts/train.py --name {name}")
        print()
        print("Chia sẻ để gộp (gửi cho người khác):")
        print(f"  models/{name}_best_score.keras")
        print(f"  models/{name}_best_lines.keras")
        print("="*60)
        return

    # Training complete
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Session name  : {name}")
    print(f"Best Score    : {best_score}")
    print(f"Best Lines    : {best_lines}")
    print(f"Final Epsilon : {agent.epsilon:.3f}")
    print()
    print("Files saved:")
    print(f"  models/{name}_best_score.keras")
    print(f"  models/{name}_best_lines.keras")
    print(f"  models/{name}.keras  (final checkpoint)")
    print()
    print("Để gộp với người khác:")
    print(f"  python scripts/merge.py --models models/{name}_best_score.keras models/OTHER_best_score.keras")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Tetris DQN agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Train thông thường (tên session = hostname máy)
  python scripts/train.py

  # Train với tên session tùy chọn (khuyến nghị khi nhiều người train)
  python scripts/train.py --name alice
  python scripts/train.py --name bob

  # Tiếp tục từ session đã lưu
  python scripts/train.py --name alice

  # Load weights từ file cụ thể rồi train tiếp
  python scripts/train.py --name alice --model models/merged.keras --epsilon 0.05

Sau khi train, gộp model từ nhiều người:
  python scripts/merge.py --models models/alice_best_score.keras models/bob_best_score.keras
"""
    )
    parser.add_argument(
        "--name", "-n", type=str, default=None,
        help="Tên session (mặc định: hostname máy). Dùng để phân biệt khi nhiều người train. "
             "VD: --name alice"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a .keras model file to load weights from and continue training. "
             "Example: --model models/best_score.keras"
    )
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="Override starting epsilon when loading a model file (default: 0.0). "
             "Range 0.0–1.0.  Example: --epsilon 0.1"
    )
    args = parser.parse_args()
    train(model_file=args.model, start_epsilon=args.epsilon, name=args.name)
