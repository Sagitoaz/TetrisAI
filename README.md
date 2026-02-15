# Tetris AI - Deep Q-Learning

Tetris AI using Deep Q-Network (DQN) with value-based learning approach.

## 🎮 Features

- **Value-Based DQN**: Agent evaluates state values instead of action Q-values
- **Smart State Space**: Agent considers all possible placements for each piece
- **Experience Replay**: Learns from past experiences
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation

## 📁 Project Structure

```
TetrisAI/
├── src/                    # Pygame game engine (core logic)
│   ├── game.py            # Main game logic
│   ├── tetromino.py       # Tetris pieces
│   ├── config.py          # Game configuration
│   └── shapes.py          # Piece shapes
├── ai/                     # AI training code
│   ├── environment.py     # Tetris environment wrapper
│   └── agent.py           # DQN agent
├── scripts/                # Training and play scripts
│   ├── train.py           # Training script
│   └── play.py            # Play with trained model
└── models/                 # Saved models (created during training)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the AI

```bash
python scripts/train.py
```

Training configuration (edit `scripts/train.py`):
- Episodes: 2000
- Memory size: 20000
- Batch size: 512
- Network: [32, 32]

### 3. Watch the AI Play

```bash
python scripts/play.py
```

Optional arguments:
```bash
python scripts/play.py --model models/best_lines.keras --episodes 5 --delay 0.1
```

## 🧠 How It Works

### Value-Based Approach

Instead of learning Q(state, action), the agent learns V(state) directly:

1. **For each piece**: Generate all possible final states (all rotations × all columns)
2. **Agent evaluates**: Predict value of each state using neural network
3. **Select best**: Choose state with highest predicted value
4. **Execute**: Place piece at that position
5. **Learn**: Update network based on actual reward received

### State Representation

Each state is represented by 4 features:
- **Lines cleared**: Number of lines cleared
- **Holes**: Empty cells with blocks above them
- **Bumpiness**: Height variation between adjacent columns
- **Aggregate height**: Sum of all column heights

### Reward Function

- **Place piece**: +1
- **Clear lines**: +(lines² × 10)
  - 1 line: +10
  - 2 lines: +40
  - 3 lines: +90
  - 4 lines: +160
- **Game over**: -2

## 📊 Training Progress

Models are saved in `models/`:
- `best_score.keras`: Model with highest score
- `best_lines.keras`: Model that cleared most lines
- `final.keras`: Final model after all training

## 🎯 Expected Results

After 2000 episodes:
- Average lines cleared: 50-100+ per game
- Best lines cleared: 200+
- Consistent gameplay without game overs

## 🔧 Hyperparameter Tuning

Key parameters in `scripts/train.py`:

```python
episodes = 2000              # More episodes = better learning
mem_size = 20000             # Larger = more diverse training
batch_size = 512             # Balance between speed and stability
epsilon_stop_episode = 1500  # When to stop exploring
discount = 0.95              # How much to value future rewards
n_neurons = [32, 32]         # Network size
```

## 📝 Notes

- Training takes ~30-60 minutes for 2000 episodes (on average CPU)
- GPU acceleration supported via TensorFlow
- Model saves automatically when new best is achieved
- Progress displayed every 50 episodes

## 🎓 Algorithm

DQN with:
- Experience replay buffer
- Epsilon-greedy exploration
- Bellman equation: Q = reward + γ × V(next_state)
- Adam optimizer with MSE loss

## 🙏 Credits

Inspired by classical Tetris AI techniques and modern deep RL approaches.
