# Tetris AI - Deep Q-Learning

Tetris AI using Deep Q-Network (DQN) with value-based learning approach.

## 🎮 Features

- **Value-Based DQN**: Agent evaluates state values instead of action Q-values
- **Smart State Space**: Agent considers all possible placements for each piece
- **Experience Replay**: Learns from past experiences
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Checkpoint/Resume**: Training tự động lưu, có thể tiếp tục bất cứ lúc nào
- **Collaborative Training**: Nhiều người train song song, merge lại thành model mạnh hơn

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
│   ├── train.py           # Training script (với checkpoint & session name)
│   ├── merge.py           # Gộp model từ nhiều người/máy
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
# Train thông thường (tên session = hostname máy, tự resume nếu có checkpoint)
python scripts/train.py

# Chỉ định tên session (khuyến nghị)
python scripts/train.py --name alice
```

Các file được tạo ra trong `models/`:

- `alice.keras` + `alice.json` — checkpoint, dùng để resume
- `alice_best_score.keras` — model đạt điểm cao nhất
- `alice_best_lines.keras` — model xóa nhiều hàng nhất

### 3. Watch the AI Play

```bash
python scripts/play.py
python scripts/play.py --model models/alice_best_score.keras --episodes 5 --delay 0.1
```

---

## 🔄 Checkpoint & Resume

Training **tự động lưu checkpoint** mỗi 500 episodes và khi thoát (Ctrl+C).

Lần sau chạy lại, chương trình hỏi có muốn tiếp tục không:

```bash
python scripts/train.py --name alice
# → "Checkpoint found at episode 1500 ... Resume? [Y/n]:"
```

Nhấn Enter (hoặc Y) để tiếp tục từ chỗ dừng.

---

## 👥 Collaborative Training (Nhiều người cùng train)

Nhiều người/máy có thể train song song và gộp model lại để tạo model mạnh hơn.

### Bước 1 — Mỗi người train với tên riêng

```bash
# Máy A (người Alice)
python scripts/train.py --name alice

# Máy B (người Bob)
python scripts/train.py --name bob
```

### Bước 2 — Copy model về cùng một thư mục

Copy các file `*_best_score.keras` từ tất cả máy vào cùng thư mục `models/`:

```
models/
├── alice_best_score.keras   ← copy từ máy Alice
├── bob_best_score.keras     ← copy từ máy Bob
└── ...
```

### Bước 3 — Merge model

```bash
# Tự động merge tất cả *_best_score.keras trong models/
python scripts/merge.py

# Hoặc chỉ định file cụ thể
python scripts/merge.py --models models/alice_best_score.keras models/bob_best_score.keras

# Merge có trọng số (model điểm cao được ưu tiên hơn)
python scripts/merge.py --models models/alice_best_score.keras models/bob_best_score.keras --weighted

# Chỉ so sánh, không gộp
python scripts/merge.py --compare-only
```

Output mặc định: `models/merged.keras`

### Bước 4 — Tiếp tục train từ model đã gộp

```bash
# Mỗi người load model merged và train tiếp với tên session của mình
python scripts/train.py --name alice --model models/merged.keras --epsilon 0.05
python scripts/train.py --name bob   --model models/merged.keras --epsilon 0.05
```

Lặp lại các bước 1–4 để model ngày càng mạnh hơn.

---

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

---

## 📊 Training Progress

Models được lưu trong `models/` với tên theo session:

| File                      | Mô tả                                   |
| ------------------------- | --------------------------------------- |
| `{name}.keras`            | Checkpoint mới nhất (dùng để resume)    |
| `{name}.json`             | Metadata: episode, epsilon, best score… |
| `{name}_best_score.keras` | Model đạt điểm cao nhất                 |
| `{name}_best_lines.keras` | Model xóa nhiều hàng nhất               |
| `merged.keras`            | Model sau khi gộp từ nhiều người        |

## 🎯 Expected Results

After 2000 episodes:

- Average lines cleared: 50-100+ per game
- Best lines cleared: 200+
- Consistent gameplay without game overs

## 🔧 Hyperparameter Tuning

Key parameters in `scripts/train.py`:

```python
episodes = 10000             # More episodes = better learning
mem_size = 200000            # Larger = more diverse training
batch_size = 512             # Balance between speed and stability
epsilon_stop_episode = 2000  # When to stop exploring
discount = 0.95              # How much to value future rewards
n_neurons = [32, 32]         # Network size
checkpoint_every = 500       # Save checkpoint every N episodes
```

## 📝 Notes

- Training tự động resume khi chạy lại cùng `--name`
- Ctrl+C dừng an toàn và lưu checkpoint ngay lập tức
- GPU acceleration supported via TensorFlow
- Progress displayed every 50 episodes

## 🎓 Algorithm

DQN with:

- Experience replay buffer
- Epsilon-greedy exploration
- Bellman equation: Q = reward + γ × V(next_state)
- Adam optimizer with MSE loss

## 🙏 Credits

Inspired by classical Tetris AI techniques and modern deep RL approaches.
