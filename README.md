# 🎮 TetrisAI - Deep Q-Network Implementation

AI chơi Tetris sử dụng Deep Q-Network (DQN) - Reinforcement Learning.

---

## 📁 Cấu Trúc Dự Án

```
TetrisAI/
├── 📚 docs/                     # Documentation
│   ├── TRAINING_GUIDE.md        # Hướng dẫn training chi tiết
│   ├── WORK_DIVISION.md         # Phân công 2 người
│   ├── TESTING.md               # Hướng dẫn test
│   ├── ROADMAP.md               # Project roadmap
│   └── PROJECT_STRUCTURE.md     # Tổng quan cấu trúc
│
├── 🤖 ai/                       # Core AI Code
│   ├── model.py                 # Neural Network (DQN)
│   ├── agent.py                 # DQN Agent (Q-learning)
│   ├── environment.py           # Game wrapper cho AI
│   ├── trainer.py               # Training infrastructure
│   └── utils.py                 # Utilities (merge, analyze, plot)
│
├── 🎮 src/                      # Tetris Game Source
│   ├── game.py                  # Game logic
│   ├── shapes.py                # Tetromino shapes
│   ├── tetromino.py             # Tetromino class
│   ├── config.py                # Game configuration
│   └── main.py                  # Game entry point
│
├── 🚀 scripts/                  # Executable Scripts
│   ├── train.py                 # Training script
│   ├── play_ai.py               # Test AI
│   ├── train_final.py           # Final training (merged)
│   └── test_setup.py            # Setup verification
│
├── 💾 models/                   # Trained models (generated)
├── 📊 logs/                     # Training logs (generated)
├── 📁 data/                     # Training data (generated)
│
├── README.md                    # File này
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Cài Đặt

```powershell
# Activate virtual environment
cd E:\WINDOW\BTL\TetrisAI
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts\test_setup.py
```

### 2. Training

**Person A - Exploration:**
```powershell
python scripts\train.py --config exploration --episodes 10000 --name person_a_exploration
```

**Person B - Exploitation:**
```powershell
python scripts\train.py --config exploitation --episodes 10000 --name person_b_exploitation
```

### 3. Testing

```powershell
# Test trained AI
python scripts\play_ai.py models\person_a_exploration\best_model --episodes 5

# Analyze results
python ai\utils.py analyze logs\person_a_exploration\training_log.csv
```

### 4. Merge & Final Training

```powershell
# Merge memories
python ai\utils.py merge `
    models\person_a_exploration\best_model_memory.pkl `
    models\person_b_exploitation\best_model_memory.pkl `
    data\merged_memory.pkl

# Train final model
python scripts\train_final.py --memory-file data\merged_memory.pkl --episodes 3000
```

---

## 🎯 AI Architecture

### DQN Components

- **Neural Network:** 3 hidden layers (256-256-128)
- **Input:** 207 features (grid 20×10 + game state)
- **Output:** 7 Q-values (actions: left, right, rotate, drop...)
- **Training:** Experience Replay + Target Network
- **Exploration:** Epsilon-greedy (1.0 → 0.01)

### Reward Function

```python
+ Lines cleared: 40/100/300/1200 (single/double/triple/tetris)
+ Score increase: score_diff × 0.1
- Height increase: height_diff × 2
- Holes created: holes_created × 10
- Game over: -500
```

---

## 📖 Tài Liệu

| File | Mục Đích |
|------|----------|
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | Hướng dẫn chi tiết training |
| **[WORK_DIVISION.md](WORK_DIVISION.md)** | Phân công công việc 2 người |
| **[TESTING.md](TESTING.md)** | Hướng dẫn test models |
| **[ROADMAP.md](ROADMAP.md)** | Roadmap dự án |

---

## 🎓 Commands Cheat Sheet

| Task | Command |
|------|---------|
| **Setup** | `.\.venv\Scripts\Activate.ps1` |
| **Install** | `pip install -r TetrisAI\requirements.txt` |
| **Verify** | `python test_setup.py` |
| **Train A** | `python train.py --config exploration --episodes 10000 --name person_a` |
| **Train B** | `python train.py --config exploitation --episodes 10000 --name person_b` |
| **Test** | `python play_ai.py models\<name>\best_model --episodes 5` |
| **Resume** | `python train.py --resume models\<name>\checkpoints\checkpoint_XXXXX` |
| **Merge** | `python ai\utils.py merge mem1.pkl mem2.pkl output.pkl` |
| **Analyze** | `python ai\utils.py analyze logs\<name>\training_log.csv` |
| **Plot** | `python ai\utils.py plot logs\<name>\training_log.csv` |

---

## 🔧 Config Presets

| Preset | Person | Epsilon | Decay | Goal |
|--------|--------|---------|-------|------|
| **exploration** | A | 1.0 → 0.1 | 0.9995 (slow) | Diverse experiences |
| **exploitation** | B | 0.5 → 0.01 | 0.995 (fast) | Refined strategies |
| **balanced** | Final | 1.0 → 0.05 | 0.997 (medium) | Best performance |

---

## 📊 Expected Results

| Metric | Person A | Person B | Final Model |
|--------|----------|----------|-------------|
| Episodes | 10,000 | 10,000 | 3,000 |
| Training Time | 36-48h | 36-48h | 12-24h |
| Avg Score | 500-1500 | 800-2000 | 1000-3000+ |
| Avg Lines | 10-30 | 15-40 | 20-50+ |
| Memory Size | ~100K | ~100K | ~200K |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Activate venv: `.\.venv\Scripts\Activate.ps1` |
| Out of memory | Add: `--batch-size 32 --memory 50000` |
| Training too slow | Normal with CPU, reduce episodes or wait |
| Can't activate venv | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Training crashed | Resume: `--resume checkpoint_path` |

---

## 📦 Dependencies

```
tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
pygame==2.5.2
matplotlib==3.8.0
pandas==2.1.0
tqdm==4.66.0
h5py==3.10.0
```

---

## 👥 Workflow (2 Người)

1. **Setup:** Cả 2 cài đặt môi trường
2. **Training:** Person A & B train parallel (36-48h)
3. **Share:** Upload `best_model_memory.pkl` lên shared folder
4. **Merge:** Một người merge memories
5. **Final:** Train final model (12-24h)
6. **Test:** So sánh performance

---

## 📝 License

Educational project - TetrisAI Team 2026

---

**🚀 Ready to train? Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md)!**
