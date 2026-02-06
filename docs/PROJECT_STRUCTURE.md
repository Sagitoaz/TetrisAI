# 📂 CẤU TRÚC DỰ ÁN - TETRIS AI

```
TetrisAI/
│
├── 📚 docs/                         # Documentation
│   ├── TRAINING_GUIDE.md            # Hướng dẫn training chi tiết
│   ├── WORK_DIVISION.md             # Phân công công việc 2 người
│   ├── TESTING.md                   # Hướng dẫn test models
│   ├── ROADMAP.md                   # Roadmap dự án
│   └── PROJECT_STRUCTURE.md         # File này
│
├── 🤖 ai/                           # AI Core Code
│   ├── model.py                     # Neural Network
│   ├── agent.py                     # DQN Agent
│   ├── environment.py               # Game wrapper
│   ├── trainer.py                   # Training loop
│   └── utils.py                     # Utilities
│
├── 🎮 src/                          # Game Source Code
│   ├── game.py                      # Game logic
│   ├── shapes.py                    # Tetromino shapes
│   ├── tetromino.py                 # Tetromino class
│   ├── config.py                    # Game configuration
│   └── main.py                      # Game entry point
│
├── 🚀 scripts/                      # Executable Scripts
│   ├── train.py                     # Training script
│   ├── play_ai.py                   # Test AI
│   ├── train_final.py               # Final training
│   └── test_setup.py                # Verify setup
│
├── 💾 DATA (GENERATED)
│   ├── models/                      # Trained models
│   ├── logs/                        # Training logs
│   └── data/                        # Training data
│
├── ⚙️ CONFIG
│   ├── README.md                    # Main README
│   ├── requirements.txt             # Python dependencies
│   ├── .gitignore                   # Git ignore rules
│   └── .venv/                       # Virtual environment
```

---

## 📖 ĐỌC FILE NÀO?

### Bắt đầu
1. **README.md** (5 phút) - Overview và quick start
2. **TRAINING_GUIDE.md** (20 phút) - Chi tiết cách training

### Công việc
3. **WORK_DIVISION.md** - Phân công Person A & B
4. **TESTING.md** - Cách test models sau training

### Tham khảo
5. **ROADMAP.md** - Roadmap phát triển

---

## 🎯 FILES QUAN TRỌNG

### Code
- `ai/agent.py` - DQN algorithm core
- `ai/environment.py` - Reward function
- `ai/trainer.py` - Training loop

### Scripts
- `scripts/train.py` - Chạy training
- `scripts/play_ai.py` - Test AI
- `scripts/test_setup.py` - Kiểm tra setup

### Docs
- `docs/TRAINING_GUIDE.md` - **ĐỌC ĐẦU TIÊN**
- `docs/WORK_DIVISION.md` - Phân công
- `docs/TESTING.md` - Hướng dẫn test

---

## 🗂️ FILES ĐƯỢC TẠO SAU TRAINING

```
models/person_a_exploration/
├── best_model_model.h5          # Neural network weights
├── best_model_memory.pkl        # 🔴 SHARE FILE NÀY
├── config.json                  # Training config
└── checkpoints/
    └── checkpoint_*.h5          # Training checkpoints

logs/person_a_exploration/
├── training_log.csv             # Episode logs
└── summary.json                 # Statistics

data/
└── merged_memory.pkl            # Merged replay memory
```

---

## ✅ SETUP CHECKLIST

- [ ] Read README.md
- [ ] Read docs/TRAINING_GUIDE.md
- [ ] Activate venv: `.\.venv\Scripts\Activate.ps1`
- [ ] Install: `pip install -r requirements.txt`
- [ ] Verify: `python scripts\test_setup.py`
- [ ] Check: All folders exist (ai/, src/, scripts/, models/, logs/, data/)

---

**Cấu trúc đơn giản, dễ hiểu, sẵn sàng để train! 🚀**
