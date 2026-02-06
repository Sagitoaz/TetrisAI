# 👥 Chia Việc: 2 Người Xây Dựng Tetris AI

## 📋 Tổng Quan

**Chiến lược: COLLABORATIVE DISTRIBUTED TRAINING**

- ✅ **Cả 2 cùng code AI** (Deep Q-Network - DQN)
- ✅ **Cùng xây dựng architecture** (Week 1-3)
- ✅ **Chia nhau train** với configs khác nhau (Week 4-5)
- ✅ **Gộp training data** từ 2 máy (Week 6)
- ✅ **Train final model** trên combined data (Week 7-8)

```
Phase 1 (Week 1-3): CÙNG XÂY DỰNG
├── Person A: Environment + Reward Function (AI logic)
└── Person B: Neural Network + DQN Agent (AI core)
    ↓ Integrate
    
Phase 2 (Week 4-5): PARALLEL TRAINING
├── Person A: Train config 1 → Save experiences
└── Person B: Train config 2 → Save experiences
    ↓ Upload to shared folder

Phase 3 (Week 6): GỘP DỮ LIỆU
├── Merge 200K experiences từ 2 người
└── Select best checkpoints

Phase 4 (Week 7-8): FINAL TRAINING
└── Train final model trên pooled data → Best AI
```

---

## 🎯 Phân Công Chi Tiết

### 👤 Person A: Game Environment & AI Reward Logic

**Vai trò:** Thiết kế cách AI "nhìn" game và "học" từ actions

**Week 1-3 Tasks:**
- [ ] **Environment Wrapper** (`ai/environment.py`)
  - Wrap game Tetris để AI có thể interact
  - Extract state từ game (board matrix 20x10)
  - Map 7 actions (left, right, rotate, drop...)
  - Implement step function (action → next_state, reward, done)
  
- [ ] **Reward Function Design** (🔥 QUAN TRỌNG - Core AI logic)
  - Thiết kế reward function để AI học chiến thuật
  - +points khi xóa dòng (40, 100, 300, 1200)
  - -points khi tạo holes, tăng height
  - Implement helper functions (count_holes, calc_height, calc_bumpiness)
  - Tune reward weights để AI học tốt
  
- [ ] **Training Infrastructure** (`ai/trainer.py`)
  - Training loop manager
  - Checkpoint saving/loading
  - Replay memory serialization (để share với Person B)
  - Logging system (CSV, TensorBoard)

**Week 4-5: Training Track 1**
- [ ] Train AI với **exploration-heavy config**
  - High epsilon (1.0 → 0.1)
  - Slow decay (0.9995)
  - Collect diverse experiences
  - Train 5000 episodes
  - Save replay memory → Upload shared folder

**Week 6: Data Analysis**
- [ ] Analyze training data từ cả 2 tracks
- [ ] Merge replay memories
- [ ] Compare checkpoint performance

**Week 7-8: Final Optimization**
- [ ] Tune final reward function
- [ ] Create visualization tools
- [ ] Documentation

---

### 👤 Person B: Neural Network & DQN Algorithm

**Vai trò:** Thiết kế "bộ não" AI và thuật toán học

**Week 1-3 Tasks:**
- [ ] **Neural Network Model** (`ai/model.py`)
  - Build DQN architecture với TensorFlow/Keras
  - Input: 20x10 grid
  - Hidden layers: Dense(256), Dense(256), Dense(128)
  - Output: 7 Q-values (1 cho mỗi action)
  - Compile với optimizer (Adam) và loss (MSE)
  
- [ ] **DQN Agent** (`ai/agent.py`) (🔥 CORE AI ALGORITHM)
  - Implement Q-Learning algorithm
  - Experience Replay Memory (deque 100K capacity)
  - Target Network (stable Q-learning)
  - Epsilon-greedy policy (exploration vs exploitation)
  - `act()` method: Choose action based on Q-values
  - `remember()`: Store experiences
  - `replay()`: Train from batch samples (Bellman equation)
  - Model save/load functions

**Week 4-5: Training Track 2**
- [ ] Train AI với **exploitation-heavy config**
  - Lower epsilon (0.5 → 0.01)
  - Fast decay (0.995)
  - Collect refined experiences
  - Train 5000 episodes
  - Save replay memory → Upload shared folder

**Week 6: Model Selection**
- [ ] Evaluate all checkpoints (both tracks)
- [ ] Select top 5 models
- [ ] Implement ensemble methods

**Week 7-8: Final Model**
- [ ] Optimize neural network architecture
- [ ] Fine-tune hyperparameters
- [ ] Code documentation

---

## 📅 Timeline Chi Tiết

### **WEEK 1: Foundation (CÙNG HỌC)**

**Cả 2 người:**
- [ ] Học DQN basics (videos, papers)
- [ ] Hiểu Q-Learning, Experience Replay, Target Network
- [ ] Phân tích game code (`tetris_game.py`, `src/game.py`)
- [ ] Cùng design: State representation, Action space, Reward strategy
- [ ] Meeting: Agree on architecture
- [ ] Setup project structure, Git, shared folder

**Deliverables:**
- Docs: `docs/DESIGN.md` (cả 2 cùng viết)
- Hiểu rõ DQN algorithm

---

### **WEEK 2: Core Implementation (CHIA VIỆC)**

**Person A:**
- [ ] Code `ai/environment.py`
  - `reset()`, `step()`, `_get_state()`
  - `_execute_action()` mapping
  - `_get_metrics()` helpers
- [ ] Code reward function
  - `_calculate_reward()`
  - `_count_holes()`, `_calculate_height()`, `_calculate_bumpiness()`
- [ ] Test environment manually

**Person B:**
- [ ] Code `ai/model.py`
  - `build_dqn_model()` function
  - Neural network architecture
- [ ] Code `ai/agent.py` start
  - `ReplayMemory` class
  - `DQNAgent.__init__()`
  - `act()` method (epsilon-greedy)
- [ ] Test model creation

**Cả 2:**
- Code review lẫn nhau
- Help debug

---

### **WEEK 3: Integration (CÙNG LÀM)**

**Person A:**
- [ ] Finish `ai/trainer.py`
  - Training loop
  - Checkpoint management
  - Memory serialization

**Person B:**
- [ ] Finish `ai/agent.py`
  - `remember()` method
  - `replay()` method (Q-learning update)
  - Target network sync
- [ ] Implement Double DQN (optional improvement)

**Cả 2 cùng:**
- [ ] Integrate code
- [ ] Create `train.py` main script
- [ ] Test training loop (50 episodes test)
- [ ] Fix bugs together
- [ ] Verify checkpoints & memory save correctly
- [ ] Code review & cleanup

**Deliverables:**
- Working DQN implementation
- Can train without crashes

---

### **WEEK 4-5: Parallel Training (CHIA RA)**

**Person A - Config 1 (Exploration):**
```
Hyperparameters:
- epsilon: 1.0 → 0.1
- epsilon_decay: 0.9995 (slow)
- learning_rate: 0.0001
- batch_size: 32
```
- [ ] Run training: `python train.py --config config_A.json --episodes 5000`
- [ ] Monitor daily (scores, epsilon, loss)
- [ ] Save checkpoints mỗi 500 episodes
- [ ] Upload to `shared/track_A/`

**Person B - Config 2 (Exploitation):**
```
Hyperparameters:
- epsilon: 0.5 → 0.01
- epsilon_decay: 0.995 (fast)
- learning_rate: 0.00005
- batch_size: 64
```
- [ ] Run training: `python train.py --config config_B.json --episodes 5000`
- [ ] Monitor daily (scores, epsilon, loss)
- [ ] Save checkpoints mỗi 500 episodes
- [ ] Upload to `shared/track_B/`

**Daily:** 15-min sync call
**Mid-week:** Compare progress
**End:** Verify data uploaded

**Deliverables:**
- 10,000 total episodes trained
- ~200K experiences collected
- ~20 checkpoints total

---

### **WEEK 6: Data Pooling (CÙNG LÀM)**

**Day 1-2: Merge Data**
- [ ] Person A: Write merge script
- [ ] Person B: Verify data integrity
- [ ] Cả 2: Run merge → `memory_combined.pkl` (200K experiences)

**Day 3-4: Evaluate Checkpoints**
- [ ] Person B: Write evaluation script
- [ ] Person A: Run evaluation
- [ ] Cả 2: Analyze results, select top 5 models

**Day 5-7: Prepare Final Training**
- [ ] Cả 2: Design final training strategy
- [ ] Person A: Setup final training config
- [ ] Person B: Implement ensemble methods
- [ ] Prepare scripts for week 7

**Deliverables:**
- Combined dataset ready
- Best checkpoints identified
- Final training plan

---

### **WEEK 7: Final Training (CÙNG LÀM)**

**Day 1-4: Train Final Model**
- [ ] Load combined replay memory
- [ ] Person B: Monitor training
- [ ] Person A: Track metrics & save checkpoints
- [ ] Train 3000 episodes
- [ ] Cả 2: Daily review & tune if needed

**Day 5-7: Ensemble & Evaluation**
- [ ] Person B: Create ensemble (top 3-5 models)
- [ ] Person A: Run 100-game evaluation
- [ ] Cả 2: Compare results
- [ ] Select best final model

**Deliverables:**
- Final trained model
- Evaluation results
- Performance metrics

---

### **WEEK 8: Documentation (CHIA VIỆC)**

**Person A Tasks:**
- [ ] Write `README.md` - Project overview
- [ ] Write `docs/TRAINING.md` - Training process
- [ ] Create visualizations (training curves, charts)
- [ ] Record demo video (AI playing)

**Person B Tasks:**
- [ ] Write `docs/RESULTS.md` - Evaluation results
- [ ] Write `docs/ARCHITECTURE.md` - Technical details
- [ ] Code cleanup & comments
- [ ] Create result tables & statistics

**Cùng làm:**
- [ ] Create presentation (15-20 slides)
- [ ] Prepare demo for submission
- [ ] Final code review
- [ ] Test everything one last time

**Deliverables:**
- Complete documentation
- Demo video
- Presentation
- Clean codebase

---

## 📁 Folder Structure

```
TetrisAI/
├── ai/                          # AI Code (Week 1-3: BOTH)
│   ├── environment.py           # Person A lead
│   ├── model.py                 # Person B lead
│   ├── agent.py                 # Person B lead
│   ├── trainer.py               # Person A lead
│   └── utils.py
│
├── configs/
│   ├── config_A.json           # Person A training config
│   ├── config_B.json           # Person B training config
│   └── config_final.json
│
├── shared/                      # Shared training data
│   ├── track_A/                # Person A checkpoints
│   ├── track_B/                # Person B checkpoints
│   ├── replay_memories/
│   │   ├── memory_A.pkl
│   │   ├── memory_B.pkl
│   │   └── memory_combined.pkl
│   └── final_model/
│
├── scripts/                     # Utility scripts
│   ├── merge_memories.py       # Week 6
│   ├── evaluate.py
│   └── ensemble.py
│
├── docs/
│   ├── DESIGN.md               # Week 1
│   ├── TRAINING.md             # Week 8
│   ├── RESULTS.md              # Week 8
│   └── ARCHITECTURE.md         # Week 8
│
├── train.py                     # Main training script
├── train_final.py               # Final training
└── README.md
```

---

## 🤝 Collaboration

### Daily (15 min):
- Morning: Check training status
- Evening: Share progress, discuss issues

### Weekly (1 hour):
- Monday: Week planning
- Wednesday: Code review / integration
- Friday: Week review

### Tools:
- **Git**: Code sharing (branches: `feature/environment`, `feature/agent`)
- **Google Drive/Dropbox**: Shared data folder
- **Discord/Telegram**: Daily communication
- **Google Docs**: Collaborative docs

---

## 📊 Workload Split

| Phase | Person A | Person B | Together |
|-------|----------|----------|----------|
| Week 1 | 30% | 30% | 40% |
| Week 2 | 50% | 50% | - |
| Week 3 | 30% | 30% | 40% |
| Week 4-5 | 50% | 50% | - |
| Week 6 | 40% | 40% | 20% |
| Week 7 | 40% | 40% | 20% |
| Week 8 | 50% | 50% | - |

**Total: Fair 50/50 split**

---

## 🎯 Success Criteria

### Week 3 ✅
- Cả 2 hiểu DQN
- Code integrated
- Can train 50 episodes

### Week 5 ✅✅
- 5000 episodes mỗi người
- 200K experiences collected
- AI can play (score >500)

### Week 6 ✅✅✅
- Data merged successfully
- Best checkpoints identified

### Week 8 ✅✅✅✅
- Final model trained
- Score >5000 average
- Complete documentation
- Demo ready

---

## 🚨 Troubleshooting

### AI không học:
- [ ] Person A: Check reward function (log values, test manually)
- [ ] Person B: Check network (loss decreasing? gradients OK?)
- [ ] Cả 2: Review code together, simplify if needed

### Training chậm:
- [ ] Tắt rendering
- [ ] Reduce episodes (5000 → 3000)
- [ ] Use Google Colab GPU
- [ ] Train overnight

### Hết thời gian:
**Minimum (Week 5):** Basic training done
**Target (Week 7):** Data pooled, final training done
**Ideal (Week 8):** Everything polished

**Emergency:** Skip ensemble, use best single model, focus on solid report

---

## 💡 Key Points

### Tại sao chia việc này tốt:

✅ **Cả 2 đều code AI:**
- Person A: AI reward logic + training infrastructure
- Person B: AI neural network + learning algorithm

✅ **Collaboration thực sự:**
- Week 1,3,6,7: Work together
- Week 2,4-5,8: Independent but coordinated

✅ **Data pooling unique:**
- 2 training strategies → diverse data
- Combined = better final model
- Giống real ML teams

✅ **Fair & educational:**
- 50/50 workload
- Both learn full DQN
- Portfolio-worthy project

### Những việc CẢ 2 đều làm:

1. **Code AI components** (environment/model/agent)
2. **Implement DQN algorithm**
3. **Train models** (parallel tracks)
4. **Evaluate & analyze** results
5. **Document & present**

**→ Không ai chỉ làm "supporting tasks"!**

---

## 📚 Learning Resources

**Week 1:**
- "Neural Networks" - 3Blue1Brown (YouTube)
- "Deep Q-Learning" - DeepMind
- Paper: "Playing Atari with Deep RL"

**Week 2-3:**
- TensorFlow/Keras tutorials
- DQN implementations on GitHub
- Stack Overflow for bugs

**Week 4-8:**
- Hyperparameter tuning guides
- Ensemble methods
- Documentation best practices

---

## 🎉 Expected Outcome

**Technical:**
- Working DQN agent scoring >5000
- 200K training experiences
- Multiple model checkpoints
- Ensemble model

**Learning:**
- Deep understanding of DQN
- Experience with TensorFlow/Keras
- Reinforcement Learning concepts
- Teamwork & collaboration
- ML project workflow

**Portfolio:**
- Complete AI project
- Clean codebase
- Documentation
- Demo video
- Presentation

---

## ✅ Quick Start

1. **Cả 2 đọc hết file này**
2. **Discuss & agree trên approach**
3. **Setup: Git, shared folder, environment**
4. **Week 1 Day 1: Start learning together!**

---

**Good luck! Đây sẽ là một project tuyệt vời cho môn AI!** 🚀🤖

*Version: 4.0 - Task-Focused Collaborative Approach*  
*Last updated: January 25, 2026*
