# 🎮 HƯỚNG DẪN TRAINING - TETRIS AI

> **Mục đích:** Hướng dẫn chi tiết cách setup và treo máy training AI cho 2 người  
> **Thời gian training:** 48-72 giờ mỗi người  
> **Kết quả:** 2 models với experiences khác nhau để merge lại

---

## 📋 MỤC LỤC

1. [Chuẩn Bị Môi Trường](#1-chuẩn-bị-môi-trường)
2. [Cài Đặt Dependencies](#2-cài-đặt-dependencies)
3. [Phân Công Training](#3-phân-công-training)
4. [Chạy Training](#4-chạy-training)
5. [Theo Dõi Training](#5-theo-dõi-training)
6. [Xử Lý Khi Training Lỗi](#6-xử-lý-khi-training-lỗi)
7. [Thu Thập Kết Quả](#7-thu-thập-kết-quả)
8. [Merge Data và Final Training](#8-merge-data-và-final-training)

---

## 1. CHUẨN BỊ MÔI TRƯỜNG

### 1.1. Kiểm Tra Python

```powershell
# Kiểm tra Python version (cần >= 3.8)
python --version

# Nếu chưa có, download từ: https://www.python.org/downloads/
```

### 1.2. Tạo Virtual Environment

```powershell
# Di chuyển vào thư mục dự án
cd E:\WINDOW\BTL\TetrisAI

# Tạo virtual environment (nếu chưa có)
python -m venv .venv

# Kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1

# Nếu lỗi PowerShell execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1.3. Kiểm Tra Cấu Trúc Thư Mục

```
TetrisAI/
├── ai/                    # ✅ Core AI code
│   ├── __init__.py
│   ├── agent.py          # DQN Agent
│   ├── model.py          # Neural Network
│   ├── environment.py    # Game wrapper
│   ├── trainer.py        # Training loop
│   └── utils.py          # Utilities
├── TetrisAI/             # ✅ Game code
│   └── src/
│       ├── game.py
│       ├── shapes.py
│       └── config.py
├── models/               # Sẽ lưu trained models
├── logs/                 # Sẽ lưu training logs
├── data/                 # Sẽ lưu replay memories
├── train.py             # ✅ Training script
├── play_ai.py           # ✅ Testing script
└── requirements.txt      # ✅ Dependencies
```

---

## 2. CÀI ĐẶT DEPENDENCIES

### 2.1. Install Thư Viện AI/ML

```powershell
# Kích hoạt virtual environment trước
.\.venv\Scripts\Activate.ps1

# Install tất cả dependencies
pip install -r requirements.txt

# Nếu lỗi, install từng cái:
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install numpy==1.24.3
pip install pygame==2.5.2
pip install matplotlib==3.8.0
pip install pandas==2.1.0
pip install tqdm==4.66.0
```

### 2.2. Verify Installation

```powershell
# Test import
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import numpy as np; print('NumPy:', np.__version__)"
python -c "import pygame; print('Pygame:', pygame.__version__)"

# Nếu TensorFlow báo lỗi GPU, không sao - CPU vẫn chạy được
```

---

## 3. PHÂN CÔNG TRAINING

### 🎯 Chiến Lược: 2 người train với configs khác nhau

#### **👤 PERSON A - EXPLORATION CONFIG**
- **Mục tiêu:** Thu thập diverse experiences (khám phá nhiều)
- **Config:** High exploration, slow decay
- **Episodes:** 5000-10000
- **Ước tính:** 36-48 giờ

#### **👤 PERSON B - EXPLOITATION CONFIG**  
- **Mục tiêu:** Thu thập refined experiences (khai thác tốt)
- **Config:** Lower exploration, fast decay
- **Episodes:** 5000-10000
- **Ước tính:** 36-48 giờ

> **Tại sao chia như vậy?**  
> - Person A tạo nhiều experiences từ random exploration  
> - Person B tạo experiences từ learned strategies  
> - Merge lại → AI học từ cả 2 styles → Performance tốt hơn!

---

## 4. CHẠY TRAINING

### 4.1. Person A - Exploration Training

```powershell
# Kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1

# Chạy training với exploration config
python train.py --config exploration --episodes 10000 --name person_a_exploration

# Hoặc custom parameters:
python train.py `
    --episodes 10000 `
    --name person_a_exploration `
    --epsilon 1.0 `
    --epsilon-min 0.1 `
    --epsilon-decay 0.9995 `
    --lr 0.001 `
    --batch-size 64 `
    --checkpoint-freq 100
```

### 4.2. Person B - Exploitation Training

```powershell
# Kích hoạt virtual environment
.\.venv\Scripts\Activate.ps1

# Chạy training với exploitation config
python train.py --config exploitation --episodes 10000 --name person_b_exploitation

# Hoặc custom parameters:
python train.py `
    --episodes 10000 `
    --name person_b_exploitation `
    --epsilon 0.5 `
    --epsilon-min 0.01 `
    --epsilon-decay 0.995 `
    --lr 0.0005 `
    --batch-size 128 `
    --checkpoint-freq 100
```

### 4.3. Giải Thích Parameters

| Parameter | Ý nghĩa | Person A | Person B |
|-----------|---------|----------|----------|
| `--episodes` | Số episodes train | 10000 | 10000 |
| `--epsilon` | Exploration rate ban đầu | 1.0 (100%) | 0.5 (50%) |
| `--epsilon-min` | Exploration rate tối thiểu | 0.1 | 0.01 |
| `--epsilon-decay` | Tốc độ giảm exploration | 0.9995 (chậm) | 0.995 (nhanh) |
| `--lr` | Learning rate | 0.001 | 0.0005 |
| `--batch-size` | Batch size | 64 | 128 |
| `--checkpoint-freq` | Save checkpoint mỗi N episodes | 100 | 100 |

---

## 5. THEO DÕI TRAINING

### 5.1. Xem Training Progress

Training sẽ hiển thị real-time progress:

```
============================================================
🚀 STARTING TRAINING
============================================================
   Press Ctrl+C to stop training and save progress
============================================================

Episode     0 | Score:    245 | Lines:   3 | Steps:  456 | Reward:   125.3 | Loss: 0.0234 | ε: 0.998 | Mem:    456 | Time: 12.3s
Episode    10 | Score:    312 | Lines:   5 | Steps:  589 | Reward:   198.7 | Loss: 0.0198 | ε: 0.988 | Mem:   5890 | Time: 15.1s
Episode    20 | Score:    428 | Lines:   8 | Steps:  723 | Reward:   267.4 | Loss: 0.0165 | ε: 0.978 | Mem:  14468 | Time: 18.4s
...
```

**Giải thích các chỉ số:**
- **Score:** Điểm số đạt được trong episode
- **Lines:** Số dòng xóa được
- **Steps:** Số bước thực hiện
- **Reward:** Tổng reward (càng cao càng tốt)
- **Loss:** Training loss (càng thấp càng tốt)
- **ε (epsilon):** Exploration rate (giảm dần theo thời gian)
- **Mem:** Số experiences trong memory
- **Time:** Thời gian chạy episode

### 5.2. Kiểm Tra Files Đã Tạo

```powershell
# Xem checkpoints
ls models\person_a_exploration\checkpoints\

# Xem logs
ls logs\person_a_exploration\

# Đọc CSV log
Get-Content logs\person_a_exploration\training_log.csv | Select-Object -First 20
```

### 5.3. Training Progress Files

```
models/
└── person_a_exploration/
    ├── config.json                    # Training config
    ├── best_model_model.h5           # Best model weights
    ├── best_model_target.h5          # Best target network
    ├── best_model_memory.pkl         # Best model memory
    ├── best_model_state.pkl          # Best model state
    └── checkpoints/
        ├── checkpoint_00100_*.h5     # Checkpoint at episode 100
        ├── checkpoint_00200_*.h5     # Checkpoint at episode 200
        └── ...

logs/
└── person_a_exploration/
    ├── training_log.csv              # Detailed episode logs
    ├── training_stats.json           # Statistics
    └── summary.json                  # Final summary
```

---

## 6. XỬ LÝ KHI TRAINING LỖI

### 6.1. Nếu Training Bị Dừng Giữa Chừng

**Nguyên nhân:** Mất điện, crash, Ctrl+C...

**Giải pháp:** Resume từ checkpoint

```powershell
# Tìm checkpoint gần nhất
ls models\person_a_exploration\checkpoints\ | Sort-Object -Descending | Select-Object -First 1

# Resume training từ checkpoint
python train.py `
    --config exploration `
    --episodes 10000 `
    --name person_a_exploration `
    --resume models\person_a_exploration\checkpoints\checkpoint_02300
```

### 6.2. Lỗi "Out of Memory"

**Nguyên nhân:** RAM không đủ

**Giải pháp:**
```powershell
# Giảm batch size
python train.py --config exploration --batch-size 32 --episodes 10000

# Giảm memory capacity
python train.py --config exploration --memory 50000 --episodes 10000
```

### 6.3. Lỗi "Module not found"

**Nguyên nhân:** Chưa install dependencies hoặc chưa activate venv

**Giải pháp:**
```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Re-install dependencies
pip install -r requirements.txt
```

### 6.4. Training Quá Chậm

**Triệu chứng:** Mỗi episode > 30 giây

**Nguyên nhân:** TensorFlow chạy trên CPU

**Giải pháp:**
- ✅ Chấp nhận (CPU vẫn train được, chỉ lâu hơn)
- ⚡ Nếu có GPU: Install tensorflow-gpu
- 🚀 Giảm số episodes xuống 5000 thay vì 10000

---

## 7. THU THẬP KẾT QUẢ

### 7.1. Khi Training Xong

Training sẽ tự động save:
- ✅ Best model
- ✅ Latest checkpoints
- ✅ Replay memory
- ✅ Training logs

### 7.2. Test Model Đã Train

```powershell
# Test best model
python play_ai.py models\person_a_exploration\best_model --episodes 5

# Test một checkpoint cụ thể
python play_ai.py models\person_a_exploration\checkpoints\checkpoint_05000 --episodes 5
```

### 7.3. Phân Tích Kết Quả

```powershell
# Xem summary
Get-Content logs\person_a_exploration\summary.json | ConvertFrom-Json

# Phân tích chi tiết
python ai\utils.py analyze logs\person_a_exploration\training_log.csv

# Vẽ biểu đồ training progress
python ai\utils.py plot logs\person_a_exploration\training_log.csv logs\person_a_exploration\progress.png
```

### 7.4. Files Cần Share

**Person A chia sẻ:**
```
models/person_a_exploration/
├── best_model_memory.pkl      # 👈 QUAN TRỌNG - Replay memory
├── best_model_model.h5         # Model weights
├── config.json                 # Configuration
└── checkpoints/
    └── checkpoint_05000_memory.pkl  # Checkpoint memory (optional)

logs/person_a_exploration/
├── training_log.csv           # Training logs
└── summary.json               # Summary
```

**Person B chia sẻ:** Tương tự

---

## 8. MERGE DATA VÀ FINAL TRAINING

### 8.1. Sau Khi Cả 2 Người Train Xong

**Person A & B:** Upload files lên shared folder (Google Drive, OneDrive...)

```
shared_folder/
├── person_a_exploration/
│   ├── best_model_memory.pkl
│   ├── best_model_model.h5
│   └── summary.json
└── person_b_exploitation/
    ├── best_model_memory.pkl
    ├── best_model_model.h5
    └── summary.json
```

### 8.2. Merge Replay Memories

**Một trong 2 người chạy:**

```powershell
# Download 2 memory files về
# Merge lại thành 1 file

python ai\utils.py merge `
    models\person_a_exploration\best_model_memory.pkl `
    models\person_b_exploitation\best_model_memory.pkl `
    data\merged_memory.pkl
```

Output:
```
🔄 Merging replay memories...
   Loading: models\person_a_exploration\best_model_memory.pkl
      Added 87432 experiences
   Loading: models\person_b_exploitation\best_model_memory.pkl
      Added 95678 experiences

   Total experiences: 183110
✅ Merged memory saved to: data\merged_memory.pkl
   Final size: 183110 experiences
```

### 8.3. Final Training với Merged Data

```powershell
# Train final model với merged experiences
python train_final.py `
    --memory-file data\merged_memory.pkl `
    --episodes 3000 `
    --name final_model `
    --config balanced
```

> **Lưu ý:** File `train_final.py` cần thêm logic load merged memory

### 8.4. So Sánh Models

```powershell
# So sánh 3 models: A, B, và Final
python ai\utils.py compare `
    logs\person_a_exploration `
    logs\person_b_exploitation `
    logs\final_model `
    logs\comparison.png
```

---

## 🎯 CHECKLIST TRAINING

### Person A - Exploration

- [ ] Activate virtual environment
- [ ] Install dependencies
- [ ] Chạy training với exploration config
- [ ] Để máy chạy 36-48 giờ
- [ ] Kiểm tra training không bị lỗi
- [ ] Backup checkpoints thường xuyên
- [ ] Test best model
- [ ] Upload `best_model_memory.pkl` lên shared folder
- [ ] Share training logs và summary

### Person B - Exploitation

- [ ] Activate virtual environment
- [ ] Install dependencies
- [ ] Chạy training với exploitation config
- [ ] Để máy chạy 36-48 giờ
- [ ] Kiểm tra training không bị lỗi
- [ ] Backup checkpoints thường xuyên
- [ ] Test best model
- [ ] Upload `best_model_memory.pkl` lên shared folder
- [ ] Share training logs và summary

### Final Steps (Cả 2 cùng làm)

- [ ] Download cả 2 memory files
- [ ] Merge replay memories
- [ ] Train final model (optional)
- [ ] So sánh performance
- [ ] Viết báo cáo kết quả
- [ ] Demo final model

---

## ⚡ QUICK START COMMANDS

### Person A
```powershell
cd E:\WINDOW\BTL\TetrisAI
.\.venv\Scripts\Activate.ps1
python train.py --config exploration --episodes 10000 --name person_a_exploration
```

### Person B
```powershell
cd E:\WINDOW\BTL\TetrisAI
.\.venv\Scripts\Activate.ps1
python train.py --config exploitation --episodes 10000 --name person_b_exploitation
```

### Test Model
```powershell
python play_ai.py models\<experiment_name>\best_model --episodes 5
```

---

## 📞 TROUBLESHOOTING

| Vấn đề | Giải pháp |
|--------|-----------|
| Training quá lâu | Giảm episodes hoặc chấp nhận (CPU train chậm) |
| Out of memory | Giảm `--memory` hoặc `--batch-size` |
| Lỗi import module | Activate venv và install lại dependencies |
| Training dừng giữa chừng | Resume với `--resume checkpoint_path` |
| Model performance kém | Train thêm episodes hoặc tune hyperparameters |
| File không tìm thấy | Kiểm tra đường dẫn và cấu trúc thư mục |

---

## 📊 KẾT QUẢ MONG ĐỢI

### Sau 5000-10000 Episodes

**Person A (Exploration):**
- Score trung bình: 500-1500
- Lines cleared: 10-30
- Diverse experiences: ✅
- Memory size: ~80,000-100,000

**Person B (Exploitation):**
- Score trung bình: 800-2000
- Lines cleared: 15-40
- Refined strategies: ✅
- Memory size: ~80,000-100,000

**Final Model (Merged):**
- Score trung bình: 1000-3000+
- Lines cleared: 20-50+
- Best of both worlds: ✅

---

## 🎓 TÀI LIỆU THAM KHẢO

- **DQN Paper:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **TensorFlow Docs:** https://www.tensorflow.org/
- **Reinforcement Learning Intro:** https://spinningup.openai.com/

---

## ✅ DONE!

Sau khi đọc hướng dẫn này, bạn có thể:
1. ✅ Setup môi trường training
2. ✅ Chạy training độc lập
3. ✅ Xử lý lỗi cơ bản
4. ✅ Thu thập và chia sẻ kết quả
5. ✅ Merge data và train final model

**Good luck training! 🚀**

*Nếu có vấn đề, check lại từng bước trong guide này.*
