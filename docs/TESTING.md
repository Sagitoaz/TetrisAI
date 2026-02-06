# 🧪 HƯỚNG DẪN TESTING

## 📋 Kiểm Tra Sau Khi Training

### 1️⃣ Test Model Đã Train

```powershell
# Test best model
python play_ai.py models\<experiment_name>\best_model --episodes 5

# Ví dụ:
python play_ai.py models\person_a_exploration\best_model --episodes 5
python play_ai.py models\person_b_exploitation\best_model --episodes 5
```

**Kết quả hiển thị:**
```
📺 Episode 1/5
▶️  Starting new episode...
   Steps: 100, Score: 245, Lines: 3
   Steps: 200, Score: 512, Lines: 7
   ...
✅ Episode 1 Results:
   Score: 1245
   Lines cleared: 21
   Level reached: 3
   Steps taken: 874
```

### 2️⃣ So Sánh Performance

```powershell
# So sánh 2 models
python ai\utils.py compare `
    logs\person_a_exploration `
    logs\person_b_exploitation `
    docs\comparison.png
```

### 3️⃣ Phân Tích Chi Tiết

```powershell
# Xem thống kê chi tiết
python ai\utils.py analyze logs\<experiment_name>\training_log.csv

# Vẽ biểu đồ training progress
python ai\utils.py plot logs\<experiment_name>\training_log.csv
```

### 4️⃣ Test Với Visual Rendering

```powershell
# Xem AI chơi trực quan (chậm hơn)
python play_ai.py models\<experiment_name>\best_model --episodes 3 --render --delay 0.1
```

---

## ✅ Tiêu Chí Đánh Giá

### Model Tốt

- ✅ **Score trung bình:** > 1000
- ✅ **Lines cleared:** > 20 per game
- ✅ **Ổn định:** Ít biến động giữa các games
- ✅ **Chiến thuật:** Tránh tạo holes, giữ height thấp

### Model Cần Cải Thiện

- ❌ **Score:** < 500
- ❌ **Lines:** < 10
- ❌ **Game over nhanh:** < 200 steps
- ❌ **Random moves:** Không có pattern rõ ràng

---

## 📊 So Sánh Models

| Model | Avg Score | Avg Lines | Best Score | Episodes Trained |
|-------|-----------|-----------|------------|------------------|
| Person A | 500-1500 | 10-30 | 2000-3000 | 10,000 |
| Person B | 800-2000 | 15-40 | 2500-4000 | 10,000 |
| Final | 1000-3000+ | 20-50+ | 3000-5000+ | 3,000 |

---

## 🔍 Debug & Troubleshoot

### Model Không Load Được

```powershell
# Kiểm tra file tồn tại
Test-Path models\<name>\best_model_model.h5
Test-Path models\<name>\best_model_state.pkl

# Nếu không có, dùng checkpoint
python play_ai.py models\<name>\checkpoints\checkpoint_05000 --episodes 5
```

### Performance Kém

**Nguyên nhân có thể:**
1. Training chưa đủ lâu → Train thêm episodes
2. Hyperparameters không tốt → Thử config khác
3. Reward function chưa phù hợp → Tune weights

**Giải pháp:**
```powershell
# Train thêm từ checkpoint
python train.py --resume models\<name>\checkpoints\checkpoint_XXXXX --episodes 15000
```

---

## 📈 Metrics Quan Trọng

### Training Metrics

| Metric | Ý nghĩa | Mục tiêu |
|--------|---------|----------|
| **Score** | Điểm số game | Tăng dần theo episodes |
| **Lines** | Số dòng xóa | > 20 per game |
| **Loss** | Training loss | Giảm dần, ổn định |
| **Epsilon** | Exploration rate | Giảm từ 1.0 → 0.01 |
| **Memory** | Experiences collected | Đầy (~100K) |

### Test Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| **Avg Score** | 1000-2000 | 2000-3000+ |
| **Avg Lines** | 20-35 | 35-50+ |
| **Best Score** | 2000-3000 | 3000-5000+ |
| **Consistency** | StdDev < 500 | StdDev < 300 |

---

## 🎯 Test Scenarios

### Scenario 1: Quick Test (5 games)
```powershell
python play_ai.py models\<name>\best_model --episodes 5
```
**Mục đích:** Kiểm tra nhanh model có hoạt động

### Scenario 2: Performance Test (20 games)
```powershell
python play_ai.py models\<name>\best_model --episodes 20
```
**Mục đích:** Đánh giá performance trung bình

### Scenario 3: Visual Demo
```powershell
python play_ai.py models\<name>\best_model --episodes 3 --render --delay 0.1
```
**Mục đích:** Demo trực quan cho presentation

---

## 📝 Test Report Template

```
=== TETRIS AI TEST REPORT ===

Model: person_a_exploration
Date: 2026-02-06
Episodes Tested: 20

Results:
- Average Score: 1245.6
- Average Lines: 24.3
- Best Score: 2156
- Worst Score: 678
- Standard Deviation: 345.2

Performance: GOOD ✅
- Scores > 1000 consistently
- Good line clearing strategy
- Avoids creating holes

Recommendations:
- Model ready for production
- Can be used for final merge
```

---

## 🔄 Continuous Testing

### After Every 1000 Episodes

```powershell
# Test checkpoint
python play_ai.py models\<name>\checkpoints\checkpoint_01000 --episodes 5

# Compare với previous
python ai\utils.py analyze logs\<name>\training_log.csv
```

### After Training Complete

```powershell
# Full evaluation
python play_ai.py models\<name>\best_model --episodes 20

# Generate report
python ai\utils.py report models\<name>
```

---

## ✅ Test Checklist

### Before Sharing Model

- [ ] Chạy test 5 episodes - no errors
- [ ] Average score > 800
- [ ] Model files exist (.h5, .pkl)
- [ ] Training logs complete
- [ ] Performance acceptable

### Before Merge

- [ ] Both models tested independently
- [ ] Performance comparison done
- [ ] Memory files verified (< 600MB)
- [ ] Ready for merge command

---

**🧪 Happy Testing! Nếu có vấn đề, xem lại TRAINING_GUIDE.md**
