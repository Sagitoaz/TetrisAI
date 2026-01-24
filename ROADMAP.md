# 🎮 Roadmap Chi Tiết: Xây Dựng AI Chơi Tetris

## 📋 Mục Lục
1. [Giới Thiệu Cơ Bản](#1-giới-thiệu-cơ-bản)
2. [Chuẩn Bị Môi Trường](#2-chuẩn-bị-môi-trường)
3. [Thiết Kế Game Environment](#3-thiết-kế-game-environment)
4. [Lựa Chọn Thuật Toán AI](#4-lựa-chọn-thuật-toán-ai)
5. [Thu Thập Dữ Liệu](#5-thu-thập-dữ-liệu)
6. [Xây Dựng Mô Hình AI](#6-xây-dựng-mô-hình-ai)
7. [Training (Huấn Luyện)](#7-training-huấn-luyện)
8. [Testing & Evaluation](#8-testing--evaluation)
9. [Tối Ưu Hóa](#9-tối-ưu-hóa)
10. [Deployment](#10-deployment)

---

## 1. Giới Thiệu Cơ Bản

### 1.1 AI Là Gì?
**Artificial Intelligence (AI)** là khả năng của máy tính để học và thực hiện các nhiệm vụ như con người. Trong trường hợp này, chúng ta muốn máy tính học cách chơi Tetris.

### 1.2 Machine Learning Là Gì?
**Machine Learning (ML)** là một nhánh của AI, nơi máy tính "học" từ dữ liệu thay vì được lập trình cứng nhắc. Có 3 loại chính:
- **Supervised Learning**: Học từ dữ liệu có nhãn (ví dụ: ảnh mèo/chó)
- **Unsupervised Learning**: Tìm pattern từ dữ liệu không nhãn
- **Reinforcement Learning (RL)**: Học từ trial and error (thử và sai) - **ĐÂY LÀ CÁCH TỐT NHẤT CHO TETRIS!**

### 1.3 Reinforcement Learning Cho Tetris
**Reinforcement Learning** hoạt động như sau:
```
Agent (AI) -> Hành động (di chuyển block) -> Môi trường (Game) -> 
Phần thưởng/Phạt (điểm số/game over) -> Agent học từ đó
```

**Ví dụ đơn giản:**
- AI đặt block vào vị trí tốt → +10 điểm → AI học: "Vị trí này tốt!"
- AI để block tạo lỗ trống → -5 điểm → AI học: "Không làm thế này nữa!"
- AI xóa nhiều dòng cùng lúc → +50 điểm → AI học: "Chiến thuật này hay!"

### 1.4 Các Thuật Toán Phổ Biến Cho Tetris
1. **Deep Q-Network (DQN)** ⭐ Recommended
   - Dễ implement
   - Hiệu quả tốt
   - Phù hợp cho người mới bắt đầu

2. **Genetic Algorithm (GA)**
   - Mô phỏng tiến hóa tự nhiên
   - Đơn giản về concept
   - Cần nhiều thời gian training

3. **Policy Gradient (PPO, A3C)**
   - Nâng cao hơn
   - Hiệu quả cao
   - Khó implement hơn

**Khuyến nghị: BẮT ĐẦU VỚI DQN!**

---

## 2. Chuẩn Bị Môi Trường

### 2.1 Kiến Thức Cần Có
**Mức độ cần thiết:**
- [x] Python cơ bản (biến, hàm, class, vòng lặp)
- [x] Pygame (đã có trong dự án)
- [ ] NumPy (thư viện tính toán)
- [ ] TensorFlow hoặc PyTorch (thư viện AI)
- [ ] Hiểu cơ bản về Neural Network

**Thời gian học:** 2-3 tuần nếu chưa biết gì

### 2.2 Cài Đặt Thư Viện

```bash
# Kích hoạt virtual environment
.venv\Scripts\activate

# Cài đặt thư viện AI
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install numpy==1.24.3
pip install matplotlib==3.8.0
pip install scikit-learn==1.3.0

# Thư viện để theo dõi training
pip install tensorboard==2.15.0
pip install tqdm==4.66.0

# Thư viện lưu model
pip install h5py==3.10.0
```

### 2.3 Cấu Trúc Thư Mục Mới

```
TetrisAI/
├── main.py                     # Game thông thường
├── src/
│   ├── game.py                 # Game logic
│   ├── shapes.py               # Tetromino shapes
│   └── config.py               # Cấu hình
├── ai/                         # ⭐ FOLDER MỚI
│   ├── __init__.py
│   ├── agent.py                # AI Agent (DQN)
│   ├── model.py                # Neural Network
│   ├── memory.py               # Replay Memory
│   ├── environment.py          # Game Environment cho AI
│   └── trainer.py              # Training logic
├── train.py                    # Script training
├── play_ai.py                  # Xem AI chơi
├── models/                     # Lưu trained models
│   ├── checkpoints/
│   └── best_model.h5
├── logs/                       # Training logs
│   └── tensorboard/
└── data/                       # Training data
    └── replays/
```

---

## 3. Thiết Kế Game Environment

### 3.1 Environment Là Gì?
**Environment** là phiên bản của game được thiết kế đặc biệt để AI có thể tương tác. Nó cần có:

1. **State (Trạng thái)**: Thông tin về game hiện tại
2. **Actions (Hành động)**: Những gì AI có thể làm
3. **Rewards (Phần thưởng)**: Điểm số cho mỗi hành động
4. **Done**: Game có kết thúc chưa

### 3.2 State Representation
**State** là những gì AI "nhìn thấy". Có nhiều cách biểu diễn:

#### Option 1: Grid-based (Đơn giản) ⭐ Recommended
```python
State = {
    'board': 20x10 matrix,        # Grid của game (0=trống, 1=có block)
    'current_piece': piece_type,   # Block hiện tại (0-6)
    'next_piece': piece_type,      # Block tiếp theo
    'holes': số lỗ trống,
    'height': chiều cao trung bình,
    'bumpiness': độ gồ ghề
}
```

#### Option 2: Feature-based (Nâng cao)
```python
State = [
    aggregate_height,      # Tổng chiều cao
    complete_lines,        # Số dòng hoàn chỉnh
    holes,                 # Số lỗ trống
    bumpiness,             # Độ gồ ghề
    wells,                 # Số "giếng" (cột sâu)
    max_height,            # Chiều cao max
    min_height,            # Chiều cao min
    # ... 20-30 features
]
```

### 3.3 Actions
AI có thể thực hiện các hành động:
```python
ACTIONS = {
    0: 'do_nothing',        # Không làm gì
    1: 'left',              # Di chuyển trái
    2: 'right',             # Di chuyển phải
    3: 'rotate_clockwise',  # Xoay phải
    4: 'rotate_counter',    # Xoay trái
    5: 'hard_drop',         # Thả xuống ngay
    6: 'soft_drop'          # Thả xuống từ từ
}
```

### 3.4 Reward Function (QUAN TRỌNG!)
**Reward function** quyết định AI học như thế nào:

```python
def calculate_reward(state_before, action, state_after):
    reward = 0
    
    # Phần thưởng cho việc xóa dòng
    lines_cleared = state_after['lines'] - state_before['lines']
    if lines_cleared == 1:
        reward += 40
    elif lines_cleared == 2:
        reward += 100
    elif lines_cleared == 3:
        reward += 300
    elif lines_cleared == 4:  # Tetris!
        reward += 1200
    
    # Phạt cho việc tăng chiều cao
    height_increase = state_after['height'] - state_before['height']
    reward -= height_increase * 2
    
    # Phạt cho việc tạo lỗ
    holes_created = state_after['holes'] - state_before['holes']
    reward -= holes_created * 10
    
    # Phạt cho độ gồ ghề
    reward -= state_after['bumpiness'] * 0.5
    
    # Phạt nặng nếu game over
    if state_after['game_over']:
        reward -= 500
    
    return reward
```

---

## 4. Lựa Chọn Thuật Toán AI

### 4.1 Deep Q-Network (DQN) - RECOMMENDED

#### 4.1.1 DQN Là Gì?
**DQN** kết hợp:
- **Q-Learning**: Thuật toán RL cổ điển
- **Deep Neural Network**: Mạng neural để học pattern phức tạp

**Cách hoạt động:**
1. AI nhìn state hiện tại
2. Neural Network dự đoán "giá trị" (Q-value) của mỗi action
3. AI chọn action có Q-value cao nhất
4. Thực hiện action, nhận reward
5. Cập nhật Neural Network để dự đoán tốt hơn

#### 4.1.2 DQN Components

**1. Neural Network Architecture:**
```
Input Layer: State (có thể là 200 neurons cho 20x10 grid)
    ↓
Hidden Layer 1: 256 neurons + ReLU activation
    ↓
Hidden Layer 2: 256 neurons + ReLU activation
    ↓
Hidden Layer 3: 128 neurons + ReLU activation
    ↓
Output Layer: 7 neurons (1 cho mỗi action)
```

**2. Experience Replay Memory:**
- Lưu lại các (state, action, reward, next_state) đã trải qua
- Random sample để training → tránh overfitting
- Capacity: 50,000 - 100,000 experiences

**3. Target Network:**
- Một network giống hệt main network
- Cập nhật chậm hơn → stable training
- Sync mỗi 1000 steps

**4. Epsilon-Greedy Exploration:**
```python
# Ban đầu: epsilon = 1.0 (100% random)
# Sau mỗi episode: epsilon *= 0.995
# Cuối cùng: epsilon = 0.01 (1% random, 99% exploit)

if random() < epsilon:
    action = random_action()  # Explore
else:
    action = best_action()    # Exploit
```

### 4.2 Genetic Algorithm (Alternative)

#### 4.2.1 GA Là Gì?
Mô phỏng tiến hóa tự nhiên:
1. Tạo 100 AI ngẫu nhiên (population)
2. Cho tất cả chơi Tetris
3. Chọn 20 AI tốt nhất (selection)
4. "Lai" chúng để tạo AI mới (crossover)
5. Thêm mutation ngẫu nhiên
6. Lặp lại bước 2

**Ưu điểm:**
- Concept dễ hiểu
- Không cần gradient descent

**Nhược điểm:**
- Training chậm hơn nhiều
- Cần nhiều game instances chạy song song

---

## 5. Thu Thập Dữ Liệu

### 5.1 Self-Play Data Collection
**DQN không cần dữ liệu có sẵn!** AI tự chơi và học từ chính nó.

#### Process:
1. AI chơi game với random actions (exploration)
2. Lưu mỗi step: (state, action, reward, next_state, done)
3. Sau mỗi N steps, training network từ saved data
4. Repeat

### 5.2 Expert Demonstrations (Optional)
Có thể thu thập dữ liệu từ người chơi giỏi:

```python
# Record human gameplay
def record_human_game():
    game = TetrisGame()
    replay = []
    
    while not game.game_over:
        state = game.get_state()
        action = get_human_input()  # Từ keyboard
        reward, next_state = game.step(action)
        
        replay.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })
    
    save_replay(replay)
```

Sau đó dùng **Imitation Learning** để AI học từ replay này.

---

## 6. Xây Dựng Mô Hình AI

### 6.1 Neural Network với TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_dqn_model(state_shape, action_size):
    """
    Xây dựng DQN model
    
    Args:
        state_shape: (20, 10) cho grid-based hoặc (25,) cho features
        action_size: 7 (số actions)
    
    Returns:
        Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Flatten(input_shape=state_shape),
        
        # Hidden layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),  # Tránh overfitting
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(128, activation='relu'),
        
        # Output layer (Q-values cho mỗi action)
        layers.Dense(action_size, activation='linear')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse'  # Mean Squared Error
    )
    
    return model
```

### 6.2 DQN Agent Class

```python
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 32
        
        # Models
        self.model = build_dqn_model(state_shape, action_size)
        self.target_model = build_dqn_model(state_shape, action_size)
        self.update_target_model()
        
        # Memory
        self.memory = ReplayMemory(capacity=100000)
    
    def update_target_model(self):
        """Copy weights từ model sang target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào memory"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """Chọn action dựa trên state"""
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Random
        
        q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
        return np.argmax(q_values)  # Best action
    
    def replay(self):
        """Training từ memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # Predict Q-values
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train model
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### 6.3 Replay Memory

```python
from collections import deque
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Random sample batch"""
        batch = random.sample(self.memory, batch_size)
        
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)
```

---

## 7. Training (Huấn Luyện)

### 7.1 Training Loop

```python
def train_dqn(episodes=10000):
    """Training DQN agent"""
    
    # Initialize
    env = TetrisEnvironment()
    agent = DQNAgent(
        state_shape=(20, 10),
        action_size=7
    )
    
    # Tracking
    scores = []
    best_score = 0
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Agent chọn action
            action = agent.act(state)
            
            # Thực hiện action
            next_state, reward, done = env.step(action)
            
            # Lưu vào memory
            agent.remember(state, action, reward, next_state, done)
            
            # Training
            agent.replay()
            
            # Update
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update target network mỗi 10 episodes
        if episode % 10 == 0:
            agent.update_target_model()
        
        # Logging
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        
        print(f"Episode {episode}/{episodes}")
        print(f"Score: {total_reward}, Avg: {avg_score:.2f}")
        print(f"Epsilon: {agent.epsilon:.3f}, Steps: {steps}")
        
        # Save best model
        if total_reward > best_score:
            best_score = total_reward
            agent.model.save('models/best_model.h5')
            print(f"New best score: {best_score}!")
        
        # Save checkpoint mỗi 100 episodes
        if episode % 100 == 0:
            agent.model.save(f'models/checkpoints/episode_{episode}.h5')
    
    return agent
```

### 7.2 Hyperparameter Tuning

**Các hyperparameters quan trọng:**

| Parameter | Mô tả | Giá trị đề xuất | Thử nghiệm |
|-----------|-------|----------------|------------|
| learning_rate | Tốc độ học | 0.0001 | 0.001, 0.0005, 0.00001 |
| gamma | Discount factor | 0.95 | 0.9, 0.99 |
| epsilon_decay | Giảm exploration | 0.995 | 0.99, 0.998 |
| batch_size | Số samples mỗi update | 32 | 16, 64, 128 |
| memory_capacity | Kích thước replay memory | 100000 | 50000, 200000 |
| hidden_layers | Số neurons | 256-256-128 | 512-512-256 |

**Cách thử nghiệm:**
1. Bắt đầu với giá trị đề xuất
2. Training 500-1000 episodes
3. Nếu không cải thiện, thử giá trị khác
4. Ghi lại kết quả trong file

### 7.3 Monitoring Training

**TensorBoard** để visualize:

```python
from tensorflow.keras.callbacks import TensorBoard

# Thêm callback
tensorboard_callback = TensorBoard(
    log_dir='logs/tensorboard',
    histogram_freq=1
)

# Sử dụng trong training
model.fit(X, y, callbacks=[tensorboard_callback])

# Xem trong terminal:
# tensorboard --logdir=logs/tensorboard
```

**Metrics cần theo dõi:**
- Episode reward (tăng dần = tốt)
- Average score (tính trên 100 episodes gần nhất)
- Epsilon (giảm dần từ 1.0 → 0.01)
- Loss (giảm dần)
- Lines cleared per game
- Survival time (số steps trước khi game over)

---

## 8. Testing & Evaluation

### 8.1 Evaluation Metrics

```python
def evaluate_agent(agent, num_games=100):
    """Đánh giá agent qua nhiều games"""
    
    scores = []
    lines_cleared = []
    survival_times = []
    
    for game in range(num_games):
        env = TetrisEnvironment()
        state = env.reset()
        
        total_score = 0
        steps = 0
        
        while True:
            # Agent chơi (không exploration)
            action = agent.act(state, epsilon=0)  # Greedy
            next_state, reward, done = env.step(action)
            
            total_score += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        scores.append(total_score)
        lines_cleared.append(env.lines_cleared)
        survival_times.append(steps)
    
    # Statistics
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Average Lines: {np.mean(lines_cleared):.2f}")
    print(f"Average Survival: {np.mean(survival_times):.2f} steps")
    
    return {
        'scores': scores,
        'lines': lines_cleared,
        'survival': survival_times
    }
```

### 8.2 Visualization

```python
import matplotlib.pyplot as plt

def plot_training_progress(scores):
    """Vẽ đồ thị training progress"""
    
    plt.figure(figsize=(12, 5))
    
    # Raw scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, label='Raw')
    plt.plot(np.convolve(scores, np.ones(100)/100, mode='valid'), 
             label='Moving Average (100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    
    # Distribution
    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
```

### 8.3 A/B Testing

So sánh 2 models:

```python
def compare_models(model_a, model_b, num_games=100):
    """So sánh 2 models"""
    
    results_a = evaluate_agent(model_a, num_games)
    results_b = evaluate_agent(model_b, num_games)
    
    print("\nModel A vs Model B:")
    print(f"Score: {np.mean(results_a['scores']):.2f} vs "
          f"{np.mean(results_b['scores']):.2f}")
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(results_a['scores'], results_b['scores'])
    
    if p_value < 0.05:
        winner = "A" if np.mean(results_a['scores']) > np.mean(results_b['scores']) else "B"
        print(f"Model {winner} is significantly better (p={p_value:.4f})")
    else:
        print(f"No significant difference (p={p_value:.4f})")
```

---

## 9. Tối Ưu Hóa

### 9.1 Curriculum Learning
Train AI theo từng cấp độ:

**Level 1: Slow speed**
- Piece drop mỗi 1 giây
- AI có nhiều thời gian suy nghĩ

**Level 2: Normal speed**
- Drop mỗi 0.5 giây

**Level 3: Fast speed**
- Drop mỗi 0.2 giây
- Gần giống game thật

### 9.2 Reward Shaping
Điều chỉnh reward function để AI học nhanh hơn:

```python
# Thay vì chỉ reward cho lines cleared
reward = lines_cleared * 100

# Thêm intermediate rewards
reward = (
    lines_cleared * 100 +
    height_decrease * 5 +
    holes_filled * 10 +
    tetris_setup * 20  # Reward cho việc setup Tetris
)
```

### 9.3 Multi-step Returns
Thay vì chỉ nhìn 1 step ahead, nhìn N steps:

```python
# N-step return
def calculate_n_step_return(rewards, gamma, n=5):
    return_val = 0
    for i in range(n):
        return_val += (gamma ** i) * rewards[i]
    return return_val
```

### 9.4 Prioritized Experience Replay
Thay vì random sample, ưu tiên experiences có loss cao:

```python
# Mỗi experience có priority
priority = abs(td_error) + epsilon

# Sample theo probability
prob = priority / sum(priorities)
```

### 9.5 Double DQN
Giảm overestimation của Q-values:

```python
# Thay vì:
target = reward + gamma * max(target_model(next_state))

# Dùng:
best_action = argmax(model(next_state))
target = reward + gamma * target_model(next_state)[best_action]
```

---

## 10. Deployment

### 10.1 Save/Load Model

```python
# Save
agent.model.save('models/final_model.h5')

# Load
from tensorflow import keras
model = keras.models.load_model('models/final_model.h5')
```

### 10.2 Play AI Script

```python
# play_ai.py
import pygame
from ai.agent import DQNAgent
from src.game import TetrisGame

def play_ai():
    # Load trained model
    agent = DQNAgent(state_shape=(20, 10), action_size=7)
    agent.model.load_weights('models/best_model.h5')
    agent.epsilon = 0  # No exploration
    
    # Initialize game
    game = TetrisGame()
    clock = pygame.time.Clock()
    
    while game.running:
        # Get state
        state = game.get_state()
        
        # AI chooses action
        action = agent.act(state)
        
        # Execute action
        game.execute_action(action)
        
        # Render
        game.render()
        clock.tick(10)  # 10 FPS

if __name__ == '__main__':
    play_ai()
```

### 10.3 Web Demo (Optional)
Deploy lên web với:
- Flask/FastAPI backend
- TensorFlow.js frontend
- User có thể chơi với AI hoặc xem AI chơi

---

## 📊 Timeline Ước Tính

| Phase | Công việc | Thời gian | Độ khó |
|-------|-----------|-----------|--------|
| 1 | Học cơ bản về AI/ML | 1-2 tuần | ⭐⭐ |
| 2 | Setup environment & dependencies | 1-2 ngày | ⭐ |
| 3 | Thiết kế game environment | 3-5 ngày | ⭐⭐⭐ |
| 4 | Implement DQN agent | 5-7 ngày | ⭐⭐⭐⭐ |
| 5 | First training attempt | 2-3 ngày | ⭐⭐ |
| 6 | Debug & tune | 1-2 tuần | ⭐⭐⭐⭐ |
| 7 | Optimize & improve | 1-2 tuần | ⭐⭐⭐ |
| 8 | Final testing & deployment | 3-5 ngày | ⭐⭐ |
| **TOTAL** | | **2-3 tháng** | |

---

## 🎯 Success Criteria

**Milestone 1: Baby AI** ✅
- AI có thể chơi được (không crash ngay lập tức)
- Average survival: 50-100 pieces
- Score: 100-500

**Milestone 2: Beginner AI** ✅✅
- AI chơi tốt hơn người mới bắt đầu
- Average survival: 200-500 pieces
- Score: 2000-5000
- Clear 10-30 lines per game

**Milestone 3: Intermediate AI** ✅✅✅
- AI chơi tốt hơn người trung bình
- Average survival: 500-1000 pieces
- Score: 10000-30000
- Clear 50-100 lines per game

**Milestone 4: Expert AI** 🏆
- AI chơi như pro player
- Average survival: >1000 pieces
- Score: >50000
- Clear >100 lines per game
- Có thể setup và execute Tetris liên tục

---

## 📚 Tài Nguyên Học Tập

### Courses (Miễn phí)
1. **Machine Learning Crash Course** - Google
   - https://developers.google.com/machine-learning/crash-course

2. **Deep Reinforcement Learning** - Hugging Face
   - https://huggingface.co/learn/deep-rl-course

3. **Practical RL** - Coursera
   - https://www.coursera.org/learn/practical-rl

### Papers
1. **Playing Atari with Deep Reinforcement Learning** (DQN paper)
   - https://arxiv.org/abs/1312.5602

2. **Human-level control through deep RL**
   - https://www.nature.com/articles/nature14236

### YouTube Channels
1. **Sentdex** - Python AI tutorials
2. **CodeBullet** - AI playing games (entertaining!)
3. **Two Minute Papers** - AI research summaries

### Tetris-specific
1. **"The Tetris AI" paper** - Pierre Dellacherie
2. **ColinFay Tetris AI** - GitHub implementation
3. **Tetris AI Competition** - Past results

---

## ⚠️ Common Pitfalls & Solutions

### Problem 1: AI không học được gì
**Triệu chứng:** Score không tăng sau 1000 episodes

**Nguyên nhân:**
- Reward function không hợp lý
- Learning rate quá cao/thấp
- Network architecture không phù hợp

**Giải pháp:**
- Kiểm tra reward values (log ra)
- Thử learning_rate từ 0.001 → 0.00001
- Tăng/giảm số neurons

### Problem 2: AI chỉ học một strategy
**Triệu chứng:** AI cứ làm một thứ (vd: chỉ xếp block qua bên phải)

**Nguyên nhân:**
- Epsilon decay quá nhanh
- State representation thiếu thông tin

**Giải pháp:**
- Tăng epsilon_decay (0.999 thay vì 0.995)
- Thêm features vào state
- Dùng epsilon annealing thay vì decay

### Problem 3: Training quá chậm
**Triệu chứng:** 1 episode mất >1 phút

**Nguyên nhân:**
- Game rendering mỗi frame
- Predict model mỗi action

**Giải pháp:**
- Tắt rendering khi training
- Batch predict nhiều states
- Dùng GPU nếu có

### Problem 4: Model overfitting
**Triệu chứng:** Training score cao nhưng test score thấp

**Nguyên nhân:**
- Train quá lâu trên cùng một data
- Network quá lớn

**Giải pháp:**
- Early stopping
- Thêm Dropout layers
- Regularization (L2)

---

## 🔧 Tools & Libraries

### Must-have
- [x] **Python 3.8+**
- [x] **TensorFlow/Keras** - AI framework
- [x] **NumPy** - Tính toán
- [x] **Pygame** - Game graphics

### Recommended
- [ ] **TensorBoard** - Visualize training
- [ ] **Matplotlib** - Plot graphs
- [ ] **tqdm** - Progress bars
- [ ] **Weights & Biases** - Experiment tracking (advanced)

### Optional
- [ ] **OpenAI Gym** - RL environment framework
- [ ] **Stable-Baselines3** - Pre-built RL algorithms
- [ ] **Ray RLlib** - Distributed RL

---

## 🎓 Learning Path

### Week 1-2: Fundamentals
- [ ] Học Python cơ bản (nếu chưa biết)
- [ ] Hiểu về Neural Networks
- [ ] Hiểu về Reinforcement Learning
- [ ] Xem tutorials về DQN

### Week 3-4: Implementation
- [ ] Setup environment
- [ ] Code game environment
- [ ] Implement DQN agent
- [ ] First training run

### Week 5-6: Training & Tuning
- [ ] Multiple training runs
- [ ] Tune hyperparameters
- [ ] Fix bugs
- [ ] Improve reward function

### Week 7-8: Optimization
- [ ] Implement improvements (Double DQN, etc.)
- [ ] Compare different approaches
- [ ] Achieve target performance

### Week 9-10: Polish & Deploy
- [ ] Clean up code
- [ ] Write documentation
- [ ] Create demo
- [ ] Share with community!

---

## 💡 Tips for Success

1. **Start Simple**: Đừng cố implement mọi thứ ngay. Bắt đầu với basic DQN.

2. **Log Everything**: Print ra mọi thứ - scores, epsilon, loss. Debug dễ hơn nhiều.

3. **Save Frequently**: Save model checkpoints thường xuyên. Training có thể crash.

4. **Visualize**: Xem AI chơi thực tế. Đừng chỉ nhìn numbers.

5. **Be Patient**: Training có thể mất vài ngày. Đừng nản!

6. **Compare with Baselines**: 
   - Random AI (baseline)
   - Heuristic AI (hand-coded rules)
   - Your DQN

7. **Ask for Help**: 
   - Stack Overflow
   - Reddit r/MachineLearning
   - Discord communities

8. **Document**: Ghi lại mọi experiment. Cái nào work, cái nào không.

---

## 🎮 Let's Start!

**Ready?** Chuyển sang file `WORK_DIVISION.md` để xem cách chia việc!

**Questions?** Đọc lại phần bạn chưa hiểu. Google thêm. Hỏi ChatGPT/Claude.

**Good luck!** 🚀

---

*Document này sẽ được cập nhật liên tục khi có thêm insights và improvements.*

*Version: 1.0*  
*Last updated: January 25, 2026*
