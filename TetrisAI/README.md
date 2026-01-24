# Tetris AI Project

Project xây dựng game Tetris và AI để chơi tự động.

## Cấu trúc Project

```
TetrisAI/
├── src/                    # Source code chính
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Cấu hình và hằng số
│   ├── shapes.py          # Định nghĩa các hình dạng Tetromino
│   ├── tetromino.py       # Class Tetromino
│   └── game.py            # Class TetrisGame (logic chính)
├── main.py                # File chính để chạy game
├── tetris_game.py         # File cũ (có thể xóa)
├── requirements.txt       # Dependencies
└── README.md             # File này
```

## Cài đặt

1. **Cài đặt Python** (nếu chưa có): Python 3.8 trở lên

2. **Cài đặt thư viện:**
```bash
pip install -r requirements.txt
```

Hoặc:
```bash
pip install pygame
```

## Chạy Game

```bash
python main.py
```

## Điều khiển

- **← / →**: Di chuyển trái/phải
- **↑**: Xoay khối
- **↓**: Rơi chậm (soft drop) - tăng 1 điểm mỗi ô
- **Space**: Rơi nhanh (hard drop) - tăng 2 điểm mỗi ô
- **R**: Khởi động lại game

## Tính năng

✅ **Game Tetris hoàn chỉnh:**
- 7 loại khối Tetris với màu sắc riêng biệt
- Hiệu ứng 3D đẹp mắt cho các khối
- Xoay khối với wall-kick
- Hard drop và soft drop
- Hệ thống điểm số

✅ **Giao diện:**
- Hiển thị khối tiếp theo
- Thống kê: Score, Lines, Level
- Hướng dẫn điều khiển
- Màn hình Game Over

✅ **Gameplay:**
- Tốc độ tăng theo level
- Level tăng sau mỗi 10 dòng xóa
- Điểm số tăng theo số dòng xóa cùng lúc:
  - 1 dòng: 100 điểm × level
  - 2 dòng: 300 điểm × level
  - 3 dòng: 500 điểm × level
  - 4 dòng: 800 điểm × level

## Kế hoạch phát triển AI

Các bước tiếp theo để phát triển AI:

1. **Thu thập dữ liệu:** Ghi lại các trạng thái game và action
2. **Xây dựng heuristics:** Đánh giá trạng thái board
3. **Thuật toán AI:**
   - Genetic Algorithm
   - Deep Q-Learning (DQN)
   - hoặc các thuật toán khác

## Tác giả

Project học tập về AI và Game Development

## License

MIT License - Dùng tự do cho mục đích học tập
