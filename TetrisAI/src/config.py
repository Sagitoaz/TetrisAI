"""
Cấu hình và hằng số cho game Tetris
"""

# Màu sắc cơ bản
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (200, 200, 200)

# Màu cho các khối Tetris
COLORS = [
    (0, 255, 255),    # Cyan - I
    (255, 255, 0),    # Yellow - O
    (128, 0, 128),    # Purple - T
    (0, 255, 0),      # Green - S
    (255, 0, 0),      # Red - Z
    (0, 0, 255),      # Blue - J
    (255, 165, 0),    # Orange - L
]

# Cài đặt kích thước
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH + 300
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT + 100

# Offset vị trí lưới
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50

# Điểm số cho mỗi dòng xóa
SCORES = {
    1: 100,
    2: 300,
    3: 500,
    4: 800
}

# Tốc độ rơi ban đầu (milliseconds)
INITIAL_FALL_SPEED = 500
MIN_FALL_SPEED = 100
SPEED_DECREASE_PER_LEVEL = 50

# FPS
FPS = 60
