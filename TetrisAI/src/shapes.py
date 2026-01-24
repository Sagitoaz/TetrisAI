"""
Định nghĩa các hình dạng Tetromino
"""

# Định nghĩa các hình dạng Tetromino (7 loại)
SHAPES = [
    [[1, 1, 1, 1]],              # I - hình thẳng
    [[1, 1], [1, 1]],            # O - hình vuông
    [[0, 1, 0], [1, 1, 1]],      # T - hình chữ T
    [[0, 1, 1], [1, 1, 0]],      # S - hình chữ S
    [[1, 1, 0], [0, 1, 1]],      # Z - hình chữ Z
    [[1, 0, 0], [1, 1, 1]],      # J - hình chữ J
    [[0, 0, 1], [1, 1, 1]],      # L - hình chữ L
]

# Tên các shape để dễ debug
SHAPE_NAMES = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
