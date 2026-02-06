"""
Class Tetromino - đại diện cho một khối trong game
"""
import random
from .shapes import SHAPES
from .config import COLORS, GRID_WIDTH


class Tetromino:
    """Class đại diện cho một khối Tetromino"""
    
    def __init__(self, shape_idx=None):
        """
        Khởi tạo một khối Tetromino
        
        Args:
            shape_idx: Index của hình dạng (0-6), None để random
        """
        if shape_idx is None:
            shape_idx = random.randint(0, len(SHAPES) - 1)
        
        self.shape = SHAPES[shape_idx]
        self.color = COLORS[shape_idx]
        self.shape_idx = shape_idx
        
        # Vị trí ban đầu (giữa màn hình, trên cùng)
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.rotation = 0

    def rotate(self):
        """Xoay khối 90 độ theo chiều kim đồng hồ"""
        self.shape = list(zip(*self.shape[::-1]))
        return self

    def get_rotated_shape(self):
        """
        Lấy hình dạng sau khi xoay mà không thay đổi khối hiện tại
        
        Returns:
            Hình dạng sau khi xoay
        """
        return list(zip(*self.shape[::-1]))
    
    def copy(self):
        """Tạo bản sao của khối"""
        new_piece = Tetromino(self.shape_idx)
        new_piece.shape = [row[:] for row in self.shape]
        new_piece.x = self.x
        new_piece.y = self.y
        new_piece.rotation = self.rotation
        return new_piece
