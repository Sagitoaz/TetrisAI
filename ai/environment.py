"""
Tetris Environment - Value-Based Approach
Environment generates all possible states for current piece
Agent selects best state instead of sequence of actions
"""
import numpy as np
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tetromino import Tetromino
from src.config import GRID_WIDTH, GRID_HEIGHT
from src.shapes import SHAPES


class Tetris:
    """
    Tetris game environment for AI training
    Uses value-based approach: agent evaluates states, not actions
    """
    
    BOARD_WIDTH = GRID_WIDTH
    BOARD_HEIGHT = GRID_HEIGHT
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        self.board = [[0] * self.BOARD_WIDTH for _ in range(self.BOARD_HEIGHT)]
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        
        # Bag of pieces (7-bag random)
        self.bag = list(range(len(SHAPES)))
        random.shuffle(self.bag)
        self.current_piece = self.bag.pop()
        self.next_piece = self._get_next_piece()
        
        return self.get_state_properties(self.board)
    
    def _get_next_piece(self):
        """Get next piece from bag"""
        if len(self.bag) == 0:
            self.bag = list(range(len(SHAPES)))
            random.shuffle(self.bag)
        return self.bag.pop()
    
    def _get_rotated_piece(self, piece_id, rotation):
        """Get piece shape for given rotation (0, 90, 180, 270 degrees)"""
        shape = SHAPES[piece_id]
        
        for _ in range(rotation):
            # Rotate 90 degrees clockwise
            shape = [list(row) for row in zip(*shape[::-1])]
        
        return shape
    
    def _check_collision(self, piece_shape, pos):
        """Check if piece collides with board or boundaries"""
        for y, row in enumerate(piece_shape):
            for x, cell in enumerate(row):
                if cell:
                    board_x = pos[0] + x
                    board_y = pos[1] + y
                    
                    # Check boundaries
                    if (board_x < 0 or board_x >= self.BOARD_WIDTH or 
                        board_y < 0 or board_y >= self.BOARD_HEIGHT):
                        return True
                    
                    # Check collision with existing blocks
                    if self.board[board_y][board_x]:
                        return True
        
        return False
    
    def _add_piece_to_board(self, piece_shape, pos):
        """Create new board with piece added"""
        board = [row[:] for row in self.board]
        
        for y, row in enumerate(piece_shape):
            for x, cell in enumerate(row):
                if cell:
                    board[pos[1] + y][pos[0] + x] = 1
        
        return board
    
    def _clear_lines(self, board):
        """Clear full lines and return (lines_cleared, new_board)"""
        lines_to_clear = [i for i, row in enumerate(board) if all(row)]
        
        if lines_to_clear:
            # Remove full lines
            board = [row for i, row in enumerate(board) if i not in lines_to_clear]
            # Add empty lines at top
            for _ in lines_to_clear:
                board.insert(0, [0] * self.BOARD_WIDTH)
        
        return len(lines_to_clear), board
    
    def _count_holes(self, board):
        """Count holes (empty cells with filled cells above)"""
        holes = 0
        
        for col in range(self.BOARD_WIDTH):
            found_block = False
            for row in range(self.BOARD_HEIGHT):
                if board[row][col]:
                    found_block = True
                elif found_block:
                    holes += 1
        
        return holes
    
    def _get_bumpiness(self, board):
        """Calculate bumpiness (sum of height differences between adjacent columns)"""
        heights = self._get_column_heights(board)
        total_bumpiness = 0
        max_bumpiness = 0
        
        for i in range(len(heights) - 1):
            bumpiness = abs(heights[i] - heights[i + 1])
            total_bumpiness += bumpiness
            max_bumpiness = max(max_bumpiness, bumpiness)
        
        return total_bumpiness, max_bumpiness
    
    def _get_column_heights(self, board):
        """Get height of each column"""
        heights = []
        
        for col in range(self.BOARD_WIDTH):
            height = 0
            for row in range(self.BOARD_HEIGHT):
                if board[row][col]:
                    height = self.BOARD_HEIGHT - row
                    break
            heights.append(height)
        
        return heights
    
    def _get_height_stats(self, board):
        """Get aggregate height, max height, min height"""
        heights = self._get_column_heights(board)
        
        if not any(heights):
            return 0, 0, 0
        
        return sum(heights), max(heights), min(heights)
    
    def get_state_properties(self, board):
        """
        Extract state properties from board
        Returns: [lines_cleared, holes, total_bumpiness, aggregate_height]
        """
        lines, board = self._clear_lines(board)
        holes = self._count_holes(board)
        total_bumpiness, _ = self._get_bumpiness(board)
        aggregate_height, _, _ = self._get_height_stats(board)
        
        return [lines, holes, total_bumpiness, aggregate_height]
    
    def get_next_states(self):
        """
        Get all possible next states for current piece
        Returns dict: {(x, rotation): state_properties}
        """
        states = {}
        piece_id = self.current_piece
        
        # Determine rotations to try (skip duplicates for symmetric pieces)
        if piece_id == 6:  # O piece
            rotations = [0]
        elif piece_id == 0:  # I piece
            rotations = [0, 1]
        else:
            rotations = [0, 1, 2, 3]
        
        for rotation in rotations:
            piece_shape = self._get_rotated_piece(piece_id, rotation)
            
            # Get piece bounds
            min_x = min(x for y, row in enumerate(piece_shape) 
                       for x, cell in enumerate(row) if cell)
            max_x = max(x for y, row in enumerate(piece_shape) 
                       for x, cell in enumerate(row) if cell)
            
            # Try all x positions
            for x in range(-min_x, self.BOARD_WIDTH - max_x):
                pos = [x, 0]
                
                # Drop piece down
                while not self._check_collision(piece_shape, [pos[0], pos[1] + 1]):
                    pos[1] += 1
                
                # Check if position is valid (not spawning inside existing blocks)
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece_shape, pos)
                    states[(x, rotation)] = self.get_state_properties(board)
        
        return states
    
    def get_state_size(self):
        """Size of state vector"""
        return 4  # [lines, holes, bumpiness, height]
    
    def play(self, x, rotation):
        """
        Play a move (place piece at position x with given rotation)
        Returns: (reward, game_over)
        """
        piece_shape = self._get_rotated_piece(self.current_piece, rotation)
        pos = [x, 0]
        
        # Drop piece
        while not self._check_collision(piece_shape, [pos[0], pos[1] + 1]):
            pos[1] += 1
        
        # Check if game over (piece spawns at invalid position)
        if pos[1] < 0 or self._check_collision(piece_shape, pos):
            self.game_over = True
            return -2, True
        
        # Add piece to board
        self.board = self._add_piece_to_board(piece_shape, pos)
        
        # Clear lines
        lines, self.board = self._clear_lines(self.board)
        self.lines_cleared += lines
        
        # Calculate reward
        # Reward = 1 point for placing + (lines^2 * width) for clearing lines
        reward = 1 + (lines ** 2) * self.BOARD_WIDTH
        self.score += reward
        self.pieces_placed += 1
        
        # Get next piece
        self.current_piece = self.next_piece
        self.next_piece = self._get_next_piece()
        
        # Check if next piece can spawn
        if self._check_collision(self._get_rotated_piece(self.current_piece, 0), [3, 0]):
            self.game_over = True
            reward -= 2  # Penalty for game over
        
        return reward, self.game_over
    
    def get_game_score(self):
        """Get current game score"""
        return self.score
