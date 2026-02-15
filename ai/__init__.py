"""
TetrisAI - Deep Q-Network Implementation
Core AI module for training Tetris playing agent
"""

__version__ = "2.0.0"
__author__ = "TetrisAI Team"

from .agent import DQNAgent
from .environment import Tetris

__all__ = ['DQNAgent', 'Tetris']
