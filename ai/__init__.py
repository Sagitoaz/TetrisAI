"""
TetrisAI - Deep Q-Network Implementation
Core AI module for training Tetris playing agent
"""

__version__ = "1.0.0"
__author__ = "TetrisAI Team"

from .model import DQNModel
from .agent import DQNAgent
from .environment import TetrisEnvironment
from .trainer import Trainer

__all__ = ['DQNModel', 'DQNAgent', 'TetrisEnvironment', 'Trainer']
