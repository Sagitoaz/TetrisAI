"""
Game Environment Wrapper for AI Training
Converts Tetris game into an RL environment
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game import TetrisGame
from src.config import GRID_WIDTH, GRID_HEIGHT


class TetrisEnvironment:
    """
    Tetris Environment for Reinforcement Learning
    
    Wraps the TetrisGame to provide:
    - State representation
    - Action space
    - Reward function
    - Step function for RL training
    """
    
    # Action mappings
    ACTIONS = {
        0: 'nothing',      # Do nothing (let piece fall)
        1: 'left',         # Move left
        2: 'right',        # Move right
        3: 'rotate',       # Rotate clockwise
        4: 'soft_drop',    # Soft drop (move down)
        5: 'hard_drop',    # Hard drop (instant drop)
        6: 'rotate_left'   # Rotate counter-clockwise (if needed)
    }
    
    def __init__(self, render=False):
        """
        Initialize environment
        
        Args:
            render: Whether to render the game visually
        """
        self.render_mode = render
        self.game = TetrisGame() if render else self._create_headless_game()
        
        # State dimensions
        self.grid_size = GRID_WIDTH * GRID_HEIGHT  # 20x10 = 200
        self.feature_size = 7  # Additional features
        self.state_size = self.grid_size + self.feature_size  # 207
        
        # Action space
        self.action_size = 7
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 5000
        
    def _create_headless_game(self):
        """Create game instance without pygame rendering"""
        import sys
        import os
        
        # Temporarily redirect pygame init
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        game = TetrisGame()
        return game
    
    def reset(self):
        """
        Reset environment to initial state
        
        Returns:
            Initial state representation
        """
        self.game.reset_game()
        self.episode_steps = 0
        return self._get_state()
    
    def step(self, action):
        """
        Execute action and return results
        
        Args:
            action: Action to execute (0-6)
            
        Returns:
            next_state: Next state after action
            reward: Reward received
            done: Whether episode ended
            info: Additional information
        """
        # Store previous state metrics
        prev_score = self.game.score
        prev_lines = self.game.lines_cleared
        prev_height = self._calculate_height()
        prev_holes = self._count_holes()
        
        # Execute action
        action_executed = self._execute_action(action)
        
        # Let piece fall naturally if no action or after action
        if not self.game.game_over:
            self.game.update(self.game.fall_speed)
        
        # Calculate reward
        reward = self._calculate_reward(prev_score, prev_lines, prev_height, prev_holes)
        
        # Get new state
        next_state = self._get_state()
        
        # Check if done
        done = self.game.game_over
        
        # Episode timeout check
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            done = True
        
        # Additional info
        info = {
            'score': self.game.score,
            'lines': self.game.lines_cleared,
            'level': self.game.level,
            'steps': self.episode_steps,
            'action_executed': action_executed
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action):
        """
        Execute the given action on the game
        
        Args:
            action: Action index (0-6)
            
        Returns:
            True if action was executed successfully
        """
        if self.game.game_over:
            return False
        
        action_name = self.ACTIONS.get(action, 'nothing')
        
        if action_name == 'left':
            return self.game.move(-1)
        elif action_name == 'right':
            return self.game.move(1)
        elif action_name == 'rotate':
            return self.game.rotate_piece()
        elif action_name == 'soft_drop':
            if not self.game.check_collision(self.game.current_piece, offset_y=1):
                self.game.current_piece.y += 1
                return True
            return False
        elif action_name == 'hard_drop':
            self.game.drop()
            return True
        elif action_name == 'rotate_left':
            # Rotate 3 times for counter-clockwise
            for _ in range(3):
                if not self.game.rotate_piece():
                    return False
            return True
        else:  # 'nothing'
            return True
    
    def _get_state(self):
        """
        Get current state representation
        
        Returns:
            State vector (numpy array of size state_size)
        """
        # Grid state (flattened)
        grid_flat = []
        for row in self.game.grid:
            for cell in row:
                grid_flat.append(1 if cell else 0)
        
        # Additional features
        features = [
            self._calculate_height() / GRID_HEIGHT,           # Normalized height
            self._count_holes() / (GRID_WIDTH * GRID_HEIGHT), # Normalized holes
            self._calculate_bumpiness() / GRID_HEIGHT,        # Normalized bumpiness
            self._count_complete_lines() / GRID_HEIGHT,       # Normalized complete lines
            self.game.current_piece.type / 7,                 # Current piece type (normalized)
            self.game.next_piece.type / 7,                    # Next piece type (normalized)
            self.game.lines_cleared / 100                     # Normalized lines cleared
        ]
        
        # Combine grid and features
        state = np.array(grid_flat + features, dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, prev_score, prev_lines, prev_height, prev_holes):
        """
        Calculate reward for the action taken
        
        This is the CORE of how AI learns!
        
        Args:
            prev_score: Score before action
            prev_lines: Lines cleared before action
            prev_height: Height before action
            prev_holes: Holes before action
            
        Returns:
            Reward value (float)
        """
        reward = 0.0
        
        # 1. Reward for clearing lines (MAJOR POSITIVE)
        lines_cleared = self.game.lines_cleared - prev_lines
        if lines_cleared > 0:
            # Exponential reward for clearing multiple lines
            line_rewards = {
                1: 40,      # Single
                2: 100,     # Double
                3: 300,     # Triple
                4: 1200     # Tetris!
            }
            reward += line_rewards.get(lines_cleared, 0)
        
        # 2. Reward for score increase
        score_diff = self.game.score - prev_score
        reward += score_diff * 0.1
        
        # 3. Penalty for height increase (want to keep it low)
        current_height = self._calculate_height()
        height_diff = current_height - prev_height
        reward -= height_diff * 2
        
        # 4. MAJOR penalty for creating holes
        current_holes = self._count_holes()
        holes_created = current_holes - prev_holes
        reward -= holes_created * 10
        
        # 5. Penalty for bumpiness (want smooth surface)
        bumpiness = self._calculate_bumpiness()
        reward -= bumpiness * 0.5
        
        # 6. MAJOR penalty for game over
        if self.game.game_over:
            reward -= 500
        
        # 7. Small reward for survival
        reward += 0.1
        
        return reward
    
    def _count_holes(self):
        """
        Count number of holes in the grid
        A hole is an empty cell with a filled cell above it
        
        Returns:
            Number of holes
        """
        holes = 0
        for x in range(GRID_WIDTH):
            block_found = False
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes
    
    def _calculate_height(self):
        """
        Calculate aggregate height of the grid
        Sum of heights of all columns
        
        Returns:
            Total height
        """
        heights = []
        for x in range(GRID_WIDTH):
            column_height = 0
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x]:
                    column_height = GRID_HEIGHT - y
                    break
            heights.append(column_height)
        
        return sum(heights) / len(heights) if heights else 0
    
    def _calculate_bumpiness(self):
        """
        Calculate bumpiness (surface roughness)
        Sum of absolute differences between adjacent column heights
        
        Returns:
            Bumpiness value
        """
        heights = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if self.game.grid[y][x]:
                    heights.append(GRID_HEIGHT - y)
                    break
            else:
                heights.append(0)
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
    def _count_complete_lines(self):
        """
        Count number of complete lines ready to clear
        
        Returns:
            Number of complete lines
        """
        complete = 0
        for row in self.game.grid:
            if all(row):
                complete += 1
        return complete
    
    def render(self):
        """Render the game if render mode is enabled"""
        if self.render_mode:
            self.game.draw()
    
    def close(self):
        """Close the environment"""
        if hasattr(self.game, 'screen'):
            import pygame
            pygame.quit()
    
    def get_metrics(self):
        """
        Get current game metrics for analysis
        
        Returns:
            Dictionary with game metrics
        """
        return {
            'score': self.game.score,
            'lines_cleared': self.game.lines_cleared,
            'level': self.game.level,
            'height': self._calculate_height(),
            'holes': self._count_holes(),
            'bumpiness': self._calculate_bumpiness(),
            'complete_lines': self._count_complete_lines()
        }
