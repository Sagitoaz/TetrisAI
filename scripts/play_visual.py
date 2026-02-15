"""
Watch AI Play Tetris with Visual Interface
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from ai.agent import DQNAgent
from ai.environment import Tetris
from src.config import *
from src.shapes import SHAPES


# Colors for visualization
COLORS = [
    (0, 0, 0),        # Empty
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
]

GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BG_COLOR = (20, 20, 20)

# Window settings
CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
INFO_WIDTH = 250
WINDOW_WIDTH = CELL_SIZE * BOARD_WIDTH + INFO_WIDTH
WINDOW_HEIGHT = CELL_SIZE * BOARD_HEIGHT


class TetrisVisualizer:
    """Visualize AI playing Tetris"""
    
    def __init__(self, model_path='models/best_lines.keras', fps=10):
        """
        Initialize visualizer
        
        Args:
            model_path: Path to trained model
            fps: Display update speed (frames per second)
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Tetris AI - Visual Play')
        self.clock = pygame.time.Clock()
        self.fps = fps
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Load AI
        print(f"Loading model: {model_path}")
        self.env = Tetris()
        self.agent = DQNAgent(
            state_size=self.env.get_state_size(),
            n_neurons=[32, 32],
            activations=['relu', 'relu', 'linear']
        )
        self.agent.load_model(model_path)
        self.agent.epsilon = 0  # No exploration
        print("Model loaded successfully!")
        
        # Game stats
        self.episode = 0
        self.running = True
        self.paused = False
        
    def draw_board(self, board, current_piece_shape=None, piece_pos=None, ghost_pos=None):
        """Draw the game board with current piece and ghost"""
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                cell = board[y][x]
                color = COLORS[cell if cell < len(COLORS) else 0]
                
                # Check if this cell is part of ghost piece
                is_ghost = False
                if ghost_pos and current_piece_shape:
                    for py, row in enumerate(current_piece_shape):
                        for px, pcell in enumerate(row):
                            if pcell and x == ghost_pos[0] + px and y == ghost_pos[1] + py:
                                is_ghost = True
                                break
                
                # Check if this cell is part of current piece
                is_current = False
                if piece_pos and current_piece_shape:
                    for py, row in enumerate(current_piece_shape):
                        for px, pcell in enumerate(row):
                            if pcell and x == piece_pos[0] + px and y == piece_pos[1] + py:
                                is_current = True
                                color = COLORS[(self.env.current_piece % len(COLORS)) + 1]
                                break
                
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                
                if is_current:
                    # Draw current piece (bright)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, WHITE, rect, 2)
                elif is_ghost:
                    # Draw ghost piece (semi-transparent)
                    ghost_color = (100, 100, 100)
                    pygame.draw.rect(self.screen, ghost_color, rect)
                    pygame.draw.rect(self.screen, GRAY, rect, 1)
                else:
                    # Draw normal cell
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, GRAY, rect, 1)
    
    def draw_info_panel(self, stats):
        """Draw information panel"""
        panel_x = BOARD_WIDTH * CELL_SIZE
        
        # Background
        pygame.draw.rect(
            self.screen,
            (30, 30, 30),
            (panel_x, 0, INFO_WIDTH, WINDOW_HEIGHT)
        )
        
        y_offset = 30
        
        # Title
        title = self.font_medium.render("TETRIS AI", True, WHITE)
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 60
        
        # Stats
        info_items = [
            ("Episode", stats['episode']),
            ("Score", stats['score']),
            ("Lines", stats['lines']),
            ("Pieces", stats['pieces']),
        ]
        
        for label, value in info_items:
            text = self.font_small.render(f"{label}:", True, WHITE)
            self.screen.blit(text, (panel_x + 20, y_offset))
            
            value_text = self.font_medium.render(str(value), True, (0, 255, 100))
            self.screen.blit(value_text, (panel_x + 20, y_offset + 25))
            
            y_offset += 70
        
        # Controls
        y_offset += 40
        controls = [
            "Controls:",
            "SPACE - Pause",
            "R - Reset",
            "UP/DOWN - Speed",
            "ESC - Quit"
        ]
        
        for i, text in enumerate(controls):
            color = WHITE if i == 0 else (150, 150, 150)
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (panel_x + 20, y_offset + i * 25))
        
        # Pause indicator
        if self.paused:
            pause_text = self.font_large.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(panel_x + INFO_WIDTH // 2, WINDOW_HEIGHT - 100))
            self.screen.blit(pause_text, text_rect)
        
        # FPS indicator
        fps_text = self.font_small.render(f"Speed: {self.fps} fps", True, (100, 200, 255))
        self.screen.blit(fps_text, (panel_x + 20, WINDOW_HEIGHT - 40))
    
    def handle_events(self):
        """Handle user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                elif event.key == pygame.K_r:
                    return 'reset'
                
                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 5)
                
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 5)
        
        return None
    
    def play_episode(self):
        """Play one episode with visualization"""
        current_state = self.env.reset()
        done = False
        
        while not done and self.running:
            # Handle events
            action = self.handle_events()
            if action == 'reset':
                break
            
            if not self.paused:
                # Get AI decision
                next_states = self.env.get_next_states()
                if not next_states:
                    break
                
                state_dict = {tuple(v): k for k, v in next_states.items()}
                best_state = self.agent.best_state(state_dict.keys())
                best_action = state_dict[best_state]  # (x, rotation)
                
                # Get piece shape for animation
                piece_shape = self.env._get_rotated_piece(
                    self.env.current_piece, 
                    best_action[1]
                )
                
                # Calculate final position (ghost position)
                ghost_pos = [best_action[0], 0]
                while not self.env._check_collision(piece_shape, [ghost_pos[0], ghost_pos[1] + 1]):
                    ghost_pos[1] += 1
                
                # Animate piece falling
                piece_pos = [best_action[0], 0]
                animation_steps = max(1, ghost_pos[1])
                
                for step in range(animation_steps + 1):
                    # Handle events during animation
                    self.handle_events()
                    
                    if not self.paused:
                        piece_pos[1] = step
                    
                    # Render
                    self.screen.fill(BG_COLOR)
                    self.draw_board(
                        self.env.board,
                        current_piece_shape=piece_shape,
                        piece_pos=piece_pos if step <= ghost_pos[1] else None,
                        ghost_pos=ghost_pos
                    )
                    
                    stats = {
                        'episode': self.episode,
                        'score': self.env.get_game_score(),
                        'lines': self.env.lines_cleared,
                        'pieces': self.env.pieces_placed
                    }
                    
                    self.draw_info_panel(stats)
                    pygame.display.flip()
                    
                    # Speed control for animation
                    if not self.paused:
                        self.clock.tick(self.fps * 2)  # Animation faster than game speed
                
                # Execute action (place piece)
                reward, done = self.env.play(best_action[0], best_action[1])
                
                # Show final result briefly
                self.screen.fill(BG_COLOR)
                self.draw_board(self.env.board)
                stats = {
                    'episode': self.episode,
                    'score': self.env.get_game_score(),
                    'lines': self.env.lines_cleared,
                    'pieces': self.env.pieces_placed
                }
                self.draw_info_panel(stats)
                pygame.display.flip()
                self.clock.tick(self.fps / 2)  # Pause briefly after placing
        
        return self.env.get_game_score(), self.env.lines_cleared
    
    def run(self, episodes=None):
        """
        Run the visualizer
        
        Args:
            episodes: Number of episodes to play (None = infinite)
        """
        print("\n" + "="*60)
        print("TETRIS AI - VISUAL MODE")
        print("="*60)
        print(f"Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset episode")
        print("  UP/DOWN - Adjust speed")
        print("  ESC - Quit")
        print("="*60 + "\n")
        
        self.episode = 0
        
        while self.running:
            if episodes is not None and self.episode >= episodes:
                break
            
            self.episode += 1
            print(f"Episode {self.episode} starting...")
            
            score, lines = self.play_episode()
            
            print(f"Episode {self.episode} finished - Score: {score}, Lines: {lines}")
        
        pygame.quit()
        print("\nThanks for watching! 🎮")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch AI play Tetris visually')
    parser.add_argument('--model', type=str, default='models/best_lines.keras',
                       help='Path to trained model (default: models/best_lines.keras)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of episodes to play (default: infinite)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Display speed in FPS (default: 10)')
    
    args = parser.parse_args()
    
    visualizer = TetrisVisualizer(model_path=args.model, fps=args.fps)
    visualizer.run(episodes=args.episodes)
