"""
Class TetrisGame - Logic chính của game Tetris
"""
import pygame
import sys
from .tetromino import Tetromino
from .config import *


class TetrisGame:
    """Class quản lý logic và hiển thị game Tetris"""
    
    def __init__(self):
        """Khởi tạo game"""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris AI Project')
        self.clock = pygame.time.Clock()
        
        # Font chữ
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 20)
        
        self.reset_game()

    def reset_game(self):
        """Reset game về trạng thái ban đầu"""
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.fall_time = 0
        self.fall_speed = INITIAL_FALL_SPEED

    def check_collision(self, piece, offset_x=0, offset_y=0):
        """
        Kiểm tra va chạm của khối với lưới hoặc biên
        
        Args:
            piece: Khối cần kiểm tra
            offset_x: Độ lệch theo trục x
            offset_y: Độ lệch theo trục y
            
        Returns:
            True nếu có va chạm, False nếu không
        """
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + offset_x
                    new_y = piece.y + y + offset_y
                    
                    # Kiểm tra biên trái, phải, dưới
                    if new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT:
                        return True
                    
                    # Kiểm tra va chạm với các khối đã đặt
                    if new_y >= 0 and self.grid[new_y][new_x]:
                        return True
        return False

    def lock_piece(self):
        """Khóa khối hiện tại vào lưới"""
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell and self.current_piece.y + y >= 0:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color

    def clear_lines(self):
        """Xóa các dòng đầy và tính điểm"""
        lines_to_clear = []
        
        # Tìm các dòng đầy
        for y in range(GRID_HEIGHT):
            if all(self.grid[y]):
                lines_to_clear.append(y)

        # Xóa dòng và cập nhật điểm
        if lines_to_clear:
            for line in lines_to_clear:
                del self.grid[line]
                self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
            
            self.lines_cleared += len(lines_to_clear)
            self.score += SCORES.get(len(lines_to_clear), 0) * self.level
            
            # Tăng level và tốc độ
            self.level = self.lines_cleared // 10 + 1
            self.fall_speed = max(MIN_FALL_SPEED, 
                                INITIAL_FALL_SPEED - (self.level - 1) * SPEED_DECREASE_PER_LEVEL)
        
        return len(lines_to_clear)

    def move(self, dx):
        """
        Di chuyển khối sang trái hoặc phải
        
        Args:
            dx: Hướng di chuyển (-1: trái, 1: phải)
            
        Returns:
            True nếu di chuyển thành công, False nếu không
        """
        if not self.check_collision(self.current_piece, offset_x=dx):
            self.current_piece.x += dx
            return True
        return False

    def rotate_piece(self):
        """
        Xoay khối với wall-kick
        
        Returns:
            True nếu xoay thành công, False nếu không
        """
        original_shape = self.current_piece.shape
        self.current_piece.rotate()
        
        # Kiểm tra va chạm
        if self.check_collision(self.current_piece):
            # Thử wall kick (đẩy khối ra khỏi tường)
            for offset in [0, 1, -1, 2, -2]:
                if not self.check_collision(self.current_piece, offset_x=offset):
                    self.current_piece.x += offset
                    return True
            
            # Không xoay được, khôi phục hình dạng cũ
            self.current_piece.shape = original_shape
            return False
        
        return True

    def drop(self):
        """Rơi nhanh xuống (hard drop)"""
        drop_distance = 0
        while not self.check_collision(self.current_piece, offset_y=1):
            self.current_piece.y += 1
            drop_distance += 1
        
        self.score += drop_distance * 2

    def update(self, delta_time):
        """
        Cập nhật trạng thái game
        
        Args:
            delta_time: Thời gian trôi qua từ frame trước (ms)
        """
        if self.game_over:
            return

        self.fall_time += delta_time
        
        # Rơi tự động
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            
            if not self.check_collision(self.current_piece, offset_y=1):
                self.current_piece.y += 1
            else:
                # Khóa khối và tạo khối mới
                self.lock_piece()
                self.clear_lines()
                self.current_piece = self.next_piece
                self.next_piece = Tetromino()
                
                # Kiểm tra game over
                if self.check_collision(self.current_piece):
                    self.game_over = True

    def draw_grid(self):
        """Vẽ lưới game"""
        # Vẽ nền lưới
        pygame.draw.rect(self.screen, DARK_GRAY, 
                        (GRID_OFFSET_X, GRID_OFFSET_Y, 
                         GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))

        # Vẽ các khối đã đặt
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x]:
                    self._draw_block(x, y, self.grid[y][x], GRID_OFFSET_X, GRID_OFFSET_Y)

        # Vẽ đường lưới
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(self.screen, GRAY,
                           (GRID_OFFSET_X + x * BLOCK_SIZE, GRID_OFFSET_Y),
                           (GRID_OFFSET_X + x * BLOCK_SIZE, GRID_OFFSET_Y + GRID_HEIGHT * BLOCK_SIZE))
        
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, GRAY,
                           (GRID_OFFSET_X, GRID_OFFSET_Y + y * BLOCK_SIZE),
                           (GRID_OFFSET_X + GRID_WIDTH * BLOCK_SIZE, GRID_OFFSET_Y + y * BLOCK_SIZE))

    def _draw_block(self, x, y, color, offset_x, offset_y):
        """
        Vẽ một block với hiệu ứng 3D
        
        Args:
            x, y: Vị trí trên lưới
            color: Màu của block
            offset_x, offset_y: Offset để vẽ
        """
        px = offset_x + x * BLOCK_SIZE
        py = offset_y + y * BLOCK_SIZE
        
        # Vẽ block chính
        pygame.draw.rect(self.screen, color,
                       (px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2))
        
        # Vẽ viền tạo hiệu ứng 3D
        pygame.draw.rect(self.screen, WHITE,
                       (px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), 1)

    def draw_piece(self, piece, offset_x=GRID_OFFSET_X, offset_y=GRID_OFFSET_Y):
        """
        Vẽ khối Tetromino
        
        Args:
            piece: Khối cần vẽ
            offset_x, offset_y: Offset để vẽ
        """
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = piece.y + y
                    if grid_y >= 0:  # Chỉ vẽ trong phạm vi nhìn thấy
                        self._draw_block(piece.x + x, grid_y, piece.color, offset_x, offset_y)

    def draw_next_piece(self):
        """Vẽ khối tiếp theo"""
        next_x = 400
        next_y = 100
        
        # Vẽ khung
        pygame.draw.rect(self.screen, DARK_GRAY, (next_x - 10, next_y - 40, 150, 120), border_radius=5)
        text = self.small_font.render('NEXT', True, WHITE)
        self.screen.blit(text, (next_x + 30, next_y - 35))
        
        # Vẽ khối
        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    px = next_x + x * BLOCK_SIZE
                    py = next_y + y * BLOCK_SIZE
                    pygame.draw.rect(self.screen, self.next_piece.color,
                                   (px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2))
                    pygame.draw.rect(self.screen, WHITE,
                                   (px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), 1)

    def draw_stats(self):
        """Vẽ thống kê game"""
        stats_x = 400
        stats_y = 250
        
        stats = [
            ('Score', self.score),
            ('Lines', self.lines_cleared),
            ('Level', self.level),
        ]
        
        for i, (label, value) in enumerate(stats):
            text = self.small_font.render(f'{label}: {value}', True, WHITE)
            self.screen.blit(text, (stats_x, stats_y + i * 35))

    def draw_controls(self):
        """Vẽ hướng dẫn điều khiển"""
        controls_x = 400
        controls_y = 400
        
        # Tiêu đề
        title = self.small_font.render('Controls:', True, WHITE)
        self.screen.blit(title, (controls_x, controls_y))
        
        # Danh sách điều khiển
        controls = [
            '← → : Move',
            '↑ : Rotate',
            '↓ : Soft Drop',
            'Space : Hard Drop',
            'R : Restart',
        ]
        
        for i, control in enumerate(controls):
            text = self.tiny_font.render(control, True, LIGHT_GRAY)
            self.screen.blit(text, (controls_x, controls_y + 30 + i * 25))

    def draw_game_over(self):
        """Vẽ màn hình Game Over"""
        # Overlay tối
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Text
        game_over_text = self.font.render('GAME OVER', True, WHITE)
        score_text = self.small_font.render(f'Final Score: {self.score}', True, WHITE)
        restart_text = self.small_font.render('Press R to Restart', True, WHITE)
        
        self.screen.blit(game_over_text, 
                       (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 250))
        self.screen.blit(score_text,
                       (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 300))
        self.screen.blit(restart_text,
                       (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 350))

    def draw(self):
        """Vẽ toàn bộ màn hình"""
        self.screen.fill(BLACK)
        
        self.draw_grid()
        self.draw_piece(self.current_piece)
        self.draw_next_piece()
        self.draw_stats()
        self.draw_controls()
        
        if self.game_over:
            self.draw_game_over()
        
        pygame.display.flip()

    def handle_input(self):
        """
        Xử lý input từ người chơi
        
        Returns:
            False nếu người chơi thoát game, True nếu tiếp tục
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                # Restart game
                if event.key == pygame.K_r:
                    self.reset_game()
                
                # Điều khiển game (chỉ khi chưa game over)
                if not self.game_over:
                    if event.key == pygame.K_LEFT:
                        self.move(-1)
                    elif event.key == pygame.K_RIGHT:
                        self.move(1)
                    elif event.key == pygame.K_DOWN:
                        if not self.check_collision(self.current_piece, offset_y=1):
                            self.current_piece.y += 1
                            self.score += 1
                    elif event.key == pygame.K_UP:
                        self.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        self.drop()
        
        return True

    def run(self):
        """Vòng lặp chính của game"""
        running = True
        while running:
            delta_time = self.clock.tick(FPS)
            
            running = self.handle_input()
            self.update(delta_time)
            self.draw()
        
        pygame.quit()
        sys.exit()
