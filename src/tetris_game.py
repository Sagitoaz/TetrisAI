import pygame
import random
import sys

# Khởi tạo Pygame
pygame.init()

# Màu sắc
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

# Định nghĩa các hình dạng Tetromino
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]],  # L
]

# Cài đặt game
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH + 300
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT + 100

# Điểm số cho mỗi dòng xóa
SCORES = {
    1: 100,
    2: 300,
    3: 500,
    4: 800
}


class Tetromino:
    def __init__(self, shape_idx=None):
        if shape_idx is None:
            shape_idx = random.randint(0, len(SHAPES) - 1)
        self.shape = SHAPES[shape_idx]
        self.color = COLORS[shape_idx]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.rotation = 0

    def rotate(self):
        """Xoay khối"""
        self.shape = list(zip(*self.shape[::-1]))
        return self

    def get_rotated_shape(self):
        """Lấy hình dạng sau khi xoay mà không thay đổi khối hiện tại"""
        return list(zip(*self.shape[::-1]))


class TetrisGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Tetris AI Project')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
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
        self.fall_speed = 500  # milliseconds

    def check_collision(self, piece, offset_x=0, offset_y=0):
        """Kiểm tra va chạm"""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = piece.x + x + offset_x
                    new_y = piece.y + y + offset_y
                    
                    # Kiểm tra biên
                    if new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT:
                        return True
                    
                    # Kiểm tra va chạm với các khối đã đặt
                    if new_y >= 0 and self.grid[new_y][new_x]:
                        return True
        return False

    def lock_piece(self):
        """Khóa khối vào lưới"""
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell and self.current_piece.y + y >= 0:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = self.current_piece.color

    def clear_lines(self):
        """Xóa các dòng đầy và tính điểm"""
        lines_to_clear = []
        for y in range(GRID_HEIGHT):
            if all(self.grid[y]):
                lines_to_clear.append(y)

        if lines_to_clear:
            for line in lines_to_clear:
                del self.grid[line]
                self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
            
            self.lines_cleared += len(lines_to_clear)
            self.score += SCORES.get(len(lines_to_clear), 0) * self.level
            self.level = self.lines_cleared // 10 + 1
            self.fall_speed = max(100, 500 - (self.level - 1) * 50)

    def move(self, dx):
        """Di chuyển khối sang trái hoặc phải"""
        if not self.check_collision(self.current_piece, offset_x=dx):
            self.current_piece.x += dx
            return True
        return False

    def rotate_piece(self):
        """Xoay khối"""
        original_shape = self.current_piece.shape
        self.current_piece.rotate()
        
        if self.check_collision(self.current_piece):
            # Thử wall kick
            for offset in [0, 1, -1, 2, -2]:
                if not self.check_collision(self.current_piece, offset_x=offset):
                    self.current_piece.x += offset
                    return True
            # Không xoay được, khôi phục hình dạng cũ
            self.current_piece.shape = original_shape
            return False
        return True

    def drop(self):
        """Rơi nhanh xuống"""
        while not self.check_collision(self.current_piece, offset_y=1):
            self.current_piece.y += 1
            self.score += 2

    def update(self, delta_time):
        """Cập nhật trạng thái game"""
        if self.game_over:
            return

        self.fall_time += delta_time
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            
            if not self.check_collision(self.current_piece, offset_y=1):
                self.current_piece.y += 1
            else:
                self.lock_piece()
                self.clear_lines()
                self.current_piece = self.next_piece
                self.next_piece = Tetromino()
                
                if self.check_collision(self.current_piece):
                    self.game_over = True

    def draw_grid(self):
        """Vẽ lưới"""
        grid_offset_x = 50
        grid_offset_y = 50

        # Vẽ nền lưới
        pygame.draw.rect(self.screen, DARK_GRAY, 
                        (grid_offset_x, grid_offset_y, 
                         GRID_WIDTH * BLOCK_SIZE, GRID_HEIGHT * BLOCK_SIZE))

        # Vẽ các khối đã đặt
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x]:
                    pygame.draw.rect(self.screen, self.grid[y][x],
                                   (grid_offset_x + x * BLOCK_SIZE + 1,
                                    grid_offset_y + y * BLOCK_SIZE + 1,
                                    BLOCK_SIZE - 2, BLOCK_SIZE - 2))
                    # Thêm hiệu ứng 3D
                    pygame.draw.rect(self.screen, WHITE,
                                   (grid_offset_x + x * BLOCK_SIZE + 1,
                                    grid_offset_y + y * BLOCK_SIZE + 1,
                                    BLOCK_SIZE - 2, BLOCK_SIZE - 2), 1)

        # Vẽ đường lưới
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(self.screen, GRAY,
                           (grid_offset_x + x * BLOCK_SIZE, grid_offset_y),
                           (grid_offset_x + x * BLOCK_SIZE, grid_offset_y + GRID_HEIGHT * BLOCK_SIZE))
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, GRAY,
                           (grid_offset_x, grid_offset_y + y * BLOCK_SIZE),
                           (grid_offset_x + GRID_WIDTH * BLOCK_SIZE, grid_offset_y + y * BLOCK_SIZE))

    def draw_piece(self, piece, offset_x=50, offset_y=50):
        """Vẽ khối Tetromino"""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    px = offset_x + (piece.x + x) * BLOCK_SIZE
                    py = offset_y + (piece.y + y) * BLOCK_SIZE
                    
                    if py >= offset_y:  # Chỉ vẽ trong phạm vi nhìn thấy
                        pygame.draw.rect(self.screen, piece.color,
                                       (px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2))
                        # Hiệu ứng 3D
                        pygame.draw.rect(self.screen, WHITE,
                                       (px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2), 1)

    def draw_next_piece(self):
        """Vẽ khối tiếp theo"""
        next_x = 400
        next_y = 100
        
        # Vẽ khung
        pygame.draw.rect(self.screen, DARK_GRAY, (next_x - 10, next_y - 40, 150, 120))
        text = self.small_font.render('NEXT', True, WHITE)
        self.screen.blit(text, (next_x + 30, next_y - 35))
        
        # Vẽ khối
        for y, row in enumerate(self.next_piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.next_piece.color,
                                   (next_x + x * BLOCK_SIZE + 1,
                                    next_y + y * BLOCK_SIZE + 1,
                                    BLOCK_SIZE - 2, BLOCK_SIZE - 2))
                    pygame.draw.rect(self.screen, WHITE,
                                   (next_x + x * BLOCK_SIZE + 1,
                                    next_y + y * BLOCK_SIZE + 1,
                                    BLOCK_SIZE - 2, BLOCK_SIZE - 2), 1)

    def draw_stats(self):
        """Vẽ thống kê game"""
        stats_x = 400
        stats_y = 250
        
        stats = [
            f'Score: {self.score}',
            f'Lines: {self.lines_cleared}',
            f'Level: {self.level}',
        ]
        
        for i, stat in enumerate(stats):
            text = self.small_font.render(stat, True, WHITE)
            self.screen.blit(text, (stats_x, stats_y + i * 35))

    def draw_controls(self):
        """Vẽ hướng dẫn điều khiển"""
        controls_x = 400
        controls_y = 400
        
        controls = [
            'Controls:',
            '← → : Move',
            '↑ : Rotate',
            '↓ : Soft Drop',
            'Space : Hard Drop',
            'R : Restart',
        ]
        
        for i, control in enumerate(controls):
            color = WHITE if i == 0 else LIGHT_GRAY
            font = self.small_font if i == 0 else pygame.font.Font(None, 20)
            text = font.render(control, True, color)
            self.screen.blit(text, (controls_x, controls_y + i * 25))

    def draw(self):
        """Vẽ toàn bộ màn hình"""
        self.screen.fill(BLACK)
        
        self.draw_grid()
        self.draw_piece(self.current_piece)
        self.draw_next_piece()
        self.draw_stats()
        self.draw_controls()
        
        if self.game_over:
            # Vẽ màn hình Game Over
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font.render('GAME OVER', True, WHITE)
            score_text = self.small_font.render(f'Final Score: {self.score}', True, WHITE)
            restart_text = self.small_font.render('Press R to Restart', True, WHITE)
            
            self.screen.blit(game_over_text, 
                           (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 250))
            self.screen.blit(score_text,
                           (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 300))
            self.screen.blit(restart_text,
                           (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 350))
        
        pygame.display.flip()

    def handle_input(self):
        """Xử lý input từ người chơi"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                
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
            delta_time = self.clock.tick(60)
            
            running = self.handle_input()
            self.update(delta_time)
            self.draw()
        
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    game = TetrisGame()
    game.run()
