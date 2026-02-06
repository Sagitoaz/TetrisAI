"""
Tetris AI Project - Main Entry Point
File chính để chạy game Tetris
"""
from src.game import TetrisGame


def main():
    """Hàm main - khởi động game"""
    print("=" * 50)
    print("TETRIS AI PROJECT")
    print("=" * 50)
    print("Starting game...")
    
    game = TetrisGame()
    game.run()


if __name__ == '__main__':
    main()
