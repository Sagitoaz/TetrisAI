"""
Model Merging Script
Gộp nhiều model .keras thành một model mạnh hơn bằng weight averaging.

Cách dùng:
    # Tự động gộp tất cả *_best_score.keras trong thư mục models/
    python scripts/merge.py

    # Chỉ định file cụ thể (khuyến nghị khi nhiều người train)
    python scripts/merge.py --models models/alice_best_score.keras models/bob_best_score.keras

    # Chỉ định output
    python scripts/merge.py --output models/merged.keras

    # Gộp có trọng số (model tốt hơn được cân nặng hơn)
    python scripts/merge.py --weighted

    # Chỉ so sánh, không gộp
    python scripts/merge.py --compare-only

Workflow nhiều người train:
    # Mỗi người train trên máy riêng với --name khác nhau:
    #   Máy A:  python scripts/train.py --name alice
    #   Máy B:  python scripts/train.py --name bob
    #
    # Copy các file *_best_score.keras từ tất cả máy vào cùng thư mục models/
    # rồi chạy merge:
    #   python scripts/merge.py --models models/alice_best_score.keras models/bob_best_score.keras
    #
    # Sau merge, tiếp tục train với model mới:
    #   python scripts/train.py --name alice --model models/merged.keras --epsilon 0.05
"""
import sys
import os
import argparse
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent import DQNAgent
from ai.environment import Tetris
from keras.models import load_model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(model_path, n_episodes=50):
    """
    Chạy model qua n_episodes ván Tetris (epsilon=0, không explore).
    Trả về dict với avg_score, avg_lines, max_score, max_lines.
    """
    env = Tetris()
    agent = DQNAgent(
        state_size=env.get_state_size(),
        epsilon_stop_episode=0,   # không decay
    )
    agent.load_model(model_path)
    agent.epsilon = 0.0           # pure exploitation

    scores = []
    lines  = []

    for _ in range(n_episodes):
        state = env.reset()
        done  = False

        while not done:
            next_states = env.get_next_states()
            state_dict  = {tuple(v): k for k, v in next_states.items()}
            best_state  = agent.best_state(state_dict.keys())
            best_action = state_dict[best_state]
            _, done     = env.play(best_action[0], best_action[1])

        scores.append(env.get_game_score())
        lines.append(env.lines_cleared)

    return {
        'avg_score':  float(np.mean(scores)),
        'avg_lines':  float(np.mean(lines)),
        'max_score':  int(max(scores)),
        'max_lines':  int(max(lines)),
    }


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def average_weights(model_paths, weights=None):
    """
    Trung bình hóa weights của nhiều model.

    Args:
        model_paths: Danh sách đường dẫn .keras
        weights:     None  → trung bình đều
                     list  → trọng số tương ứng từng model (sẽ được normalize)

    Returns:
        Model Keras đã gộp weights.
    """
    if len(model_paths) < 2:
        raise ValueError("Cần ít nhất 2 model để gộp.")

    models = [load_model(p) for p in model_paths]

    # Normalize weights
    if weights is None:
        w = [1.0 / len(models)] * len(models)
    else:
        total = sum(weights)
        w = [x / total for x in weights]

    # Tính average weights layer by layer
    base_model = models[0]
    averaged_weights = []

    for layer_idx in range(len(base_model.get_weights())):
        layer_avg = sum(
            m.get_weights()[layer_idx] * w[i]
            for i, m in enumerate(models)
        )
        averaged_weights.append(layer_avg)

    base_model.set_weights(averaged_weights)
    return base_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gộp nhiều model Tetris DQN thành một model mạnh hơn"
    )
    parser.add_argument(
        '--models', nargs='+', default=None,
        help="Đường dẫn các file .keras cần gộp. "
             "Mặc định: tất cả .keras trong models/ (trừ merged.keras)"
    )
    parser.add_argument(
        '--output', default='models/merged.keras',
        help="Đường dẫn lưu model sau khi gộp (mặc định: models/merged.keras)"
    )
    parser.add_argument(
        '--weighted', action='store_true',
        help="Gộp có trọng số dựa trên avg_score (model điểm cao → cân nặng hơn)"
    )
    parser.add_argument(
        '--compare-only', action='store_true',
        help="Chỉ so sánh các model, không gộp"
    )
    parser.add_argument(
        '--eval-episodes', type=int, default=30,
        help="Số ván chạy khi đánh giá mỗi model (mặc định: 30)"
    )
    args = parser.parse_args()

    # ---- Tìm file model ----
    if args.models:
        model_paths = args.models
    else:
        all_keras = sorted(glob.glob('models/*.keras'))
        output_abs = os.path.abspath(args.output)

        # Ưu tiên *_best_score.keras (1 file mỗi trainer, tránh double-count)
        best_score_files = [
            p for p in all_keras
            if p.endswith('_best_score.keras') and os.path.abspath(p) != output_abs
        ]

        if best_score_files:
            model_paths = best_score_files
            print("(Auto-discovery: chỉ dùng *_best_score.keras. Dùng --models để chọn file khác.)")
        else:
            # Fallback: tất cả .keras trừ output và *.json-paired checkpoints
            model_paths = [
                p for p in all_keras
                if os.path.abspath(p) != output_abs
                and not p.endswith('_best_lines.keras')  # tránh 2 file cùng trainer
            ]

    if not model_paths:
        print("Không tìm thấy model nào. Dùng --models để chỉ định.")
        sys.exit(1)

    print("="*60)
    print("MODEL MERGE TOOL")
    print("="*60)
    print(f"Tìm thấy {len(model_paths)} model:")
    for p in model_paths:
        print(f"  {p}")
    print()

    # ---- Đánh giá từng model ----
    print(f"Đánh giá từng model ({args.eval_episodes} ván mỗi model)...")
    print("-"*60)

    results = {}
    for path in model_paths:
        print(f"  Đang đánh giá: {path} ...", end="", flush=True)
        stats = evaluate_model(path, n_episodes=args.eval_episodes)
        results[path] = stats
        print(f"  avg_score={stats['avg_score']:.1f}  avg_lines={stats['avg_lines']:.1f}  "
              f"max_score={stats['max_score']}  max_lines={stats['max_lines']}")

    print("-"*60)
    best_path = max(results, key=lambda p: results[p]['avg_score'])
    print(f"\nModel tốt nhất đơn lẻ: {best_path}")
    print(f"  avg_score={results[best_path]['avg_score']:.1f}  "
          f"avg_lines={results[best_path]['avg_lines']:.1f}")

    if args.compare_only or len(model_paths) < 2:
        return

    # ---- Gộp weights ----
    print()
    if args.weighted:
        # Trọng số = avg_score của từng model
        weights = [results[p]['avg_score'] for p in model_paths]
        # Clamp để tránh model kém kéo ngược
        min_w = min(weights)
        weights = [max(0.0, w - min_w) for w in weights]
        # Nếu mọi model bằng nhau → trung bình đều
        if sum(weights) == 0:
            weights = None
        print(f"Gộp có trọng số: {[f'{w:.1f}' for w in (weights or [])]}")
    else:
        weights = None
        print("Gộp trung bình đều (unweighted)...")

    merged_model = average_weights(model_paths, weights=weights)

    # ---- Đánh giá model sau khi gộp ----
    print("\nĐánh giá model sau khi gộp...")

    # Lưu tạm để đánh giá
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    merged_model.save(args.output)

    merged_stats = evaluate_model(args.output, n_episodes=args.eval_episodes)
    print(f"  avg_score={merged_stats['avg_score']:.1f}  "
          f"avg_lines={merged_stats['avg_lines']:.1f}  "
          f"max_score={merged_stats['max_score']}  "
          f"max_lines={merged_stats['max_lines']}")

    # ---- Kết quả ----
    print()
    print("="*60)
    print("KẾT QUẢ SO SÁNH")
    print("="*60)
    print(f"{'Model':<40}  {'Avg Score':>10}  {'Avg Lines':>10}")
    print("-"*60)
    for path, stats in sorted(results.items(), key=lambda x: -x[1]['avg_score']):
        marker = " ← best single" if path == best_path else ""
        print(f"{os.path.basename(path):<40}  {stats['avg_score']:>10.1f}  "
              f"{stats['avg_lines']:>10.1f}{marker}")
    print(f"{'[merged]':<40}  {merged_stats['avg_score']:>10.1f}  "
          f"{merged_stats['avg_lines']:>10.1f}")
    print("="*60)

    improvement = merged_stats['avg_score'] - results[best_path]['avg_score']
    if improvement > 0:
        print(f"\n✓ Merged model tốt hơn best single model "
              f"+{improvement:.1f} điểm ({improvement/results[best_path]['avg_score']*100:.1f}%)")
    else:
        print(f"\n⚠  Merged model không tốt hơn best single model "
              f"({improvement:.1f} điểm).")
        print(f"   Khuyến nghị: dùng '{best_path}' thay vì merged model.")

    print(f"\n✓ Saved merged model → {args.output}")


if __name__ == "__main__":
    main()
