import argparse
import time
from pathlib import Path

try:
    import pygame
except ImportError:
    pygame = None

import torch

from game import Minesweeper
from Models.ddqn import DDQN


def load_model(model_path: str, device: torch.device) -> DDQN:
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model_cfg = payload.get("model_config", {})
        model = DDQN(**model_cfg).to(device)
        model.load_state_dict(payload["model_state_dict"])
    else:
        model = DDQN().to(device)
        model.load_state_dict(payload)
    model.eval()
    return model


def enjoy(width: int, height: int, mines: int, model_path: str, step_delay: float = 0.35):
    if pygame is None:
        raise ImportError("practice.py 需要 pygame。请先执行: pip install pygame")
    from renderer import Renderer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Minesweeper(width, height, mines)
    model = load_model(model_path, device)
    renderer = Renderer(env.state)

    print(f"✅ 成功载入模型: {model_path}")
    print(f"当前棋盘: {height}x{width} | 雷数: {mines}")

    running = True
    while running:
        env.reset(new_bomb_no=mines)
        renderer.state = env.state
        done = False

        print(f"\n--- 新开一局 ({height}x{width}, {mines} 颗雷) ---")

        while not done and running:
            renderer.draw()
            time.sleep(step_delay)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            state = env.state.astype('float32')
            mask = env.get_mask(flatten=False)

            with torch.no_grad():
                action_idx = model.act(state, mask, epsilon=0.0, board_shape=(height, width), device=device)

            row = action_idx // width
            col = action_idx % width
            print(f"AI 选择点击坐标: ({row}, {col})")

            _, reward, done, _ = env.step(action_idx)
            renderer.state = env.state

            if done:
                renderer.draw()
                if reward == 1.0:
                    print("🎉 胜利！AI 成功排雷。")
                else:
                    print("💥 炸了！AI 翻车了。")
                time.sleep(1.5)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多形状扫雷模型演示脚本")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--mines", type=int, default=12)
    parser.add_argument("--model_path", type=str, default=str(Path("Models") / "minesweeper_anyshape_ddqn.pt"))
    parser.add_argument("--step_delay", type=float, default=0.35)
    args = parser.parse_args()
    enjoy(args.width, args.height, args.mines, args.model_path, args.step_delay)
