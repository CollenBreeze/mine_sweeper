import torch
import time
import pygame
import numpy as np
from game import Minesweeper
from Models.ddqn import DDQN
from renderer import Renderer


def enjoy():
    # --- 1. 环境与模型配置 ---
    width, height = 4, 4
    mines = 3  # 你可以手动调整难度看看 AI 能不能 hold 住
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Minesweeper(width, height, mines)
    model = DDQN(width * height, width * height).to(device)

    # --- 2. 载入 AI 的记忆 ---
    model_path = "Models/minesweeper_ddqn.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 切换到评价模式
        print(f"✅ 成功载入模型: {model_path}")
    except:
        print("❌ 未找到模型文件，请确保文件名正确。")
        return

    # --- 3. 初始化渲染器 ---
    renderer = Renderer(env.state)

    running = True
    while running:
        env.reset(new_bomb_no=mines)
        done = False
        renderer.state = env.state  # 同步状态

        print(f"\n--- 新开一局 ({mines}颗雷) ---")

        while not done and running:
            # A. 渲染当前画面
            renderer.draw()
            time.sleep(0.5)  # 暂停0.5秒，让你看清 AI 的选择

            # B. 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # C. AI 思考 (不带随机性)
            state = env.state.flatten()
            mask = env.get_mask()

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = model(state_t, mask_t)
                # 选取 Q 值最大的动作
                action_idx = q_values.argmax().item()

            # D. 执行动作
            i, j = action_idx // width, action_idx % width
            print(f"AI 选择点击坐标: ({i}, {j})")

            next_state, reward, done, _ = env.step(action_idx)
            renderer.state = env.state

            if done:
                renderer.draw()  # 最后一帧
                if reward == 1.0:
                    print("🎉 胜利！AI 成功排雷。")
                else:
                    print("💥 炸了！AI 翻车了。")
                time.sleep(2)  # 结束后停两秒再开下一局

    pygame.quit()


if __name__ == "__main__":
    enjoy()