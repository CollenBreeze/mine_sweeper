import torch
import torch.nn as nn
from game import Minesweeper
from Models.ddqn import DDQN
from utils import CurriculumTeacher, Buffer
import numpy as np
import torch.nn.functional as F


# 删掉 train_step_with_gain，改回标准版，因为不再需要用 loss 算难度了
def compute_td_loss(current_model, target_model, buffer, optimizer, batch_size, gamma, device):
    states, actions, masks, rewards, next_states, next_masks, dones = buffer.sample(batch_size)

    state = torch.FloatTensor(np.array(states)).to(device)
    next_state = torch.FloatTensor(np.array(next_states)).to(device)
    action = torch.LongTensor(actions).to(device)
    reward = torch.FloatTensor(rewards).to(device)
    mask = torch.FloatTensor(np.array(masks)).to(device)
    next_mask = torch.FloatTensor(np.array(next_masks)).to(device)
    done = torch.FloatTensor(dones).to(device)

    q_values = current_model(state, mask)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_from_current = current_model(next_state, next_mask)
        next_actions = next_q_from_current.argmax(dim=1)
        next_q_from_target = target_model(next_state, next_mask)
        next_q_value = next_q_from_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.mse_loss(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(current_model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始训练！设备: {device}")

    # ================= 参数调优区 =================
    epsilon = 1.0
    epsilon_min = 0.05  # 最低保留 5% 的探索率
    epsilon_decay = 0.9997  # 让它在前期多随机一会儿 (衰减慢一点)
    gamma = 0.95  # 4x4 网格较小，0.95 的视距足够了，能收敛更快
    batch_size = 128  # 提高单次喂给 GPU 的数据量，稳定梯度
    episodes = 50000  # 总训练局数
    update_target_every = 1000  # 目标网络同步频率

    INP_DIM = 16  # 4x4
    ACTION_DIM = 16

    env = Minesweeper(4, 4, 2)
    # 初始化老师：评估周期 200 局，目标 2 到 9 颗雷
    teacher = CurriculumTeacher(init_mines=2, min_mines=2, max_mines=9, window_size=200)

    model = DDQN(INP_DIM, ACTION_DIM).to(device)
    target_model = DDQN(INP_DIM, ACTION_DIM).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = Buffer(capacity=50000)

    # ================= 统计变量区 =================
    group_episodes = 200  # 每 200 局输出一次统计报告
    group_rewards = []
    group_wins = 0
    total_steps = 0

    for ep in range(episodes):
        # 1. 老师给出这局的雷数 (混合抽样)
        current_mines = teacher.sample_mines()
        env.reset(new_bomb_no=current_mines)

        state = env.state.flatten()
        mask = env.get_mask()
        done = False
        total_reward = 0
        is_win = False  # 记录是否胜利

        while not done:
            total_steps += 1

            # A. 决策
            if np.random.rand() < epsilon:
                valid_actions = np.where(mask == 1)[0]
                action_idx = np.random.choice(valid_actions)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_t, mask_t)
                    action_idx = q_values.argmax().item()

            # B. 执行
            next_state_np, reward, done, next_mask_np = env.step(action_idx)

            # C. 记忆
            buffer.push(state, action_idx, mask, reward, next_state_np, next_mask_np, done)

            state = next_state_np
            mask = next_mask_np
            total_reward += reward

            # 判定胜负 (reward == 1.0 即为通关大奖)
            if done and reward == 1.0:
                is_win = True

            # D. 学习
            if len(buffer.buffer) > batch_size:
                loss = compute_td_loss(model, target_model, buffer, optimizer, batch_size, gamma, device)

            # E. 定期同步目标网络
            if total_steps % update_target_every == 0:
                target_model.load_state_dict(model.state_dict())

        # 局后处理
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 记录给老师考核
        teacher.record_game(is_win)

        # 记录组统计信息
        group_rewards.append(total_reward)
        if is_win:
            group_wins += 1

        # ================= 直观的日志输出 =================
        if (ep + 1) % group_episodes == 0:
            avg_win_rate = group_wins / group_episodes
            avg_reward = np.mean(group_rewards)
            print(f"🎯 训练组 {(ep + 1) // group_episodes:03d} | 总局数: {ep + 1:05d} | "
                  f"核心雷数: {teacher.current_target_mines} | "
                  f"平均胜率: {avg_win_rate:5.1%} | 平均得分: {avg_reward:5.2f} | "
                  f"探索率(Eps): {epsilon:.3f}")

            # 老师进行考核，决定是否升/降级
            teacher.check_and_update_difficulty()

            # 重置组统计
            group_rewards = []
            group_wins = 0

    # 训练结束保存模型
    torch.save(model.state_dict(), "Models/minesweeper_ddqn.pth")
    print("训练完成，模型已保存！")