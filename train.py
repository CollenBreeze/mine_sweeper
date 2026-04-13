import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import Minesweeper
from Models.ddqn import DDQN
from utils import MultiBoardTeacher, MultiShapeBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


DEFAULT_BOARD_SPECS = [
    {"name": "4x4", "width": 4, "height": 4, "init_mines": 2, "min_mines": 2, "max_mines": 5, "weight": 0.45},
    {"name": "6x6", "width": 6, "height": 6, "init_mines": 4, "min_mines": 4, "max_mines": 10, "weight": 0.35},
    {"name": "8x8", "width": 8, "height": 8, "init_mines": 8, "min_mines": 8, "max_mines": 18, "weight": 0.20},
]


def compute_td_loss(current_model, target_model, buffer, optimizer, batch_size, gamma, device):
    states, actions, masks, rewards, next_states, next_masks, dones, board_shape = buffer.sample(batch_size)

    state = torch.as_tensor(states, dtype=torch.float32, device=device)
    next_state = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    action = torch.as_tensor(actions, dtype=torch.long, device=device)
    reward = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    mask = torch.as_tensor(masks, dtype=torch.float32, device=device)
    next_mask = torch.as_tensor(next_masks, dtype=torch.float32, device=device)
    done = torch.as_tensor(dones, dtype=torch.float32, device=device)

    q_values = current_model(state, mask)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_from_current = current_model(next_state, next_mask)
        next_actions = next_q_from_current.argmax(dim=1)
        next_q_from_target = target_model(next_state, next_mask)
        next_q_value = next_q_from_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.smooth_l1_loss(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(current_model.parameters(), 5.0)
    optimizer.step()

    return float(loss.item()), board_shape


def train(args):
    set_seed(args.seed)
    if not torch.cuda.is_available():
        torch.backends.mkldnn.enabled = False
        torch.set_num_threads(max(1, min(4, args.cpu_threads)))
        torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始训练！设备: {device}")

    board_specs = DEFAULT_BOARD_SPECS
    teacher = MultiBoardTeacher(board_specs=board_specs, window_size=args.teacher_window)
    model = DDQN(hidden_dim=args.hidden_dim, res_blocks=args.res_blocks).to(device)
    target_model = DDQN(hidden_dim=args.hidden_dim, res_blocks=args.res_blocks).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = MultiShapeBuffer(capacity_per_shape=args.capacity_per_shape)

    epsilon = args.epsilon_start
    total_steps = 0
    running_losses = []
    group_rewards = []
    group_wins = 0
    board_episode_counts = {spec['name']: 0 for spec in board_specs}
    board_win_counts = {spec['name']: 0 for spec in board_specs}

    models_dir = Path("Models")
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / args.checkpoint_name
    metadata_path = models_dir / args.metadata_name

    for ep in range(args.episodes):
        task = teacher.sample_task()
        width, height, mines = task["width"], task["height"], task["mines"]
        board_name = task["name"]
        env = Minesweeper(width, height, mines)

        state = env.state.astype(np.float32).copy()
        mask = env.get_mask(flatten=False)
        done = False
        total_reward = 0.0
        is_win = False

        while not done:
            total_steps += 1
            valid_actions = np.flatnonzero(mask.reshape(-1) > 0.5)
            if len(valid_actions) == 0:
                break

            if np.random.rand() < epsilon:
                action_idx = int(np.random.choice(valid_actions))
            else:
                action_idx = model.act(state, mask, epsilon=0.0, board_shape=(height, width), device=device)

            next_state_flat, reward, done, next_mask_flat = env.step(action_idx)
            next_state = next_state_flat.reshape(height, width).astype(np.float32)
            next_mask = next_mask_flat.reshape(height, width).astype(np.float32)

            buffer.push(state, action_idx, mask, reward, next_state, next_mask, done)

            state = next_state
            mask = next_mask
            total_reward += reward
            if done and reward == 1.0:
                is_win = True

            if total_steps >= args.learning_starts and buffer.can_sample(args.batch_size):
                loss, sampled_shape = compute_td_loss(
                    model, target_model, buffer, optimizer, args.batch_size, args.gamma, device
                )
                running_losses.append(loss)

            if total_steps % args.update_target_every == 0:
                target_model.load_state_dict(model.state_dict())

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        teacher.record_game(board_name, is_win)
        group_rewards.append(total_reward)
        group_wins += int(is_win)
        board_episode_counts[board_name] += 1
        board_win_counts[board_name] += int(is_win)

        if (ep + 1) % args.log_every == 0:
            avg_win_rate = group_wins / args.log_every
            avg_reward = float(np.mean(group_rewards)) if group_rewards else 0.0
            avg_loss = float(np.mean(running_losses[-200:])) if running_losses else 0.0
            print(
                f"🎯 训练组 {(ep + 1) // args.log_every:03d} | 总局数: {ep + 1:05d} | "
                f"平均胜率: {avg_win_rate:5.1%} | 平均得分: {avg_reward:6.3f} | "
                f"平均损失: {avg_loss:7.4f} | 探索率(Eps): {epsilon:.3f}"
            )
            print("   老师状态:", teacher.status_string())
            print("   经验池:", buffer.summary())

            for name in board_episode_counts:
                if board_episode_counts[name] > 0:
                    wr = board_win_counts[name] / board_episode_counts[name]
                    print(f"   - {name}: 最近 {board_episode_counts[name]} 局胜率 {wr:5.1%}")

            teacher.check_and_update_difficulty()
            group_rewards = []
            group_wins = 0
            board_episode_counts = {spec['name']: 0 for spec in board_specs}
            board_win_counts = {spec['name']: 0 for spec in board_specs}

        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.episodes:
            payload = {
                "model_state_dict": model.state_dict(),
                "target_model_state_dict": target_model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "res_blocks": args.res_blocks,
                },
                "train_config": vars(args),
                "board_specs": board_specs,
                "episodes_finished": ep + 1,
                "epsilon": epsilon,
            }
            torch.save(payload, checkpoint_path)
            metadata_path.write_text(json.dumps(payload["train_config"], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"💾 已保存 checkpoint: {checkpoint_path}")

    print(f"训练完成，模型已保存到: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多形状扫雷 DDQN 训练脚本")
    parser.add_argument("--episodes", type=int, default=300000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epsilon_start", type=float, default=0.1)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.99995)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--update_target_every", type=int, default=1000)
    parser.add_argument("--capacity_per_shape", type=int, default=40000)
    parser.add_argument("--teacher_window", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--res_blocks", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--checkpoint_name", type=str, default="minesweeper_anyshape_ddqn.pt")
    parser.add_argument("--metadata_name", type=str, default="minesweeper_anyshape_ddqn.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_threads", type=int, default=4)
    train(parser.parse_args())
