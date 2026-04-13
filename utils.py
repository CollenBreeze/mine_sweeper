import copy
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np


class CurriculumTeacher:
    def __init__(self, init_mines=2, min_mines=2, max_mines=9, window_size=200):
        self.current_target_mines = int(init_mines)
        self.min_mines = int(min_mines)
        self.max_mines = int(max_mines)
        self.window_size = int(window_size)
        self.results_history = deque(maxlen=window_size)

    def sample_mines(self):
        choices = [self.current_target_mines - 1, self.current_target_mines, self.current_target_mines + 1]
        probs = [0.15, 0.70, 0.15]

        if self.current_target_mines == self.min_mines:
            choices = [self.min_mines, min(self.max_mines, self.min_mines + 1)]
            probs = [0.80, 0.20]
        elif self.current_target_mines == self.max_mines:
            choices = [max(self.min_mines, self.max_mines - 1), self.max_mines]
            probs = [0.20, 0.80]

        sampled = np.random.choice(choices, p=probs)
        return int(max(self.min_mines, min(self.max_mines, sampled)))

    def record_game(self, is_win):
        self.results_history.append(1 if is_win else 0)

    def check_and_update_difficulty(self):
        if len(self.results_history) < self.window_size:
            return None

        win_rate = float(np.mean(self.results_history))
        if win_rate >= 0.65 and self.current_target_mines < self.max_mines:
            self.current_target_mines += 1
            print(f"📈 {win_rate:.1%} 达标，核心难度提升到 {self.current_target_mines} 颗雷")
            self.results_history.clear()
        elif win_rate <= 0.15 and self.current_target_mines > self.min_mines:
            self.current_target_mines -= 1
            print(f"📉 {win_rate:.1%} 过低，核心难度回退到 {self.current_target_mines} 颗雷")
            self.results_history.clear()
        return win_rate


class MultiBoardTeacher:
    """
    每种棋盘形状各自保留一个“老师”，互不干扰地调整雷数难度。

    board_specs 形如：
    [
        {"name": "4x4", "width": 4, "height": 4, "init_mines": 2, "min_mines": 2, "max_mines": 5, "weight": 0.45},
        ...
    ]
    """

    def __init__(self, board_specs: List[Dict], window_size: int = 150):
        self.board_specs = copy.deepcopy(board_specs)
        self.spec_by_name = {}
        self.teachers = {}
        self.latest_win_rates: Dict[str, Optional[float]] = {}

        for idx, spec in enumerate(self.board_specs):
            spec = dict(spec)
            name = spec.get("name") or f"{spec['width']}x{spec['height']}"
            spec["name"] = name
            spec.setdefault("weight", 1.0)
            spec.setdefault("init_mines", spec.get("min_mines", 1))
            spec.setdefault("min_mines", 1)
            max_default = max(2, (spec["width"] * spec["height"]) // 4)
            spec.setdefault("max_mines", max_default)
            self.board_specs[idx] = spec
            self.spec_by_name[name] = spec
            self.teachers[name] = CurriculumTeacher(
                init_mines=spec["init_mines"],
                min_mines=spec["min_mines"],
                max_mines=spec["max_mines"],
                window_size=window_size,
            )
            self.latest_win_rates[name] = None

    def sample_task(self):
        names = [spec["name"] for spec in self.board_specs]
        weights = [spec.get("weight", 1.0) for spec in self.board_specs]
        board_name = random.choices(names, weights=weights, k=1)[0]
        spec = self.spec_by_name[board_name]
        mines = self.teachers[board_name].sample_mines()
        return {
            "name": board_name,
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "mines": int(mines),
        }

    def record_game(self, board_name: str, is_win: bool):
        self.teachers[board_name].record_game(is_win)

    def check_and_update_difficulty(self):
        results = {}
        for name, teacher in self.teachers.items():
            win_rate = teacher.check_and_update_difficulty()
            if win_rate is not None:
                self.latest_win_rates[name] = win_rate
            results[name] = {
                "target_mines": teacher.current_target_mines,
                "recent_win_rate": self.latest_win_rates[name],
            }
        return results

    def status_string(self) -> str:
        parts = []
        for spec in self.board_specs:
            name = spec["name"]
            win_rate = self.latest_win_rates.get(name)
            win_rate_text = "--" if win_rate is None else f"{win_rate:5.1%}"
            parts.append(f"{name}:雷数目标={self.teachers[name].current_target_mines},胜率={win_rate_text}")
        return " | ".join(parts)


class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, mask, reward, new_state, new_mask, terminal):
        self.buffer.append((state, action, mask, reward, new_state, new_mask, terminal))

    def sample(self, batch_size):
        states, actions, masks, rewards, new_states, new_mask, terminals = zip(*random.sample(self.buffer, batch_size))
        return states, actions, masks, rewards, new_states, new_mask, terminals


class MultiShapeBuffer:
    """
    按棋盘 shape 分桶的回放池。

    好处：
    - 一个模型可以跨形状训练；
    - 采样时保证同一 batch 的状态 shape 一致，不需要 padding，也不会破坏 action_idx 的扁平索引。
    """

    def __init__(self, capacity_per_shape=50000):
        self.capacity_per_shape = int(capacity_per_shape)
        self.buffers = defaultdict(lambda: deque(maxlen=self.capacity_per_shape))

    def push(self, state, action, mask, reward, next_state, next_mask, done):
        shape = tuple(state.shape)
        self.buffers[shape].append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                np.asarray(mask, dtype=np.float32).copy(),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                np.asarray(next_mask, dtype=np.float32).copy(),
                float(done),
            )
        )

    def can_sample(self, batch_size):
        return any(len(buf) >= batch_size for buf in self.buffers.values())

    def sample(self, batch_size):
        eligible = [(shape, len(buf)) for shape, buf in self.buffers.items() if len(buf) >= batch_size]
        if not eligible:
            raise ValueError("当前没有任何 shape 的经验数量达到 batch_size，无法采样")

        weights = np.asarray([count for _, count in eligible], dtype=np.float64)
        weights /= weights.sum()
        shape_idx = np.random.choice(len(eligible), p=weights)
        shape = eligible[shape_idx][0]
        batch = random.sample(self.buffers[shape], batch_size)
        states, actions, masks, rewards, next_states, next_masks, dones = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.stack(masks),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.stack(next_masks),
            np.asarray(dones, dtype=np.float32),
            shape,
        )

    def summary(self):
        return {f"{shape[0]}x{shape[1]}": len(buf) for shape, buf in self.buffers.items()}

    def __len__(self):
        return sum(len(buf) for buf in self.buffers.values())
