#通用的工具

import numpy as np
import random
from collections import deque


class CurriculumTeacher:
    def __init__(self, init_mines=2, min_mines=2, max_mines=9, window_size=200):
        self.current_target_mines = init_mines  # 当前的核心目标难度
        self.min_mines = min_mines
        self.max_mines = max_mines
        self.window_size = window_size  # 考核周期（多少局评测一次）
        self.results_history = deque(maxlen=window_size)

    def sample_mines(self):
        """
        核心逻辑：每次开局，不仅给当前目标难度的雷数，还混入少量更低级和更高级的知识。
        """
        choices = [self.current_target_mines - 1, self.current_target_mines, self.current_target_mines + 1]
        probs = [0.15, 0.70, 0.15]  # 15%复习简单，70%练当前，15%挑战更难

        # 处理触底和触顶的边界情况
        if self.current_target_mines == self.min_mines:
            choices = [self.min_mines, self.min_mines + 1]
            probs = [0.80, 0.20]
        elif self.current_target_mines == self.max_mines:
            choices = [self.max_mines - 1, self.max_mines]
            probs = [0.20, 0.80]

        # 随机抽样并确保不会超出上下限
        sampled = np.random.choice(choices, p=probs)
        return int(max(self.min_mines, min(self.max_mines, sampled)))

    def record_game(self, is_win):
        """记录每局的胜负"""
        self.results_history.append(1 if is_win else 0)

    def check_and_update_difficulty(self):
        """定期考核，达标则升级"""
        if len(self.results_history) >= self.window_size:
            win_rate = np.mean(self.results_history)

            # 如果当前难度胜率超过 65%，并且还能升级
            if win_rate >= 0.65 and self.current_target_mines < self.max_mines:
                self.current_target_mines += 1
                print(f"📈 考核通过！胜率 {win_rate:.1%}。核心难度提升至: {self.current_target_mines} 颗雷")
                self.results_history.clear()  # 升级后清空成绩单重新考

            # 如果被打崩了（胜率低于15%），降级复习
            elif win_rate <= 0.15 and self.current_target_mines > self.min_mines:
                self.current_target_mines -= 1
                print(f"📉 陷入瓶颈！胜率 {win_rate:.1%}。核心难度降至: {self.current_target_mines} 颗雷")
                self.results_history.clear()

            return win_rate
        return None

class Buffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen = capacity)

    def push(self,state,action,mask,reward,new_state,new_mask,terminal):
        self.buffer.append((state,action,mask,reward,new_state,new_mask,terminal))

    def sample(self,batch_size):
        states,actions,masks,rewards,new_states,new_mask,terminals = zip(*random.sample(self.buffer, batch_size))
        return states,actions,masks,rewards,new_states,new_mask,terminals
