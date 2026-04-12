import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class DDQN(nn.Module):
    def __init__(self, inp_dim, action_dim):
        super(DDQN, self).__init__()
        self.epsilon = 1.0  # 初始探索率

        # 1. 特征提取层：把棋盘数字转化成特征信号
        self.feature = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 2. 优势头 (Advantage)：判断每个格子“点下去”的相对好坏
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # 3. 价值头 (Value)：判断当前整个棋盘局势的底分
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask):
        x = x / 8.0  # 归一化输入
        feat = self.feature(x)

        adv = self.advantage(feat)
        val = self.value(feat)

        # 合并为 Q 值：Q = V + (A - mean(A))
        q = val + (adv - adv.mean(dim=1, keepdim=True))

        # 动作掩码：把已点的格子对应的 Q 值设为极小值
        fill_value = torch.full_like(q, -1e9)
        final_q = torch.where(mask.bool(), q, fill_value)
        return final_q

    def act(self, state, mask):
        # Epsilon-Greedy 决策逻辑
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            mask = torch.FloatTensor(mask).unsqueeze(0)
            with torch.no_grad():
                q_value = self.forward(state, mask)
            return q_value.max(1)[1].item()
        else:
            # 随机从合法格子中选一个
            indices = np.nonzero(mask)[0]
            return random.choice(indices)


class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, mask, reward, next_state, next_mask, done):
        self.buffer.append((state, action, mask, reward, next_state, next_mask, done))

    def sample(self, batch_size):
        # 随机抽样记忆
        return zip(*random.sample(self.buffer, batch_size))