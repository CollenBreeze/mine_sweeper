import math
import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DDQN(nn.Module):
    """
    适配任意棋盘宽高的全卷积 Dueling DDQN。

    设计要点：
    - 不再依赖 flatten + 全连接层，所以模型参数不随地图大小变化。
    - 输入可为 (B, H, W) / (B, 1, H, W) / (B, H*W)；若传入 flatten，则必须额外提供 board_shape，
      或者让模型自己从平方数长度里推断正方形棋盘。
    - 输出始终为 (B, H*W) 的 Q 值，方便继续复用现有 action_idx 逻辑。
    """

    def __init__(self, inp_dim=None, action_dim=None, board_shape: Optional[Sequence[int]] = None,
                 hidden_dim: int = 64, res_blocks: int = 4):
        super().__init__()
        self.epsilon = 1.0
        self.default_board_shape = tuple(board_shape) if board_shape is not None else None
        self.hidden_dim = int(hidden_dim)
        self.res_blocks = int(res_blocks)

        # 输入通道：
        # 1) scalar 值通道
        # 2) hidden 通道
        # 3) zero 通道
        # 4-11) 数字 1~8 的 one-hot 通道
        # 12) 当前可点 mask 通道
        in_channels = 12

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(res_blocks)])
        self.advantage = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.value = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _resolve_board_shape(self, flat_dim: int, board_shape: Optional[Sequence[int]]) -> Tuple[int, int]:
        if board_shape is None:
            board_shape = self.default_board_shape

        if board_shape is not None:
            if len(board_shape) != 2:
                raise ValueError("board_shape 必须是 (height, width)")
            h, w = int(board_shape[0]), int(board_shape[1])
            if h * w != flat_dim:
                raise ValueError(f"board_shape={board_shape} 与输入长度 {flat_dim} 不匹配")
            return h, w

        side = int(math.isqrt(flat_dim))
        if side * side != flat_dim:
            raise ValueError(
                "收到 flatten 输入，但无法自动推断棋盘尺寸。"
                "请在构造模型时传 board_shape=(height, width)，"
                "或者在 forward/act 时显式传入 board_shape。"
            )
        return side, side

    def _reshape_inputs(self, x: torch.Tensor, mask: torch.Tensor,
                        board_shape: Optional[Sequence[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4 and x.size(1) == 1:
            x = x[:, 0]
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask[:, 0]

        if x.dim() == 2:
            h, w = self._resolve_board_shape(x.size(1), board_shape)
            x = x.view(x.size(0), h, w)
            mask = mask.view(mask.size(0), h, w)
        elif x.dim() == 3:
            if mask.dim() != 3:
                raise ValueError("当 x 为 (B, H, W) 时，mask 也必须是 (B, H, W)")
        else:
            raise ValueError("x 只支持 (B, H*W)、(B, H, W) 或 (B, 1, H, W)")

        return x.float(), mask.float()

    def _encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scalar = x.clamp(min=-1.0, max=8.0) / 8.0
        hidden = (x < -0.5).float()
        zero = (x == 0).float()
        number_channels = [(x == float(n)).float() for n in range(1, 9)]
        channels = [scalar.unsqueeze(1), hidden.unsqueeze(1), zero.unsqueeze(1)]
        channels.extend(ch.unsqueeze(1) for ch in number_channels)
        channels.append(mask.unsqueeze(1))
        return torch.cat(channels, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                board_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        x, mask = self._reshape_inputs(x, mask, board_shape)
        feat = self._encode(x, mask)
        feat = self.stem(feat)
        feat = self.backbone(feat)

        adv = self.advantage(feat).squeeze(1)
        val = self.value(feat).view(-1, 1, 1)
        q = val + (adv - adv.mean(dim=(1, 2), keepdim=True))

        q_flat = q.flatten(1)
        mask_flat = mask.flatten(1)
        fill_value = torch.full_like(q_flat, -1e9)
        return torch.where(mask_flat > 0.5, q_flat, fill_value)

    def act(self, state, mask, epsilon: Optional[float] = None,
            board_shape: Optional[Sequence[int]] = None, device: Optional[torch.device] = None) -> int:
        eps = self.epsilon if epsilon is None else float(epsilon)

        if isinstance(mask, np.ndarray):
            valid_actions = np.flatnonzero(mask.reshape(-1) > 0.5)
        else:
            valid_actions = torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=False).view(-1).cpu().numpy()

        if len(valid_actions) == 0:
            return 0

        if random.random() < eps:
            return int(random.choice(valid_actions.tolist()))

        model_device = device if device is not None else next(self.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32, device=model_device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=model_device).unsqueeze(0)
        with torch.no_grad():
            q_value = self.forward(state_t, mask_t, board_shape=board_shape)
        return int(q_value.argmax(dim=1).item())

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location=None):
        payload = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model_cfg = payload.get("model_config", {})
            model = cls(**model_cfg)
            model.load_state_dict(payload["model_state_dict"])
            return model, payload

        model = cls()
        model.load_state_dict(payload)
        return model, {"model_config": {}}
