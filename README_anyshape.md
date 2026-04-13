# 扫雷 DDQN 任意形状版（v2）

这版是在上一版基础上做的定向修正：
- **保留你原来的铺空白加速思路**：`game.py` 继续用 `@njit + 栈式泛洪`；
- **补上手动点击的显示缩放自适应**：`play.py` 和 `renderer.py` 会自动检测窗口实际大小与逻辑渲染大小的比例，适配高 DPI / 系统缩放。

## 这版改了什么

### 1. 模型不再绑定格子总数
- `Models/ddqn.py`：全卷积特征提取 + Dueling 头
- 输出仍然是 `H*W` 个动作 Q 值
- 同一份权重可在 `4x4 / 6x6 / 8x8 / 5x9 / 10x14 ...` 上直接跑

### 2. 训练支持多种棋盘混合
- `MultiBoardTeacher`：每种棋盘形状单独维护老师和难度
- `MultiShapeBuffer`：按 shape 分桶采样，保证 batch 内 shape 一致，不需要 padding

### 3. `game.py` 继续保留加速泛洪
- 统一内部棋盘坐标约定为 `(height, width)`
- 扁平动作索引统一使用 `idx = row * width + col`
- `unfog_zeros()` 继续使用 `@njit(fastmath=True)`
- 这版没有再把铺空白改成纯 Python

### 4. `play.py` 点击支持自动检测显示缩放
- 原版点击坐标换算是硬编码常数
- 现在改成根据 `Renderer` 的逻辑窗口大小、实际窗口大小自动换算
- 优先使用 `pygame.SCALED`，让高 DPI 下的鼠标事件跟逻辑坐标自动对齐
- 若后端返回的是物理像素坐标，会按检测到的缩放比例自动折算回棋盘坐标

## 怎么用

### 训练
```bash
python train.py
```

### 自动演示
```bash
python practice.py --width 10 --height 6 --mines 10
```

### 手动点击
```bash
python play.py
```

程序启动时会打印一行缩放检测信息，例如：
```text
显示缩放检测: logical=(600, 600), actual=(900, 900), scale=(1.500, 1.500), scaled_mode=True
```

## 附带成果
- `Models/minesweeper_anyshape_warmstart.pt`
- `Models/minesweeper_anyshape_warmstart.json`

它们主要用来证明：模型、训练、保存、加载、演示、手动点击这条链路都已经打通。
