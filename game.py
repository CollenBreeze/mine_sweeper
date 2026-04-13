import time
import numpy as np
from numpy import zeros, random, multiply, count_nonzero, add

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class Minesweeper:
    """
    支持任意宽高的扫雷环境。

    约定：
    - grid/state 的 shape 始终是 (grid_height, grid_width)，即 (行, 列)
    - flatten 后的动作索引采用标准 row-major：idx = row * grid_width + col
    - 铺空白仍然沿用原来的 njit + 栈式泛洪算法，不改逻辑，只修坐标约定
    """

    def __init__(self, grid_width, grid_height, bomb_no):
        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.bomb_no = int(bomb_no)
        self.box_count = self.grid_width * self.grid_height
        self.uncovered_count = 0
        self.reset()

    def get_mask(self, flatten=True):
        mask = (1 - self.fog).astype(np.float32)
        return mask.flatten() if flatten else mask

    def reset(self, new_bomb_no=None):
        if new_bomb_no is not None:
            self.bomb_no = int(new_bomb_no)

        if self.bomb_no < 1:
            raise ValueError("bomb_no 至少为 1")
        if self.bomb_no >= self.box_count:
            raise ValueError("bomb_no 必须小于总格子数，否则无可玩空间")

        self.grid = zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.fog = zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.bomb_locs = random.choice(self.box_count, self.bomb_no, replace=False)
        self.uncovered_count = 0

        for loc in self.bomb_locs:
            row = int(loc // self.grid_width)
            col = int(loc % self.grid_width)
            self.grid[row, col] = -1

        for loc in self.bomb_locs:
            row = int(loc // self.grid_width)
            col = int(loc % self.grid_width)
            for r in range(max(0, row - 1), min(self.grid_height, row + 2)):
                for c in range(max(0, col - 1), min(self.grid_width, col + 2)):
                    if self.grid[r, c] != -1:
                        self.grid[r, c] += 1

        self.update_state()
        return self.state

    def update_state(self):
        self.state = multiply(self.fog, self.grid)
        self.state = add(self.state, (self.fog - 1))

    def plant_bombs(self):
        reordered_bomb_locs = []
        for bomb_loc in self.bomb_locs:
            row = int(bomb_loc // self.grid_width)
            col = int(bomb_loc % self.grid_width)
            self.grid[row, col] = -1
            reordered_bomb_locs.append((row, col))
        self.bomb_locs = reordered_bomb_locs

    def hint_maker(self):
        for row, col in self.bomb_locs:
            for r in range(row - 1, row + 2):
                for c in range(col - 1, col + 2):
                    if 0 <= r < self.grid_height and 0 <= c < self.grid_width and self.grid[r, c] != -1:
                        self.grid[r, c] += 1

    def choose(self, row, col):
        if row < 0 or row >= self.grid_height or col < 0 or col >= self.grid_width:
            return self.state, True, -1.0

        if self.grid[row, col] == -1:
            self.fog[row, col] = 1
            self.update_state()
            return self.state, True, -1.0

        if self.fog[row, col] == 1:
            return self.state, True, -1.0

        old_uncovered = self.uncovered_count

        if self.grid[row, col] == 0:
            unfog_zeros(self.grid, self.fog, row, col)
        else:
            self.fog[row, col] = 1

        self.update_state()
        self.uncovered_count = count_nonzero(self.fog)
        newly_opened = self.uncovered_count - old_uncovered

        if self.uncovered_count == self.box_count - self.bomb_no:
            return self.state, True, 1.0

        step_reward = min(0.05 * newly_opened, 0.5)
        return self.state, False, step_reward

    def step(self, action_idx):
        row = int(action_idx // self.grid_width)
        col = int(action_idx % self.grid_width)
        next_state, done, reward = self.choose(row, col)
        return next_state.flatten(), reward, done, self.get_mask(flatten=True)

    def step_spatial(self, action_idx):
        row = int(action_idx // self.grid_width)
        col = int(action_idx % self.grid_width)
        next_state, done, reward = self.choose(row, col)
        return next_state.copy(), reward, done, self.get_mask(flatten=False)


@njit(fastmath=True)
def unfog_zeros(grid, fog, row, col):
    h, w = grid.shape
    stack = []
    stack.append((row, col))
    while len(stack) > 0:
        i, j = stack.pop()
        for r in range(i - 1, i + 2):
            for c in range(j - 1, j + 2):
                if 0 <= r < h and 0 <= c < w:
                    if grid[r, c] == 0 and fog[r, c] == 0:
                        stack.append((r, c))
                    fog[r, c] = 1


def speed_test(iterations=100000):
    start = time.perf_counter()
    for _ in range(iterations):
        game = Minesweeper(10, 16, 20)
        game.choose(5, 5)
    end = time.perf_counter()
    return end - start


if __name__ == '__main__':
    used_time = speed_test()
    print(used_time)
