#导入库
import time
from threading import get_ident

import numpy as np
from numpy import zeros , random , multiply , count_nonzero , add
from numpy import int_ as intnp
from numba import njit



#游戏
class Minesweeper:
    def __init__(self,grid_width,grid_height,bomb_no):

        #grid -1雷 0空白 1-8雷数 生成后固定不变
        #fog 0没翻开 1翻开了 随玩家操作而变化
        #bomb_locs 包含一维炸弹位置
        #state 返回到玩家的状态，由grid和fog逻辑运算而来 -1未翻开/炸  0翻开 空白  1-8翻开 雷数

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.bomb_no = bomb_no
        self.box_count = grid_width * grid_height
        self.uncovered_count = 0

        self.reset()

    def get_mask(self):
        # self.fog 是 0 的地方（未翻开），mask 设为 1
        # self.fog 是 1 的地方（已翻开），mask 设为 0
        # .flatten() 是为了把二维矩阵变成 AI 需要的一维数组
        return (1 - self.fog).flatten().astype(np.float32)

    def reset(self, new_bomb_no=None):
        if new_bomb_no is not None:
            self.bomb_no = new_bomb_no  # 允许动态改变雷数

        self.grid = zeros((self.grid_width, self.grid_height))
        self.fog = zeros((self.grid_width, self.grid_height))
        self.bomb_locs = random.choice(self.box_count, self.bomb_no, replace=False)
        self.uncovered_count = 0

        for loc in self.bomb_locs:
            i = loc // self.grid_width
            j = loc % self.grid_width
            self.grid[i][j] = -1

        for loc in self.bomb_locs:
            i = loc // self.grid_width
            j = loc % self.grid_width
            for r in range(max(0, i - 1), min(self.grid_width, i + 2)):
                for c in range(max(0, j - 1), min(self.grid_height, j + 2)):
                    if self.grid[r][c] != -1:
                        self.grid[r][c] += 1
        self.update_state()

    #在用户选择完成，运算完成后，返回状态更新
    def update_state(self):
        self.state = multiply(self.fog,self.grid)
        self.state = add(self.state,(self.fog-1))

    #在初始化时，在grid里，埋雷，并重新定义bomb_locs
    def plant_bombs(self):
        reordered_bomb_locs = []
        grid_width = self.grid_width
        for bomb_lock in self.bomb_locs:
            row = int(bomb_lock/grid_width)
            col = int(bomb_lock%grid_width)
            self.grid[row][col] = -1
            reordered_bomb_locs.append((row,col))
        self.bomb_locs = reordered_bomb_locs

    #在埋雷后，在grid里，写提示数字
    def hint_maker(self):
        for r,c in self.bomb_locs:
            for i in range(r-1,r+2):
                for j in range(c-1,c+2):
                    if i > -1 and j > -1 and i < self.grid_height and j < self.grid_width and self.grid[i][j] != -1:
                        self.grid[i][j] += 1

    #玩家选择grid里的一个格子
    def choose(self, i, j):
        # 1. 踩雷暴毙 (重罚)
        if self.grid[i][j] == -1:
            self.fog[i][j] = 1
            self.update_state()
            return self.state, True, -1.0

            # 2. 点到了已经翻开的格子 (虽然有Mask拦截，但为了防止死循环，若发生直接重罚并结束该局)
        if self.fog[i][j] == 1:
            return self.state, True, -1.0

        old_uncovered = self.uncovered_count

        # 3. 泛洪算法或普通点击
        if self.grid[i][j] == 0:
            unfog_zeros(self.grid, self.fog, i, j)
        else:
            self.fog[i][j] = 1

        self.update_state()
        self.uncovered_count = count_nonzero(self.fog)
        newly_opened = self.uncovered_count - old_uncovered

        # 4. 胜利条件 (大奖)
        if self.uncovered_count == self.box_count - self.bomb_no:
            return self.state, True, 1.0

        # 5. 安全步：奖励与翻开格子数成正比，但单步上限设为 0.5，防止超过赢局奖励
        step_reward = min(0.05 * newly_opened, 0.5)
        return self.state, False, step_reward

    def step(self, action_idx):
        # 1. 自动转换坐标
        i = action_idx // self.grid_width
        j = action_idx % self.grid_width

        # 2. 调用你之前的 choose 方法
        next_state, done, reward = self.choose(i, j)

        # 3. 返回 AI 需要的一维数据
        return next_state.flatten(), reward, done, self.get_mask()

#对于0的部分，采用泛洪算法，使用堆栈的方法进行快速计算
#pop（0）采用广度优先算法，（）采用深度优先算法
@njit(fastmath=True)
def unfog_zeros(grid,fog,i,j):
    h,w = grid.shape
    que = []
    que.append((i,j))
    while len(que) > 0:
        i,j = que.pop()
        for r in range(i-1,i+2):
            for c in range(j-1,j+2):
                if (r >= 0 and r < h and c >= 0 and c < w):
                    if (grid[r][c] == 0 and fog[r][c] == 0):
                        que.append((r,c))
                    fog[r][c] = 1

#测试逻辑运行速度
def speed_test(iterations=100000):
    start = time.perf_counter()
    for _ in range(iterations):
        game = Minesweeper(10,10,2)
        game.choose(5,5)
    end = time.perf_counter()
    return end-start

if __name__ == '__main__':
    used_time = speed_test()
    print(used_time)