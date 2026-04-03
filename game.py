#导入库
import time
from threading import get_ident

import numpy as np
from numpy import zeros , random , multiply , count_nonzero , add
from numpy import int_ as intnp
from numba import njit



#游戏
class Minesweeper():
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

    def reset(self):
        self.grid = zeros((self.grid_height, self.grid_width),dtype=intnp)
        self.fog = zeros((self.grid_height, self.grid_width),dtype=intnp)
        self.state = zeros((self.grid_height, self.grid_width),dtype=intnp)
        self.bomb_locs = random.choice(range(self.box_count),self.bomb_no,replace=False)
        self.uncovered_count = 0
        self.plant_bombs()      #埋炸弹
        self.hint_maker()       #生成提示
        self.update_state()     #用户获取状态更新

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
                    if i > -1 and j > -1 and i < self.grid_width and j < self.grid_height and self.grid[i][j] != -1:
                        self.grid[i][j] += 1

    #玩家选择grid里的一个格子
    def choose(self,i,j):

        if self.grid[i][j] == 0:
            unfog_zeros(self.grid,self.fog,i,j)
            self.uncovered_count = count_nonzero(self.fog)
            self.update_state()
            if self.uncovered_count == self.box_count-self.bomb_no:
                return self.state,True,1
            return self.state,False,0.5
        elif self.grid[i][j] > 0:
            self.fog[i][j] = 1
            self.uncovered_count = count_nonzero(self.fog)
            self.update_state()
            if self.uncovered_count == self.box_count-self.bomb_no:
                return self.state,True,1
            return self.state,False,0.5
        else:
            return self.state,True,-1

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