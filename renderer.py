#起到渲染器的作用，把抽象的state映射为图像

import pygame



class Renderer():

    #初始化翻开和未翻开区域的颜色
    def __init__(self,state):
        h,w = state.shape
        self.block_size = 40
        self.window_h = h * self.block_size
        self.window_w = w * self.block_size
        self.state = state

        self.Back_colors ={
            'shadow':(128, 128, 128),  # 阴影灰色：用于未翻开的格子边缘
            'opened':(255, 255, 255),  # 翻开后的纯白色背景
            'unopened':(192, 192, 192)  # 未翻开的浅灰色块
        }

        self.Num_colors = {
            1: (0, 0, 255),      # 1-蓝色 (Blue)
            2: (0, 128, 0),      # 2-绿色 (Green)
            3: (255, 0, 0),      # 3-红色 (Red)
            4: (0, 0, 128),      # 4-青色/深蓝 (Navy)
            5: (128, 0, 0),      # 5-红褐色/橙红 (Maroon)
            6: (0, 128, 128),    # 6-青色 (Teal)
            7: (0, 0, 0),        # 7-黑色 (Black)
            8: (128, 128, 128),  # 8-灰色 (Gray)
        }
        self.init()

    def init(self):
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 25 , bold=True)
        self.screen = pygame.display.set_mode((self.window_h, self.window_w))
        self.clock = pygame.time.Clock()

    def draw(self):
        self.drawGrid()
        pygame.display.update()

    #用于debug
    def whereBugs(self):
        return pygame.event.get()

    def addText(self,number,x,y):
        self.screen.blit(self.font.render(str(number),True,color=self.Num_colors[number]),(x,y))
        pygame.display.update()

    def drawGrid(self):
        j = 0
        for column in range(0,self.window_w,self.block_size):
            i = 0
            for row in range(0,self.window_h,self.block_size):
                if self.state[i][j] == -1:
                    pygame.draw.rect(self.screen,self.Back_colors['unopened'],self.block_size)
                if self.state[i][j] >= 0:
                    pygame.draw.rect(self.screen,self.Back_colors['opened'],self.block_size)
                if self.state[i][j] > 0:
                    self.addText(self.state[i][j],column+12,row+8)
                i += 1
            j += 1