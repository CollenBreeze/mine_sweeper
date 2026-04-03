import pygame
from game import Minesweeper
from renderer import Renderer

class Play:
    def __init__(self,width = 15,height = 15,no = 15):
        self.width = width
        self.height = height
        self.no = no
        self.env = Minesweeper(self.width, self.height, self.no)
        self.renderer = Renderer(state=self.env.state)

    #获取鼠标按下位置xy所对应的grid中的行列坐标
    def click(self,x,y):
        i = int(0.4 * x -6)//self.width
        j = int(0.4 * y -8)//self.height
        next_state , terminal , reward = self.env.choose(i,j)
        self.renderer.state = next_state
        self.renderer.draw()
        return next_state,terminal,reward

def main():
    play = Play()
    play.renderer.draw()
    # 供开发者调试用
    print(play.env.grid)

    while True:
        events = play.renderer.catchEvent()
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                y,x = pygame.mouse.get_pos()
                next_state,terminal,reward = play.click(x,y)
                print(reward)
                print(play.env.uncovered_count)
                if terminal:
                    if reward == -1:
                        print('Bomb')
                    else:
                        print('Win')
                    play.env.reset()
                    play.renderer.state = play.env.state
                    play.renderer.draw()
                    print(play.env.grid)

if __name__ == '__main__':
    main()