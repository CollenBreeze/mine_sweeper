import pygame

from game import Minesweeper
from renderer import Renderer


class Play:
    def __init__(self, width=15, height=15, no=25, block_size=40):
        self.width = int(width)
        self.height = int(height)
        self.no = int(no)
        self.env = Minesweeper(self.width, self.height, self.no)
        self.renderer = Renderer(state=self.env.state, block_size=block_size, auto_scale=True)

    def click(self, x, y):
        cell = self.renderer.screen_to_grid(x, y)
        if cell is None:
            return self.env.state, False, 0.0, None

        row, col = cell
        next_state, terminal, reward = self.env.choose(row, col)
        self.renderer.state = next_state
        self.renderer.draw()
        return next_state, terminal, reward, (row, col)


def main():
    play = Play()
    play.renderer.draw()
    print(play.env.grid)
    print('显示缩放检测:', play.renderer.scale_debug_string())

    while True:
        events = play.renderer.catchEvent()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                if hasattr(event, 'pos'):
                    x, y = event.pos
                else:
                    x, y = pygame.mouse.get_pos()

                next_state, terminal, reward, cell = play.click(x, y)
                if cell is None:
                    continue

                print(f'点击格子: {cell} | 奖励: {reward}')
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
