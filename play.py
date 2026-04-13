import argparse
from pathlib import Path

import pygame

from game import Minesweeper
from help import Help, get_default_model_path
from renderer import Renderer


class Play:
    def __init__(self, width=15, height=15, no=15, block_size=40, model_path=None):
        self.width = int(width)
        self.height = int(height)
        self.no = int(no)
        self.model_path = model_path or get_default_model_path()
        self.env = Minesweeper(self.width, self.height, self.no)
        self.renderer = Renderer(state=self.env.state, block_size=block_size, auto_scale=True, ui_height=70)
        self.help_button_rect = pygame.Rect(10, 18, 120, 34)
        self.helper = None

        if self.model_path:
            try:
                self.helper = Help(self.model_path)
            except Exception as exc:
                self.helper = None
                print(f"AI 帮助模块初始化失败: {exc}")

        self.model_name = Path(self.model_path).name if self.model_path else "未设置默认模型"
        self.status_message = "左键点击格子开始游戏"

    def draw_ui(self):
        pygame.draw.rect(self.renderer.screen, (0, 120, 215), self.help_button_rect, border_radius=6)
        pygame.draw.rect(self.renderer.screen, (255, 255, 255), self.help_button_rect, width=2, border_radius=6)

        btn_text = self.renderer.small_font.render("AI 帮助", True, (255, 255, 255))
        btn_rect = btn_text.get_rect(center=self.help_button_rect.center)
        self.renderer.screen.blit(btn_text, btn_rect)

        self.renderer.addUiText(f"模型: {self.model_name}", 145, 10, small=True)
        self.renderer.addUiText(f"状态: {self.status_message}", 145, 38, small=True)

    def draw(self):
        self.renderer.draw()
        self.draw_ui()
        pygame.display.update()

    def click_cell(self, row, col, source="玩家"):
        next_state, terminal, reward = self.env.choose(row, col)
        self.renderer.state = next_state
        self.status_message = f"{source}点击: ({row}, {col}) | 奖励: {reward:.2f}"
        self.draw()
        return next_state, terminal, reward, (row, col)

    def click_board(self, x, y):
        cell = self.renderer.screen_to_grid(x, y)
        if cell is None:
            return self.env.state, False, 0.0, None

        row, col = cell
        return self.click_cell(row, col, source="玩家")

    def click_help_button(self):
        if self.helper is None:
            self.status_message = "未设置可用默认模型，请先在 main.py 中选择模型"
            self.draw()
            return self.env.state, False, 0.0, None

        try:
            suggestion = self.helper.best_action_from_env(self.env)
        except Exception as exc:
            self.status_message = f"AI 帮助失败: {exc}"
            self.draw()
            print(self.status_message)
            return self.env.state, False, 0.0, None

        if suggestion is None:
            self.status_message = "当前没有可点击的格子"
            self.draw()
            return self.env.state, False, 0.0, None

        row, col = suggestion["row"], suggestion["col"]
        next_state, terminal, reward = self.env.choose(row, col)
        self.renderer.state = next_state
        self.status_message = f"AI 推荐并点击: ({row}, {col}) | Q={suggestion['score']:.3f}"
        self.draw()
        return next_state, terminal, reward, (row, col)

    def handle_mouse_click(self, x, y):
        logical_x, logical_y = self.renderer.normalize_mouse_pos((x, y))
        if self.help_button_rect.collidepoint(logical_x, logical_y):
            return self.click_help_button()
        return self.click_board(x, y)

    def reset_board(self):
        self.env.reset()
        self.renderer.state = self.env.state
        self.status_message = "已重开新的一局"
        self.draw()


def main(width=8, height=8, no=10, block_size=40, model_path=None):
    play = Play(width=width, height=height, no=no, block_size=block_size, model_path=model_path)
    play.draw()

    while True:
        events = play.renderer.catchEvent()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if hasattr(event, 'pos'):
                    x, y = event.pos
                else:
                    x, y = pygame.mouse.get_pos()

                next_state, terminal, reward, cell = play.handle_mouse_click(x, y)
                if cell is None:
                    continue

                print(f'点击格子: {cell} | 奖励: {reward}')
                if terminal:
                    if reward == -1:
                        play.status_message = '踩雷了，准备重开'
                        print('Bomb')
                    else:
                        play.status_message = '成功通关，准备重开'
                        print('Win')
                    play.draw()
                    pygame.time.delay(650)
                    play.reset_board()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="手动玩扫雷（带 AI 帮助按钮）")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--mines", type=int, default=10)
    parser.add_argument("--block_size", type=int, default=40)
    parser.add_argument("--model_path", type=str, default=get_default_model_path() or "")
    args = parser.parse_args()
    main(args.width, args.height, args.mines, args.block_size, args.model_path or None)
