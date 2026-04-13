import os
import pygame

from game import Minesweeper


class Renderer:
    def __init__(self, state, font_path="QINGNIAO.ttf", block_size=40, auto_scale=True, ui_height=0):
        rows, cols = state.shape
        self.block_size = int(block_size)
        self.rows = int(rows)
        self.cols = int(cols)
        self.ui_height = max(0, int(ui_height))
        self.window_h = self.rows * self.block_size + self.ui_height
        self.window_w = self.cols * self.block_size
        self.state = state
        self.font_path = font_path
        self.auto_scale = bool(auto_scale)
        self.scaled_mode_enabled = False
        self.logical_surface_size = (self.window_w, self.window_h)
        self.actual_window_size = (self.window_w, self.window_h)
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.Back_colors = {
            'shadow': (128, 128, 128),
            'opened': (255, 255, 255),
            'unopened': (192, 192, 192),
            'ui_panel': (70, 70, 70),
            'ui_text': (255, 255, 255),
        }
        self.Num_colors = {
            1: (0, 0, 255),
            2: (0, 128, 0),
            3: (255, 0, 0),
            4: (0, 0, 128),
            5: (128, 0, 0),
            6: (0, 128, 128),
            7: (0, 0, 0),
            8: (128, 128, 128),
        }
        self.init()

    def init(self):
        pygame.init()
        font_file = self.font_path if os.path.exists(self.font_path) else pygame.font.get_default_font()
        self.font = pygame.font.Font(font_file, 25)
        self.small_font = pygame.font.Font(font_file, 18)

        scaled_flag = getattr(pygame, 'SCALED', 0)
        flags = scaled_flag if self.auto_scale else 0
        try:
            self.screen = pygame.display.set_mode((self.window_w, self.window_h), flags)
            self.scaled_mode_enabled = bool(flags & scaled_flag)
        except pygame.error:
            self.screen = pygame.display.set_mode((self.window_w, self.window_h))
            self.scaled_mode_enabled = False

        self.clock = pygame.time.Clock()
        self.refresh_display_metrics()

    def refresh_display_metrics(self):
        logical_w, logical_h = self.screen.get_size()
        try:
            actual_w, actual_h = pygame.display.get_window_size()
        except Exception:
            actual_w, actual_h = logical_w, logical_h

        self.logical_surface_size = (int(logical_w), int(logical_h))
        self.actual_window_size = (int(actual_w), int(actual_h))
        self.scale_x = (actual_w / logical_w) if logical_w else 1.0
        self.scale_y = (actual_h / logical_h) if logical_h else 1.0
        return {
            'logical_size': self.logical_surface_size,
            'actual_size': self.actual_window_size,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'scaled_mode_enabled': self.scaled_mode_enabled,
        }

    def draw(self):
        self.screen.fill(self.Back_colors['shadow'])
        if self.ui_height > 0:
            pygame.draw.rect(
                self.screen,
                self.Back_colors['ui_panel'],
                (0, 0, self.window_w, self.ui_height),
            )
            pygame.draw.line(
                self.screen,
                self.Back_colors['opened'],
                (0, self.ui_height - 1),
                (self.window_w, self.ui_height - 1),
                1,
            )
        self.drawGrid()
        pygame.display.update()

    def catchEvent(self):
        return pygame.event.get()

    def addText(self, number, x, y):
        txt = self.font.render(str(int(number)), True, self.Num_colors[number])
        rect = txt.get_rect(center=(x + self.block_size // 2, y + self.block_size // 2))
        self.screen.blit(txt, rect)

    def addUiText(self, text, x, y, small=True):
        font = self.small_font if small else self.font
        txt = font.render(str(text), True, self.Back_colors['ui_text'])
        self.screen.blit(txt, (x, y))

    def normalize_mouse_pos(self, pos):
        self.refresh_display_metrics()
        x, y = float(pos[0]), float(pos[1])
        logical_w, logical_h = self.logical_surface_size

        sx = self.scale_x if self.scale_x > 0 else 1.0
        sy = self.scale_y if self.scale_y > 0 else 1.0

        candidates = [
            (x, y, 'logical'),
            (x / sx, y / sy, 'physical_to_logical'),
        ]

        def score(candidate):
            cx, cy, label = candidate
            inside = 0 <= cx < logical_w and 0 <= cy < logical_h
            overflow = 0.0
            if cx < 0:
                overflow += -cx
            elif cx >= logical_w:
                overflow += cx - logical_w + 1
            if cy < 0:
                overflow += -cy
            elif cy >= logical_h:
                overflow += cy - logical_h + 1
            prefer = 0 if (self.scaled_mode_enabled and label == 'logical') else 1
            return (0 if inside else 1, prefer, overflow)

        best_x, best_y, _ = min(candidates, key=score)
        return best_x, best_y

    def screen_to_grid(self, x, y):
        lx, ly = self.normalize_mouse_pos((x, y))
        logical_w, logical_h = self.logical_surface_size
        if not (0 <= lx < logical_w and 0 <= ly < logical_h):
            return None

        if ly < self.ui_height:
            return None

        board_y = ly - self.ui_height
        col = int(lx // self.block_size)
        row = int(board_y // self.block_size)
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        return row, col

    def scale_debug_string(self):
        self.refresh_display_metrics()
        return (
            f"logical={self.logical_surface_size}, actual={self.actual_window_size}, "
            f"scale=({self.scale_x:.3f}, {self.scale_y:.3f}), scaled_mode={self.scaled_mode_enabled}"
        )

    def drawGrid(self):
        rows, cols = self.state.shape
        for row in range(rows):
            for col in range(cols):
                x = col * self.block_size
                y = self.ui_height + row * self.block_size
                value = self.state[row, col]
                if value == -1:
                    pygame.draw.rect(
                        surface=self.screen,
                        color=self.Back_colors['unopened'],
                        rect=(x, y, self.block_size - 1, self.block_size - 1),
                        border_radius=3,
                    )
                else:
                    pygame.draw.rect(
                        surface=self.screen,
                        color=self.Back_colors['opened'],
                        rect=(x, y, self.block_size - 1, self.block_size - 1),
                    )
                    if value > 0:
                        self.addText(int(value), x, y)


if __name__ == '__main__':
    test_env = Minesweeper(10, 16, 15)
    test_renderer = Renderer(state=test_env.state, ui_height=60)
    test_renderer.state = test_env.state
    print(test_renderer.catchEvent())
    print(test_renderer.scale_debug_string())
    test_renderer.draw()
