import math

import pygame
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap


def get_colormap(name: str) -> list[[int, int, int]]:
    colors = []
    for color in colormaps[name].colors:
        colors.append([math.floor(channel * 256) for channel in color])
    return colors


def mandelbrot(x: float, y: float, cutoff: int) -> int:
    """Compute the margins of the mandelbrot set"""
    z = 0 + 0j
    c = x + y * 1j
    iterations = 0
    while iterations < cutoff and abs(z) <= 2:
        z = z ** 2 + c
        iterations += 1
    # The first iteration could be considered the zeroth, as z will always be 0
    # in that iteration, so the loop will be executed at least once.
    return iterations - 1


def compute_mandelbrot(width: int, height: int, x: [float, float], y: [float, float], cutoff: int):
    pixes = np.zeros((width, height))
    x_scale = abs(x[0] - x[1]) / width
    y_scale = abs(y[0] - y[1]) / height

    for i in range(width):
        for j in range(height):
            pixes[i][j] = mandelbrot(x[0] + i * x_scale, y[0] + j * y_scale, cutoff)
    return pixes


def apply_colormap(divergence: np.array, cutoff: int, colormap: list[[float, float, float]]):
    conv = np.floor(divergence / cutoff * len(colormap)).astype(np.uint64)
    n, m = conv.shape
    pixels = np.zeros((n, m, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            pixels[i, j, :] = colormap[conv[i][j]]
    return pixels


def ranges(screen: pygame.Surface, center: pygame.Vector2, size: float) -> [[float, float], [float, float]]:
    x_center, y_center = center
    width, height = screen.get_width(), screen.get_height()
    ratio = height / width
    x_range = (x_center - size / 2, x_center + size / 2)
    y_range = ((y_center - size / 2) * ratio, (y_center + size / 2) * ratio)
    return x_range, y_range


def mouse_direction(screen: pygame.Surface) -> pygame.Vector2:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    width, height = screen.get_width(), screen.get_height()
    return pygame.Vector2(
        x=(mouse_x - width / 2) / width,
        y=(mouse_y - height / 2) / height,
    )


def mouse_position(screen: pygame.Surface, center: pygame.Vector2, size: float) -> pygame.Vector2:
    direction = mouse_direction(screen)
    x_range, y_range = ranges(screen, center, size)
    return pygame.Vector2(
        x=direction.x * (x_range[1] - x_range[0]),
        y=direction.y * (y_range[1] - y_range[0]),
    )


def get_possible_colors() -> list[str]:
    # These are specified in matplotlib's documentation
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
    quantitative_colors = {
        'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
        'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
        'tab20c'
    }
    # Most colors also have an "_r" variant
    quantitative_colors |= {f"{color}_r" for color in quantitative_colors}
    return [
        color for color in colormaps.keys()
        if isinstance(colormaps[color], ListedColormap) and color not in quantitative_colors
    ]


def run():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("Mandelbrot")

    clock = pygame.time.Clock()
    running = True

    color = "magma"
    possible_colors = get_possible_colors()
    colors = get_colormap(color)
    font = pygame.sysfont.SysFont("helveticaneue", 24)

    # The (mathematical) center of the screen
    center = pygame.Vector2(-1, 0)
    # The (mathematical) with of the screen
    size = 2
    zoom_factor = 1.2

    dt = 0

    cutoff = 10

    detail_scale = 1.3
    info = pygame.rect.Rect(0, 0, 400, 100)

    def handle_events():
        nonlocal running, center, size, x_range, y_range, cutoff

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                left, middle, right = event.buttons
                if left:
                    x_range, y_range = ranges(screen, center, size)
                    diff = pygame.Vector2(
                        x=event.rel[0] / screen.get_width() * (x_range[1] - x_range[0]),
                        y=event.rel[1] / screen.get_height() * (y_range[1] - y_range[0]),
                    )
                    center = center - diff
            elif event.type == pygame.MOUSEWHEEL:
                mouse_position_before_zoom = mouse_position(screen, center, size)
                if event.precise_y < 0:
                    # Zoom out
                    size *= zoom_factor
                else:
                    # Zoom in
                    size /= zoom_factor
                mouse_position_after_zoom = mouse_position(screen, center, size)

                diff = (mouse_position_before_zoom - mouse_position_after_zoom) * zoom_factor
                center += diff

        keys = pygame.key.get_pressed()
        if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:
            # Increase 'resolution'
            cutoff = max(int(cutoff * detail_scale), cutoff + 1)
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            # Decrease 'resolution'
            cutoff = max(2, min(int(cutoff / detail_scale), cutoff - 1))
        if keys[pygame.K_c]:
            nonlocal color, colors
            direction = -1 if keys[pygame.K_LSHIFT] else 1
            next_color = (possible_colors.index(color) + direction) % len(possible_colors)
            color = possible_colors[next_color]
            colors = get_colormap(color)

    while running:
        handle_events()

        x_range, y_range = ranges(screen, center, size)
        divergence = compute_mandelbrot(
            screen.get_width(), screen.get_height(),
            x_range, y_range, cutoff
        )

        pixels = apply_colormap(divergence, cutoff, colors)
        pygame.surfarray.blit_array(screen, pixels)

        # flip() the display to put your work on screen
        pygame.display.flip()

        dt = clock.tick(60) / 1000
        screen.blit(font.render(
            f"{1 / dt:.2f} fps"
            if dt < 1 else
            f"{dt} spf",
            1, (255, 255, 255)
        ), info)
        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    run()
