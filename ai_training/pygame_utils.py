from typing import Literal, Optional, Union
import pygame
import numpy as np
import pygame.freetype


def vec_interp(a, b, t):
    return vec_add_v(vec_mult_s(a, (1 - t)), vec_mult_s(b, t))


def vec_len(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def vec_norm(vec):
    v_len = vec_len(vec)
    return (vec[0] / v_len, vec[1] / v_len)


def vec_mult_s(vec, x):
    return (vec[0] * x, vec[1] * x)


def vec_sub_v(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vec_add_v(a, b):
    return (a[0] + b[0], a[1] + b[1])


def outline_surface(
    image: pygame.Surface,
    thickness: int,
    color: tuple,
    color_key: tuple = (255, 0, 255),
) -> pygame.Surface:
    mask = pygame.mask.from_surface(image)
    mask_surf = mask.to_surface(setcolor=color)
    mask_surf.set_colorkey((0, 0, 0))

    new_img = pygame.Surface((image.get_width() + 2, image.get_height() + 2))
    new_img.fill(color_key)
    new_img.set_colorkey(color_key)

    for i in -thickness, thickness:
        new_img.blit(mask_surf, (i + thickness, thickness))
        new_img.blit(mask_surf, (thickness, i + thickness))
    new_img.blit(image, (thickness, thickness))

    return new_img


def outline_text(
    font: pygame.freetype.Font,
    text: str,
    size: float,
    color: pygame.Color,
    outlinecolor: pygame.Color,
    outline: int,
) -> pygame.Surface:
    outlineSurf = font.render(text, outlinecolor, pygame.SRCALPHA, size=size)[0]
    outlineSize = outlineSurf.get_size()
    textSurf = pygame.Surface(
        (outlineSize[0] + outline * 2, outlineSize[1] + 2 * outline), pygame.SRCALPHA
    )
    textRect = textSurf.get_rect()
    offsets = [
        (ox, oy)
        for ox in range(-outline, 2 * outline)
        for oy in range(-outline, 2 * outline)
        if ox != 0 or ox != 0
    ]
    for ox, oy in offsets:
        px, py = textRect.center
        textSurf.blit(outlineSurf, outlineSurf.get_rect(center=(px + ox, py + oy)))
    innerText = font.render(text, color, pygame.SRCALPHA, size=size)[0].convert_alpha()
    textSurf.blit(innerText, innerText.get_rect(center=textRect.center))
    return textSurf


def render_outline_text(
    screen: pygame.Surface,
    pos: tuple[float, float],
    font: pygame.freetype.Font,
    text: Union[str, bytes, None],
    size: float,
    color: pygame.Color,
    outline_color: pygame.Color,
    outline_width: int,
):
    text_surf = outline_text(font, text, size, color, outline_color, outline_width)
    screen.blit(text_surf, pos)
