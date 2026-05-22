# -*- coding: utf-8 -*-
"""Coordinate conversion utilities."""

from __future__ import annotations

import numpy as np
from app_config import GRID_WIDTH, GRID_HEIGHT, GRID_X0, GRID_Y0, GRID_X1, GRID_Y1


def grid_data_bounds(has_image: bool, img_w: int, img_h: int):
    if not has_image:
        return 0.0, 0.0, float(GRID_WIDTH), float(GRID_HEIGHT)
    if GRID_X0 == 0 and GRID_Y0 == 0 and GRID_X1 == 0 and GRID_Y1 == 0:
        return 0.0, 0.0, float(img_w), float(img_h)
    return float(GRID_X0), float(GRID_Y0), float(GRID_X1), float(GRID_Y1)


def grid_to_data(gx: float, gy: float, has_image: bool, img_w: int, img_h: int):
    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h)
    dx = dx0 + gy / GRID_WIDTH * (dx1 - dx0)
    dy = dy0 + gx / GRID_HEIGHT * (dy1 - dy0)
    return dx, dy


def data_to_grid(dx: float, dy: float, has_image: bool, img_w: int, img_h: int):
    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h)
    if dx1 == dx0 or dy1 == dy0:
        return None, None
    gx = (dy - dy0) / (dy1 - dy0) * GRID_HEIGHT
    gy = (dx - dx0) / (dx1 - dx0) * GRID_WIDTH
    return gx, gy


def grid_vec_to_data_vec(vx: float, vy: float, has_image: bool, img_w: int, img_h: int):
    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h)
    vdx = vy / GRID_WIDTH * (dx1 - dx0)
    vdy = vx / GRID_HEIGHT * (dy1 - dy0)
    return vdx, vdy


def format_coord_status(x: float, y: float, has_image: bool, img_w: int, img_h: int):
    data_part = f"Data: ({x:.1f}, {y:.1f})"
    if not has_image:
        return data_part
    px = int(np.floor(x))
    py = int(np.floor(img_h - 1 - y))
    if 0 <= px < img_w and 0 <= py < img_h:
        return f"Pixel: ({px}, {py})  |  {data_part}"
    return f"Pixel: (out)  |  {data_part}"

