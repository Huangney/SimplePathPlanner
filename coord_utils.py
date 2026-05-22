# -*- coding: utf-8 -*-
"""Coordinate conversion utilities."""

from __future__ import annotations

import numpy as np
from app_config import GRID_WIDTH, GRID_HEIGHT


def grid_data_bounds(has_image: bool, img_w: int, img_h: int, grid_x0: float, grid_y0: float, grid_x1: float, grid_y1: float):
    if not has_image:
        return 0.0, 0.0, float(GRID_WIDTH), float(GRID_HEIGHT)
    if grid_x0 == 0 and grid_y0 == 0 and grid_x1 == 0 and grid_y1 == 0:
        return 0.0, 0.0, float(img_w), float(img_h)
    return float(grid_x0), float(grid_y0), float(grid_x1), float(grid_y1)


def grid_to_data(gx: float, gy: float, has_image: bool, img_w: int, img_h: int, grid_x0: float, grid_y0: float, grid_x1: float, grid_y1: float):
    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h, grid_x0, grid_y0, grid_x1, grid_y1)
    dx = dx0 + gy / GRID_WIDTH * (dx1 - dx0)
    dy = dy0 + gx / GRID_HEIGHT * (dy1 - dy0)
    return dx, dy


def data_to_grid(dx: float, dy: float, has_image: bool, img_w: int, img_h: int, grid_x0: float, grid_y0: float, grid_x1: float, grid_y1: float):
    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h, grid_x0, grid_y0, grid_x1, grid_y1)
    if dx1 == dx0 or dy1 == dy0:
        return None, None
    gx = (dy - dy0) / (dy1 - dy0) * GRID_HEIGHT
    gy = (dx - dx0) / (dx1 - dx0) * GRID_WIDTH
    return gx, gy


def grid_vec_to_data_vec(vx: float, vy: float, has_image: bool, img_w: int, img_h: int, grid_x0: float, grid_y0: float, grid_x1: float, grid_y1: float):
    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h, grid_x0, grid_y0, grid_x1, grid_y1)
    vdx = vy / GRID_WIDTH * (dx1 - dx0)
    vdy = vx / GRID_HEIGHT * (dy1 - dy0)
    return vdx, vdy


def format_coord_status(x: float, y: float, has_image: bool, img_w: int, img_h: int, grid_x0: float, grid_y0: float, grid_x1: float, grid_y1: float):
    data_part = f"Data: ({x:.1f}, {y:.1f})"
    gx, gy = data_to_grid(x, y, has_image, img_w, img_h, grid_x0, grid_y0, grid_x1, grid_y1)
    if gx is None or gy is None:
        return f"Grid: (out)  |  {data_part}"
    if not (0.0 <= gx <= GRID_HEIGHT and 0.0 <= gy <= GRID_WIDTH):
        return f"Grid: (out)  |  {data_part}"
    return f"Grid: ({gx:.3f}, {gy:.3f})  |  {data_part}"
