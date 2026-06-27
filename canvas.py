# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from app_config import (
    ProfileConfig,
    get_profile_config,
    BACKGROUND_ALPHA,
    DEFAULT_PATH_DENSITY,
    DEFAULT_MAX_V,
    DEFAULT_MAX_A,
    DEFAULT_MAX_W,
    DEFAULT_MAX_AW,
    DEFAULT_MAX_JK,
    DEFAULT_LAT_ACCEL_MAX,
)
from path_planner import PathSamples, SpeedLimits, export_path_cpp
from canvas_render import CanvasRenderMixin
from canvas_commands import CanvasCommandMixin


def _configure_chinese_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "DengXian",
        "KaiTi",
        "FangSong",
        "STKaiti",
        "STFangsong",
    ]
    chosen = next((name for name in candidates if name in available), None)
    if chosen is None:
        return None

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [chosen, *[name for name in candidates if name != chosen], "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


_CHINESE_FONT = _configure_chinese_font()


class GridCanvas(CanvasRenderMixin, CanvasCommandMixin):
    def __init__(self, profile_config: ProfileConfig | None = None):
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.fig.canvas.manager.set_window_title("SimplePathPlanner")
        self.profile_config = profile_config if profile_config is not None else get_profile_config(0)

        self._has_image = False
        self._img_w = 0
        self._img_h = 0
        self._running = True
        self.points = []
        self.path_samples = PathSamples(
            x=np.array([], dtype=float),
            y=np.array([], dtype=float),
            theta=np.array([], dtype=float),
            s=np.array([], dtype=float),
            t=np.array([], dtype=float),
            xdot=np.array([], dtype=float),
            ydot=np.array([], dtype=float),
            w=np.array([], dtype=float),
            v_lin=np.array([], dtype=float),
            meta={
                "segments": 0,
                "total_length": 0.0,
                "sample_count": 0,
                "total_time": 0.0,
                "peak_v": 0.0,
                "peak_w": 0.0,
                "constraint_clipped": False,
            },
        )
        self.path_density = DEFAULT_PATH_DENSITY
        self.show_path = True
        self._path_data_x = np.array([], dtype=float)
        self._path_data_y = np.array([], dtype=float)
        self._velo_legend_ax = None
        self._velo_legend = None
        self._hover_marker = None
        self._hover_text = None
        self._hover_text_mode = None
        self._hover_waypoint_idx = None
        self._hover_path_sample_idx = None
        self._hover_heading_arrow = None
        self._hover_velocity_arrow = None
        self._hover_body_patch = None
        self._last_mouse_grid_xy = None
        self.background_alpha = BACKGROUND_ALPHA
        self.body_length = None
        self.body_width = None
        self.speed_limits = SpeedLimits(
            max_v=DEFAULT_MAX_V,
            max_a=DEFAULT_MAX_A,
            max_w=DEFAULT_MAX_W,
            max_aw=DEFAULT_MAX_AW,
            max_jk=DEFAULT_MAX_JK,
            lat_accel_max=DEFAULT_LAT_ACCEL_MAX,
        )
        self.solver = "coupled"

        self.coord_text = self.fig.text(
            0.01, 0.01, "", fontsize=9, va="bottom", ha="left",
            family="monospace", transform=self.fig.transFigure
        )
        self.shortcut_text = self.fig.text(
            0.01,
            0.99,
            "在任意处按 [A] 以添加新点；\n在已有曲线上按 [I] 以插入关键点；\n悬停已有关键点按 [D] 可删除；\n双击已有关键点，以编辑其属性",
            fontsize=12,
            va="top",
            ha="left",
            family=_CHINESE_FONT if _CHINESE_FONT is not None else "sans-serif",
            transform=self.fig.transFigure,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.70),
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._draw_coordinate_axes()
        self._rebuild_path()
        self._draw_path()
        self._draw_points()
        self._print_grid_info()
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
