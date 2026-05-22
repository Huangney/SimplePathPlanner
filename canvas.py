# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Arc
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from app_config import (
    ProfileConfig,
    get_profile_config,
    GRID_WIDTH,
    GRID_HEIGHT,
    BACKGROUND_ALPHA,
    DEFAULT_PATH_DENSITY,
    DEFAULT_MAX_V,
    DEFAULT_MAX_A,
    DEFAULT_MAX_W,
    DEFAULT_MAX_AW,
)
from coord_utils import (
    grid_data_bounds,
    grid_to_data,
    data_to_grid,
    grid_vec_to_data_vec,
    format_coord_status,
)
from path_planner import Waypoint, PathSamples, SpeedLimits, build_path, dump_session, load_session, export_path_cpp


class GridCanvas:
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
            meta={"segments": 0, "total_length": 0.0, "sample_count": 0, "total_time": 0.0, "peak_v": 0.0, "peak_w": 0.0, "constraint_clipped": False},
        )
        self.path_density = DEFAULT_PATH_DENSITY
        self.show_path = True
        self.speed_limits = SpeedLimits(
            max_v=DEFAULT_MAX_V,
            max_a=DEFAULT_MAX_A,
            max_w=DEFAULT_MAX_W,
            max_aw=DEFAULT_MAX_AW,
        )

        self.coord_text = self.fig.text(
            0.01, 0.01, "", fontsize=9, va="bottom", ha="left",
            family="monospace", transform=self.fig.transFigure
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
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

    def _grid_bounds_tuple(self):
        c = self.profile_config
        return c.grid_x0, c.grid_y0, c.grid_x1, c.grid_y1

    def _setup_view(self):
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        self.ax.format_coord = lambda x, y: format_coord_status(x, y, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)

    def _apply_limits(self):
        if self._has_image:
            self._zoom_to_rect_equivalent(0.0, 0.0, float(self._img_w), float(self._img_h))
        else:
            self.ax.set_xlim(0, GRID_WIDTH)
            self.ax.set_ylim(0, GRID_HEIGHT)

    def _load_background(self):
        background_image_path = self.profile_config.background_image_path
        if not background_image_path:
            return
        if not os.path.exists(background_image_path):
            print(f"[警告] 未找到背景图片: {background_image_path}")
            return
        self._img = mpimg.imread(background_image_path)
        self._img_h, self._img_w = self._img.shape[:2]
        self._has_image = True
        artist = self.ax.imshow(
            self._img,
            extent=[0, self._img_w, 0, self._img_h],
            origin="upper",
            aspect="equal",
            alpha=BACKGROUND_ALPHA,
            zorder=0,
        )
        artist.format_cursor_data = lambda data: ""
        print(f"[信息] 已加载背景图片：{self._img_w}x{self._img_h}  <-  {background_image_path}")

    def _zoom_to_rect_equivalent(self, x0: float, y0: float, x1: float, y1: float):
        # Mimic matplotlib "zoom to rectangle" with equal aspect:
        # adapt the selected data-rect to current axes ratio, then apply limits.
        if self._img_w <= 0 or self._img_h <= 0:
            return

        rx0, rx1 = (x0, x1) if x0 <= x1 else (x1, x0)
        ry0, ry1 = (y0, y1) if y0 <= y1 else (y1, y0)
        rw = max(1e-6, rx1 - rx0)
        rh = max(1e-6, ry1 - ry0)

        self.fig.canvas.draw()
        bbox = self.ax.get_window_extent()
        aw, ah = float(bbox.width), float(bbox.height)
        if aw <= 1.0 or ah <= 1.0:
            self.ax.set_xlim(rx0, rx1)
            self.ax.set_ylim(ry0, ry1)
            return

        axes_ratio = aw / ah
        rect_ratio = rw / rh
        cx = 0.5 * (rx0 + rx1)
        cy = 0.5 * (ry0 + ry1)

        if rect_ratio >= axes_ratio:
            vw = rw
            vh = rw / axes_ratio
        else:
            vh = rh
            vw = vh * axes_ratio

        nx0 = max(0.0, cx - 0.5 * vw)
        nx1 = min(float(self._img_w), cx + 0.5 * vw)
        ny0 = max(0.0, cy - 0.5 * vh)
        ny1 = min(float(self._img_h), cy + 0.5 * vh)
        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)

    def _on_resize(self, _event):
        if self._has_image:
            self._zoom_to_rect_equivalent(0.0, 0.0, float(self._img_w), float(self._img_h))
            self.fig.canvas.draw_idle()

    def _draw_grid_lines(self):
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        dx0, dy0, dx1, dy1 = grid_data_bounds(self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        for gy in range(GRID_WIDTH + 1):
            dx, _ = grid_to_data(0, gy, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            self.ax.plot([dx, dx], [dy0, dy1], color="black", linewidth=0.5, zorder=2)
        for gx in range(GRID_HEIGHT + 1):
            _, dy = grid_to_data(gx, 0, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            self.ax.plot([dx0, dx1], [dy, dy], color="black", linewidth=0.5, zorder=2)

    def _draw_coordinate_axes(self):
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        ox, oy = grid_to_data(0.0, 0.0, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        dx0, dy0, dx1, dy1 = grid_data_bounds(self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        span_x = abs(dx1 - dx0)
        span_y = abs(dy1 - dy0)
        len_y = 0.14 * span_y
        len_x = 0.28 * span_x

        self.ax.annotate("", xy=(ox, oy - len_y), xytext=(ox, oy),
                         arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=3.2), zorder=7)
        self.ax.annotate("", xy=(ox + len_x, oy), xytext=(ox, oy),
                         arrowprops=dict(arrowstyle="->", color="seagreen", lw=3.2), zorder=7)
        self.ax.text(ox, oy - len_y, " +x", color="dodgerblue", fontsize=9, va="top", ha="center", zorder=8)
        self.ax.text(ox + len_x, oy, " +y", color="seagreen", fontsize=9, va="center", ha="left", zorder=8)

        radius = 0.38 * min(len_x, len_y)
        self.ax.add_patch(Arc((ox, oy), 2 * radius, 2 * radius, angle=0, theta1=-90, theta2=0,
                              color="darkorange", lw=1.8, zorder=7))
        tail = (ox + radius * np.cos(np.deg2rad(-20)), oy + radius * np.sin(np.deg2rad(-20)))
        head = (ox + radius * np.cos(np.deg2rad(0)), oy + radius * np.sin(np.deg2rad(0)))
        self.ax.annotate("", xy=head, xytext=tail,
                         arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.8), zorder=8)
        self.ax.text(ox + 0.55 * radius, oy - 0.55 * radius, " +w", color="darkorange", fontsize=9, zorder=8)

    def _draw_points(self):
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        for idx, p in enumerate(self.points, start=1):
            dx, dy = grid_to_data(p.x, p.y, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            self.ax.plot(dx, dy, marker="o", markersize=6, color="red", zorder=5)

            hdx, hdy = grid_vec_to_data_vec(np.cos(p.theta), np.sin(p.theta), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            hnorm = np.hypot(hdx, hdy)
            if hnorm > 1e-9:
                self.ax.annotate("", xy=(dx + hdx * (42.0 / hnorm), dy + hdy * (42.0 / hnorm)), xytext=(dx, dy),
                                 arrowprops=dict(arrowstyle="->", color="limegreen", lw=2.0), zorder=6)

            if p.vx is not None and p.vy is not None:
                vdx, vdy = grid_vec_to_data_vec(float(p.vx), float(p.vy), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
                vnorm = np.hypot(vdx, vdy)
                if vnorm > 1e-9:
                    self.ax.annotate("", xy=(dx + vdx * (35.0 / vnorm), dy + vdy * (35.0 / vnorm)), xytext=(dx, dy),
                                     arrowprops=dict(arrowstyle="->", color="magenta", lw=1.8), zorder=6)
            self.ax.text(dx + 3, dy + 3, f"P{idx} ({p.x:.1f}, {p.y:.1f}, {p.theta:.2f})",
                         color="red", fontsize=8, zorder=6)

    def _draw_path(self):
        if not self.show_path or self.path_samples.x.size < 2:
            return
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        data_x, data_y = [], []
        for gx, gy in zip(self.path_samples.x, self.path_samples.y):
            dx, dy = grid_to_data(float(gx), float(gy), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            data_x.append(dx)
            data_y.append(dy)
        pts = np.column_stack([np.array(data_x, dtype=float), np.array(data_y, dtype=float)])
        segments = np.stack([pts[:-1], pts[1:]], axis=1)

        # Color trajectory by linear speed trend (low->high), and make the path thicker.
        speed = np.asarray(self.path_samples.v_lin, dtype=float)
        speed_seg = 0.5 * (speed[:-1] + speed[1:]) if speed.size >= 2 else np.zeros(segments.shape[0], dtype=float)
        if speed_seg.size == 0:
            return

        smin = float(np.min(speed_seg))
        smax = float(np.max(speed_seg))
        if abs(smax - smin) < 1e-9:
            lc = LineCollection(segments, colors="deepskyblue", linewidths=3.6, zorder=4)
        else:
            norm = Normalize(vmin=smin, vmax=smax)
            lc = LineCollection(segments, cmap="turbo", norm=norm, linewidths=3.6, zorder=4)
            lc.set_array(speed_seg)
        self.ax.add_collection(lc)

    def _rebuild_path(self):
        self.path_samples = build_path(self.points, density=self.path_density, speed_limits=self.speed_limits)

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        dx, dy = event.xdata, event.ydata
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        gx, gy = data_to_grid(dx, dy, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        parts = []
        if gx is not None and gy is not None and 0 <= gx <= GRID_HEIGHT and 0 <= gy <= GRID_WIDTH:
            parts.append(f"Grid: ({gx:.1f}, {gy:.1f})")
        if self._has_image:
            parts.append(f"Data: ({dx:.1f}, {dy:.1f})")
            parts.append(f"Image: {self._img_w}x{self._img_h}")
        else:
            parts.append(f"Pos: ({dx:.1f}, {dy:.1f})")
        self.coord_text.set_text("  |  ".join(parts))

    def _on_scroll_zoom(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2
        if event.button == "up":
            scale_factor = 1.0 / base_scale
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata, ydata = float(event.xdata), float(event.ydata)

        new_w = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]) if cur_xlim[1] != cur_xlim[0] else 0.5
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0]) if cur_ylim[1] != cur_ylim[0] else 0.5

        self.ax.set_xlim([xdata - new_w * (1.0 - relx), xdata + new_w * relx])
        self.ax.set_ylim([ydata - new_h * (1.0 - rely), ydata + new_h * rely])
        self.fig.canvas.draw_idle()

    def _print_grid_info(self):
        if not self._has_image:
            print(f"[信息] 无背景图，网格坐标直接映射：{GRID_WIDTH}x{GRID_HEIGHT}")
            return
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        dx0, dy0, dx1, dy1 = grid_data_bounds(self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        print(f"[信息] 网格数据坐标范围：x=[{dx0:.1f}, {dx1:.1f}]  y=[{dy0:.1f}, {dy1:.1f}]  "
              f"|  尺寸={dx1-dx0:.0f}x{dy1-dy0:.0f}")

    def redraw(self):
        self.ax.cla()
        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._draw_coordinate_axes()
        self._rebuild_path()
        self._draw_path()
        self._draw_points()
        self.fig.canvas.draw_idle()

    def start_terminal_loop(self):
        print("\n========== SimplePathPlanner 路径规划工具 ==========")
        self._print_help()
        while self._running:
            try:
                cmd = input("\n>>> ").strip().split()
                if cmd:
                    self._handle_command(cmd)
            except (EOFError, KeyboardInterrupt):
                break

    def _print_help(self):
        print("命令列表：")
        print("  help      显示帮助信息")
        print("  exit/q    退出程序")
        print("  grid      重绘画布")
        print("  addpoint x, y, theta[, vx, vy, vw]   添加路径点（网格坐标）")
        print("  editpoint idx x, y, theta[, vx, vy, vw]   修改指定路径点（idx 从 1 开始）")
        print("           坐标范围：x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]")
        print("  plan      重新规划路径并打印摘要")
        print("  density d 设置路径采样密度 (d >= 1.0)")
        print("  speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw>   设置全局速度约束")
        print("  showpath on/off   切换路径曲线显示")
        print("  save <文件>   保存当前路径点和设置到 JSON")
        print("  load <文件>   从 JSON 加载路径点和设置")
        print("  exportcpp <文件> [name=PathName] [scale=1.0]   导出 MCU C++ 路径头文件")

    def _handle_command(self, cmd):
        op = cmd[0].lower()
        if op in ("exit", "q"):
            self._running = False
            plt.close("all")
            print("再见。")
        elif op == "help":
            self._print_help()
        elif op == "grid":
            self.redraw()
            print("画布已重绘。")
        elif op == "addpoint":
            self._cmd_addpoint(cmd[1:])
        elif op == "editpoint":
            self._cmd_editpoint(cmd[1:])
        elif op == "plan":
            self._cmd_plan()
        elif op == "density":
            self._cmd_density(cmd[1:])
        elif op == "speedcfg":
            self._cmd_speedcfg(cmd[1:])
        elif op == "showpath":
            self._cmd_showpath(cmd[1:])
        elif op == "save":
            self._cmd_save(cmd[1:])
        elif op == "load":
            self._cmd_load(cmd[1:])
        elif op == "exportcpp":
            self._cmd_exportcpp(cmd[1:])
        else:
            print(f"未知命令: {op}。输入 'help' 查看可用命令。")

    def _cmd_addpoint(self, args):
        if not args:
            print("用法: addpoint x, y, theta")
            return
        parts = " ".join(args).replace(" ", "").split(",")
        if len(parts) not in (3, 6):
            print("用法: addpoint x, y, theta[, vx, vy, vw]")
            return
        try:
            nums = list(map(float, parts))
        except ValueError:
            print("数值格式无效。示例: addpoint 1.0, 1.0, 1.57, 0.5, 0.0, 0.2")
            return
        gx, gy, theta = nums[0], nums[1], nums[2]
        vx = vy = vw = None
        if len(nums) == 6:
            vx, vy, vw = nums[3], nums[4], nums[5]
        if not (0.0 <= gx <= GRID_HEIGHT and 0.0 <= gy <= GRID_WIDTH):
            print(f"路径点超出网格范围。x 在 [0,{GRID_HEIGHT}]，y 在 [0,{GRID_WIDTH}]")
            return
        self.points.append(Waypoint(x=gx, y=gy, theta=theta, vx=vx, vy=vy, vw=vw))
        self.redraw()
        if vx is None:
            print(f"路径点已添加：({gx:.3f}, {gy:.3f}, {theta:.3f})")
        else:
            print(f"路径点已添加：({gx:.3f}, {gy:.3f}, {theta:.3f}, vx={vx:.3f}, vy={vy:.3f}, vw={vw:.3f})")
            print("[信息] vx/vy/vw 作为该点世界坐标目标速度锚点。")

    def _cmd_editpoint(self, args):
        if len(args) < 2:
            print("用法: editpoint idx x, y, theta[, vx, vy, vw]")
            return
        if not self.points:
            print("[错误] 当前没有可修改的路径点。")
            return
        try:
            idx = int(args[0])
        except ValueError:
            print("索引格式无效。示例: editpoint 2 1.0, 2.0, 0.5")
            return
        if idx < 1 or idx > len(self.points):
            print(f"[错误] 路径点索引越界：{idx}（当前共有 {len(self.points)} 个点，合法范围 1~{len(self.points)}）")
            return

        parts = " ".join(args[1:]).replace(" ", "").split(",")
        if len(parts) not in (3, 6):
            print("用法: editpoint idx x, y, theta[, vx, vy, vw]")
            return
        try:
            nums = list(map(float, parts))
        except ValueError:
            print("数值格式无效。示例: editpoint 2 1.0, 1.0, 1.57, 0.5, 0.0, 0.2")
            return

        gx, gy, theta = nums[0], nums[1], nums[2]
        vx = vy = vw = None
        if len(nums) == 6:
            vx, vy, vw = nums[3], nums[4], nums[5]
        if not (0.0 <= gx <= GRID_HEIGHT and 0.0 <= gy <= GRID_WIDTH):
            print(f"路径点超出网格范围。x 在 [0,{GRID_HEIGHT}]，y 在 [0,{GRID_WIDTH}]")
            return

        self.points[idx - 1] = Waypoint(x=gx, y=gy, theta=theta, vx=vx, vy=vy, vw=vw)
        self.redraw()
        if vx is None:
            print(f"路径点 P{idx} 已修改为：({gx:.3f}, {gy:.3f}, {theta:.3f})")
        else:
            print(f"路径点 P{idx} 已修改为：({gx:.3f}, {gy:.3f}, {theta:.3f}, vx={vx:.3f}, vy={vy:.3f}, vw={vw:.3f})")
            print("[信息] vx/vy/vw 作为该点世界坐标目标速度锚点。")

    def _cmd_plan(self):
        self.redraw()
        meta = self.path_samples.meta
        print(
            f"[规划] 段数={meta.get('segments', 0)}  采样点={meta.get('sample_count', 0)}  "
            f"总长度={meta.get('total_length', 0.0):.3f}  总时长={meta.get('total_time', 0.0):.3f}s  "
            f"峰值线速={meta.get('peak_v', 0.0):.3f}  峰值角速={meta.get('peak_w', 0.0):.3f}  "
            f"约束裁剪={'是' if meta.get('constraint_clipped', False) else '否'}"
        )

    def _cmd_density(self, args):
        if len(args) != 1:
            print("用法: density <float>")
            return
        try:
            d = float(args[0])
        except ValueError:
            print("数值无效。示例: density 20")
            return
        if d < 1.0:
            print("密度值必须 >= 1.0")
            return
        self.path_density = d
        self.redraw()
        print(f"密度已设置：{self.path_density:.2f}")

    def _cmd_speedcfg(self, args):
        if not args:
            print("用法: speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw>")
            return
        mapping = {
            "vmax": "max_v",
            "amax": "max_a",
            "wmax": "max_w",
            "awmax": "max_aw",
        }
        updates = {
            "max_v": self.speed_limits.max_v,
            "max_a": self.speed_limits.max_a,
            "max_w": self.speed_limits.max_w,
            "max_aw": self.speed_limits.max_aw,
        }
        for token in args:
            if "=" not in token:
                print(f"参数格式无效: {token}")
                return
            key, val = token.split("=", 1)
            key = key.lower().strip()
            if key not in mapping:
                print(f"未知参数: {key}（可用: vmax, amax, wmax, awmax）")
                return
            try:
                fval = float(val)
            except ValueError:
                print(f"参数值无效: {token}")
                return
            if fval <= 0.0:
                print(f"参数必须 > 0: {token}")
                return
            updates[mapping[key]] = fval

        self.speed_limits = SpeedLimits(
            max_v=updates["max_v"],
            max_a=updates["max_a"],
            max_w=updates["max_w"],
            max_aw=updates["max_aw"],
        )
        self.redraw()
        print(
            "速度约束已更新："
            f"vmax={self.speed_limits.max_v:.3f}, "
            f"amax={self.speed_limits.max_a:.3f}, "
            f"wmax={self.speed_limits.max_w:.3f}, "
            f"awmax={self.speed_limits.max_aw:.3f}"
        )

    def _cmd_showpath(self, args):
        if len(args) != 1 or args[0].lower() not in ("on", "off"):
            print("用法: showpath on/off")
            return
        self.show_path = args[0].lower() == "on"
        self.redraw()
        print(f"路径显示：{'开启' if self.show_path else '关闭'}")

    def _cmd_save(self, args):
        if len(args) != 1:
            print("用法: save <文件>")
            return
        try:
            out = dump_session(args[0], self.points, self.path_density, self.show_path, self.speed_limits)
        except Exception as e:
            print(f"[错误] 保存失败: {e}")
            return
        print(f"[保存] 会话已保存: {out}")

    def _cmd_load(self, args):
        if len(args) != 1:
            print("用法: load <文件>")
            return
        try:
            payload = load_session(args[0])
        except Exception as e:
            print(f"[错误] 加载失败: {e}")
            return
        settings = payload.get("settings", {})
        self.points = payload.get("waypoints", [])
        self.path_density = float(settings.get("density", DEFAULT_PATH_DENSITY))
        self.show_path = bool(settings.get("showpath", True))
        loaded_limits = settings.get("speed_limits", SpeedLimits())
        if not isinstance(loaded_limits, SpeedLimits):
            loaded_limits = SpeedLimits()
        self.speed_limits = loaded_limits
        self.redraw()
        print(
            f"[加载] 会话已加载: {payload.get('path')}  (路径点数={len(self.points)}, "
            f"密度={self.path_density:.2f}, 显示路径={self.show_path}, "
            f"vmax={self.speed_limits.max_v:.3f}, amax={self.speed_limits.max_a:.3f}, "
            f"wmax={self.speed_limits.max_w:.3f}, awmax={self.speed_limits.max_aw:.3f})"
        )

    def _cmd_exportcpp(self, args):
        if not args:
            print("用法: exportcpp <文件> [name=PathName] [scale=1.0]")
            return
        out_file = args[0]
        path_name = "GeneratedPath"
        scale = 1.0
        for token in args[1:]:
            if "=" not in token:
                print(f"参数格式无效: {token}")
                return
            key, value = token.split("=", 1)
            key = key.lower().strip()
            if key == "name":
                path_name = value.strip() or "GeneratedPath"
            elif key == "scale":
                try:
                    scale = float(value)
                except ValueError:
                    print(f"scale 参数无效: {value}")
                    return
                if scale <= 0.0:
                    print("scale 必须 > 0")
                    return
            else:
                print(f"未知参数: {key}（可用: name, scale）")
                return

        self._rebuild_path()
        try:
            out = export_path_cpp(
                file_path=out_file,
                samples=self.path_samples,
                path_name=path_name,
                grid_scale=scale,
            )
        except Exception as e:
            print(f"[错误] 导出失败: {e}")
            return
        print(
            f"[导出] C++ 路径已生成: {out}  "
            f"(采样点={int(self.path_samples.x.size)}, name={path_name}, scale={scale:.6f})"
        )
