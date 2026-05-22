# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Arc

from app_config import (
    BACKGROUND_IMAGE_PATH,
    GRID_WIDTH,
    GRID_HEIGHT,
    BACKGROUND_ALPHA,
    DEFAULT_PATH_DENSITY,
)
from coord_utils import (
    grid_data_bounds,
    grid_to_data,
    data_to_grid,
    grid_vec_to_data_vec,
    format_coord_status,
)
from path_planner import Waypoint, PathSamples, build_path, dump_session, load_session


class GridCanvas:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("SimplePathPlanner")

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
            meta={"segments": 0, "total_length": 0.0, "sample_count": 0},
        )
        self.path_density = DEFAULT_PATH_DENSITY
        self.show_path = True

        self.coord_text = self.fig.text(
            0.01, 0.01, "", fontsize=9, va="bottom", ha="left",
            family="monospace", transform=self.fig.transFigure
        )
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._draw_coordinate_axes()
        self._rebuild_path()
        self._draw_path()
        self._draw_points()
        self._print_grid_info()
        self.fig.tight_layout()

    def _setup_view(self):
        self.ax.set_aspect("equal")
        self.ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.format_coord = lambda x, y: format_coord_status(x, y, self._has_image, self._img_w, self._img_h)

    def _apply_limits(self):
        if self._has_image:
            self.ax.set_xlim(0, self._img_w)
            self.ax.set_ylim(0, self._img_h)
        else:
            self.ax.set_xlim(0, GRID_WIDTH)
            self.ax.set_ylim(0, GRID_HEIGHT)

    def _load_background(self):
        if not BACKGROUND_IMAGE_PATH:
            return
        if not os.path.exists(BACKGROUND_IMAGE_PATH):
            print(f"[WARN] background image not found: {BACKGROUND_IMAGE_PATH}")
            return
        self._img = mpimg.imread(BACKGROUND_IMAGE_PATH)
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
        print(f"[INFO] loaded image: {self._img_w}x{self._img_h}  <-  {BACKGROUND_IMAGE_PATH}")

    def _draw_grid_lines(self):
        dx0, dy0, dx1, dy1 = grid_data_bounds(self._has_image, self._img_w, self._img_h)
        for gy in range(GRID_WIDTH + 1):
            dx, _ = grid_to_data(0, gy, self._has_image, self._img_w, self._img_h)
            self.ax.plot([dx, dx], [dy0, dy1], color="black", linewidth=0.5, zorder=2)
        for gx in range(GRID_HEIGHT + 1):
            _, dy = grid_to_data(gx, 0, self._has_image, self._img_w, self._img_h)
            self.ax.plot([dx0, dx1], [dy, dy], color="black", linewidth=0.5, zorder=2)

    def _draw_coordinate_axes(self):
        ox, oy = grid_to_data(0.0, 0.0, self._has_image, self._img_w, self._img_h)
        dx0, dy0, dx1, dy1 = grid_data_bounds(self._has_image, self._img_w, self._img_h)
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
        for idx, p in enumerate(self.points, start=1):
            dx, dy = grid_to_data(p.x, p.y, self._has_image, self._img_w, self._img_h)
            self.ax.plot(dx, dy, marker="o", markersize=6, color="red", zorder=5)

            hdx, hdy = grid_vec_to_data_vec(np.cos(p.theta), np.sin(p.theta), self._has_image, self._img_w, self._img_h)
            hnorm = np.hypot(hdx, hdy)
            if hnorm > 1e-9:
                self.ax.annotate("", xy=(dx + hdx * (42.0 / hnorm), dy + hdy * (42.0 / hnorm)), xytext=(dx, dy),
                                 arrowprops=dict(arrowstyle="->", color="limegreen", lw=2.0), zorder=6)

            if p.vx is not None and p.vy is not None:
                vdx, vdy = grid_vec_to_data_vec(float(p.vx), float(p.vy), self._has_image, self._img_w, self._img_h)
                vnorm = np.hypot(vdx, vdy)
                if vnorm > 1e-9:
                    self.ax.annotate("", xy=(dx + vdx * (35.0 / vnorm), dy + vdy * (35.0 / vnorm)), xytext=(dx, dy),
                                     arrowprops=dict(arrowstyle="->", color="magenta", lw=1.8), zorder=6)
            self.ax.text(dx + 3, dy + 3, f"P{idx} ({p.x:.1f}, {p.y:.1f}, {p.theta:.2f})",
                         color="red", fontsize=8, zorder=6)

    def _draw_path(self):
        if not self.show_path or self.path_samples.x.size < 2:
            return
        data_x, data_y = [], []
        for gx, gy in zip(self.path_samples.x, self.path_samples.y):
            dx, dy = grid_to_data(float(gx), float(gy), self._has_image, self._img_w, self._img_h)
            data_x.append(dx)
            data_y.append(dy)
        self.ax.plot(data_x, data_y, color="deepskyblue", linewidth=2.0, zorder=4)

    def _rebuild_path(self):
        self.path_samples = build_path(self.points, density=self.path_density)

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        dx, dy = event.xdata, event.ydata
        gx, gy = data_to_grid(dx, dy, self._has_image, self._img_w, self._img_h)
        parts = []
        if gx is not None and gy is not None and 0 <= gx <= GRID_HEIGHT and 0 <= gy <= GRID_WIDTH:
            parts.append(f"Grid: ({gx:.1f}, {gy:.1f})")
        if self._has_image:
            parts.append(f"Data: ({dx:.1f}, {dy:.1f})")
            parts.append(f"Image: {self._img_w}x{self._img_h}")
        else:
            parts.append(f"Pos: ({dx:.1f}, {dy:.1f})")
        self.coord_text.set_text("  |  ".join(parts))

    def _print_grid_info(self):
        if not self._has_image:
            print(f"[INFO] no image, grid maps directly: {GRID_WIDTH}x{GRID_HEIGHT}")
            return
        dx0, dy0, dx1, dy1 = grid_data_bounds(self._has_image, self._img_w, self._img_h)
        print(f"[INFO] grid in data coords: x=[{dx0:.1f}, {dx1:.1f}]  y=[{dy0:.1f}, {dy1:.1f}]  "
              f"|  size={dx1-dx0:.0f}x{dy1-dy0:.0f}")

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
        print("\n========== SimplePathPlanner ==========")
        self._print_help()
        while self._running:
            try:
                cmd = input("\n>>> ").strip().split()
                if cmd:
                    self._handle_command(cmd)
            except (EOFError, KeyboardInterrupt):
                break

    def _print_help(self):
        print("Commands:")
        print("  help      Show this help")
        print("  exit/q    Exit the program")
        print("  grid      Redraw the grid")
        print("  addpoint x, y, theta[, vx, vy, vw]   Add a point in grid coords")
        print(f"           range: x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]")
        print("  plan      Rebuild path and print path summary")
        print("  density d Set path sampling density (d >= 1.0)")
        print("  showpath on/off   Toggle path curve visibility")
        print("  save <file>   Save current waypoints and settings to JSON")
        print("  load <file>   Load JSON and replace current waypoints/settings")

    def _handle_command(self, cmd):
        op = cmd[0].lower()
        if op in ("exit", "q"):
            self._running = False
            plt.close("all")
            print("Bye.")
        elif op == "help":
            self._print_help()
        elif op == "grid":
            self.redraw()
            print("Grid redrawn.")
        elif op == "addpoint":
            self._cmd_addpoint(cmd[1:])
        elif op == "plan":
            self._cmd_plan()
        elif op == "density":
            self._cmd_density(cmd[1:])
        elif op == "showpath":
            self._cmd_showpath(cmd[1:])
        elif op == "save":
            self._cmd_save(cmd[1:])
        elif op == "load":
            self._cmd_load(cmd[1:])
        else:
            print(f"Unknown command: {op}. Type 'help' for available commands.")

    def _cmd_addpoint(self, args):
        if not args:
            print("Usage: addpoint x, y, theta")
            return
        parts = " ".join(args).replace(" ", "").split(",")
        if len(parts) not in (3, 6):
            print("Usage: addpoint x, y, theta[, vx, vy, vw]")
            return
        try:
            nums = list(map(float, parts))
        except ValueError:
            print("Invalid number format. Example: addpoint 1.0, 1.0, 1.57, 0.5, 0.0, 0.2")
            return
        gx, gy, theta = nums[0], nums[1], nums[2]
        vx = vy = vw = None
        if len(nums) == 6:
            vx, vy, vw = nums[3], nums[4], nums[5]
        if not (0.0 <= gx <= GRID_HEIGHT and 0.0 <= gy <= GRID_WIDTH):
            print(f"Point out of grid range. x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]")
            return
        self.points.append(Waypoint(x=gx, y=gy, theta=theta, vx=vx, vy=vy, vw=vw))
        self.redraw()
        if vx is None:
            print(f"Point added: ({gx:.3f}, {gy:.3f}, {theta:.3f})")
        else:
            print(f"Point added: ({gx:.3f}, {gy:.3f}, {theta:.3f}, vx={vx:.3f}, vy={vy:.3f}, vw={vw:.3f})")

    def _cmd_plan(self):
        self.redraw()
        meta = self.path_samples.meta
        print(f"[PLAN] segments={meta.get('segments', 0)}  samples={meta.get('sample_count', 0)}  "
              f"length={meta.get('total_length', 0.0):.3f}")

    def _cmd_density(self, args):
        if len(args) != 1:
            print("Usage: density <float>")
            return
        try:
            d = float(args[0])
        except ValueError:
            print("Invalid number. Example: density 20")
            return
        if d < 1.0:
            print("Density must be >= 1.0")
            return
        self.path_density = d
        self.redraw()
        print(f"Density set: {self.path_density:.2f}")

    def _cmd_showpath(self, args):
        if len(args) != 1 or args[0].lower() not in ("on", "off"):
            print("Usage: showpath on/off")
            return
        self.show_path = args[0].lower() == "on"
        self.redraw()
        print(f"Show path: {'on' if self.show_path else 'off'}")

    def _cmd_save(self, args):
        if len(args) != 1:
            print("Usage: save <file>")
            return
        try:
            out = dump_session(args[0], self.points, self.path_density, self.show_path)
        except Exception as e:
            print(f"[ERROR] save failed: {e}")
            return
        print(f"[SAVE] session saved: {out}")

    def _cmd_load(self, args):
        if len(args) != 1:
            print("Usage: load <file>")
            return
        try:
            payload = load_session(args[0])
        except Exception as e:
            print(f"[ERROR] load failed: {e}")
            return
        settings = payload.get("settings", {})
        self.points = payload.get("waypoints", [])
        self.path_density = float(settings.get("density", DEFAULT_PATH_DENSITY))
        self.show_path = bool(settings.get("showpath", True))
        self.redraw()
        print(f"[LOAD] session loaded: {payload.get('path')}  (points={len(self.points)}, "
              f"density={self.path_density:.2f}, showpath={self.show_path})")

