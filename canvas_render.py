from __future__ import annotations

import os
from bisect import bisect_right
import numpy as np
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Polygon
import tkinter as tk
from tkinter import messagebox, ttk

from app_config import GRID_HEIGHT, GRID_WIDTH
from coord_utils import (
    format_coord_status,
    data_to_grid,
    grid_data_bounds,
    grid_to_data,
    grid_vec_to_data_vec,
)
from path_planner import build_path


class CanvasRenderMixin:
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
            alpha=self.background_alpha,
            zorder=0,
        )
        artist.format_cursor_data = lambda data: ""
        print(f"[信息] 已加载背景图片：{self._img_w}x{self._img_h}  <-  {background_image_path}")

    def _zoom_to_rect_equivalent(self, x0: float, y0: float, x1: float, y1: float):
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
            self.ax.text(dx + 3, dy + 3, f"P{idx}", color="red", fontsize=8, zorder=6)

    def _clear_waypoint_hover_visuals(self):
        if self._hover_heading_arrow is not None:
            self._hover_heading_arrow.remove()
            self._hover_heading_arrow = None
        if self._hover_velocity_arrow is not None:
            self._hover_velocity_arrow.remove()
            self._hover_velocity_arrow = None
        if self._hover_body_patch is not None:
            self._hover_body_patch.remove()
            self._hover_body_patch = None

    @staticmethod
    def _format_optional_value(value):
        return "" if value is None else f"{float(value):.6f}"

    @staticmethod
    def _parse_optional_float(text: str):
        stripped = str(text).strip()
        if stripped == "":
            return None
        return float(stripped)

    def _nearest_waypoint_idx_from_pixel(self, x_px: float, y_px: float, threshold_px: float = 12.0):
        if not self.points:
            return None
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        point_data = np.array(
            [
                grid_to_data(p.x, p.y, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
                for p in self.points
            ],
            dtype=float,
        )
        point_pixels = self.ax.transData.transform(point_data)
        point_dist2 = (point_pixels[:, 0] - float(x_px)) ** 2 + (point_pixels[:, 1] - float(y_px)) ** 2
        nearest_idx = int(np.argmin(point_dist2))
        if float(np.sqrt(point_dist2[nearest_idx])) <= float(threshold_px):
            return nearest_idx
        return None

    def _open_waypoint_edit_dialog(self, point_idx: int):
        if point_idx < 0 or point_idx >= len(self.points):
            return

        parent = getattr(self.fig.canvas.manager, "window", None)
        if parent is None:
            print("[警告] 当前图形后端不支持弹窗编辑。")
            return

        p = self.points[point_idx]
        top = tk.Toplevel(parent)
        top.title(f"编辑路径点 P{point_idx + 1}")
        top.resizable(False, False)
        top.transient(parent)
        top.grab_set()

        content = ttk.Frame(top, padding=12)
        content.grid(row=0, column=0, sticky="nsew")

        fields = {
            "x": tk.StringVar(value=self._format_optional_value(p.x)),
            "y": tk.StringVar(value=self._format_optional_value(p.y)),
            "theta": tk.StringVar(value=self._format_optional_value(p.theta)),
            "vx": tk.StringVar(value=self._format_optional_value(p.vx)),
            "vy": tk.StringVar(value=self._format_optional_value(p.vy)),
            "w": tk.StringVar(value=self._format_optional_value(p.vw)),
            "velo": tk.StringVar(value=self._format_optional_value(p.speed)),
        }

        def add_row(row: int, items: list[tuple[str, str]]):
            col = 0
            for label_text, key in items:
                ttk.Label(content, text=label_text).grid(row=row, column=col, padx=(0, 6), pady=4, sticky="e")
                ttk.Entry(content, width=12, textvariable=fields[key]).grid(row=row, column=col + 1, padx=(0, 12), pady=4)
                col += 2

        add_row(0, [("x（必填）", "x"), ("y（必填）", "y"), ("theta（必填）", "theta")])
        add_row(1, [("vx", "vx"), ("vy(方向)", "vy")])
        add_row(2, [("w", "w"), ("velo", "velo")])

        hint = ttk.Label(content, text="留空可清空 vx/vy/w/velo；x、y、theta 为必填项")
        hint.grid(row=3, column=0, columnspan=6, pady=(8, 4), sticky="w")

        button_bar = ttk.Frame(content)
        button_bar.grid(row=4, column=0, columnspan=6, pady=(8, 0), sticky="e")

        def on_cancel():
            top.grab_release()
            top.destroy()

        def on_ok():
            try:
                x = float(fields["x"].get().strip())
                y = float(fields["y"].get().strip())
                theta = float(fields["theta"].get().strip())
                vx = self._parse_optional_float(fields["vx"].get())
                vy = self._parse_optional_float(fields["vy"].get())
                w = self._parse_optional_float(fields["w"].get())
                velo = self._parse_optional_float(fields["velo"].get())
            except ValueError:
                messagebox.showerror("输入错误", "x, y, theta, vx, vy, w, velo 的数值格式无效。", parent=top)
                return

            if not (0.0 <= x <= GRID_HEIGHT and 0.0 <= y <= GRID_WIDTH):
                messagebox.showerror(
                    "范围错误",
                    f"x 必须在 [0,{GRID_HEIGHT}]，y 必须在 [0,{GRID_WIDTH}]。",
                    parent=top,
                )
                return

            if (vx is None) != (vy is None):
                messagebox.showerror("输入错误", "vx 和 vy 需要同时填写或同时留空。", parent=top)
                return

            p.x = x
            p.y = y
            p.theta = theta
            p.vx = vx
            p.vy = vy
            p.vw = w
            p.speed = velo
            self.redraw()
            print(
                f"路径点 P{point_idx + 1} 已更新："
                f"({p.x:.3f}, {p.y:.3f}, {p.theta:.3f}), "
                f"vx={p.vx}, vy={p.vy}, w={p.vw}, velo={p.speed}"
            )
            top.grab_release()
            top.destroy()

        ttk.Button(button_bar, text="取消", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_bar, text="确定", command=on_ok).grid(row=0, column=1)

        def on_close():
            on_cancel()

        top.protocol("WM_DELETE_WINDOW", on_close)
        top.update_idletasks()
        try:
            parent.update_idletasks()
            pw = int(parent.winfo_width())
            ph = int(parent.winfo_height())
            px = int(parent.winfo_rootx())
            py = int(parent.winfo_rooty())
            ww = int(top.winfo_reqwidth())
            wh = int(top.winfo_reqheight())
            if pw > 1 and ph > 1:
                x = px + max(0, (pw - ww) // 2)
                y = py + max(0, (ph - wh) // 2)
            else:
                sw = int(top.winfo_screenwidth())
                sh = int(top.winfo_screenheight())
                x = max(0, (sw - ww) // 2)
                y = max(0, (sh - wh) // 2)
            top.geometry(f"{ww}x{wh}+{x}+{y}")
        except tk.TclError:
            pass
        try:
            top.lift()
            top.focus_force()
        except tk.TclError:
            pass

    def _on_button_press(self, event):
        if not getattr(event, "dblclick", False):
            return
        if event.inaxes != self.ax:
            return
        if getattr(event, "button", None) != 1:
            return

        point_idx = self._nearest_waypoint_idx_from_pixel(event.x, event.y)
        if point_idx is None:
            return
        self._open_waypoint_edit_dialog(point_idx)

    def _draw_hover_body(self, gx: float, gy: float, theta: float):
        length = self.body_length
        width = self.body_width
        if length is None or width is None or length <= 0.0 or width <= 0.0:
            if self._hover_body_patch is not None:
                self._hover_body_patch.remove()
                self._hover_body_patch = None
            return

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        half_l = 0.5 * float(length)
        half_w = 0.5 * float(width)
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()

        offsets = (
            (+half_l, +half_w),
            (+half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, +half_w),
        )
        corners = []
        for dl, dw in offsets:
            cx = gx + dl * c - dw * s
            cy = gy + dl * s + dw * c
            dx, dy = grid_to_data(cx, cy, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            corners.append((dx, dy))

        if self._hover_body_patch is None:
            self._hover_body_patch = Polygon(
                corners,
                closed=True,
                facecolor=(1.0, 0.95, 0.2, 0.20),
                edgecolor="goldenrod",
                linewidth=1.6,
                zorder=8,
            )
            self.ax.add_patch(self._hover_body_patch)
        else:
            self._hover_body_patch.set_xy(corners)

    def _show_waypoint_hover(self, point_idx: int):
        if point_idx < 0 or point_idx >= len(self.points):
            return
        self._clear_waypoint_hover_visuals()
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        p = self.points[point_idx]
        hx, hy = grid_to_data(p.x, p.y, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        self._draw_hover_body(p.x, p.y, p.theta)

        hdx, hdy = grid_vec_to_data_vec(np.cos(p.theta), np.sin(p.theta), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        hnorm = np.hypot(hdx, hdy)
        if hnorm > 1e-9:
            self._hover_heading_arrow = self.ax.annotate(
                "",
                xy=(hx + hdx * (21.0 / hnorm), hy + hdy * (21.0 / hnorm)),
                xytext=(hx, hy),
                arrowprops=dict(arrowstyle="->", color="limegreen", lw=2.0),
                zorder=6,
            )

        if p.vx is not None or p.vy is not None:
            vx = 0.0 if p.vx is None else float(p.vx)
            vy = 0.0 if p.vy is None else float(p.vy)
            vdx, vdy = grid_vec_to_data_vec(vx, vy, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            vnorm = np.hypot(vdx, vdy)
            if vnorm > 1e-9:
                self._hover_velocity_arrow = self.ax.annotate(
                    "",
                    xy=(hx + vdx * (17.5 / vnorm), hy + vdy * (17.5 / vnorm)),
                    xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="->", color="magenta", lw=1.8),
                    zorder=6,
                )

        label_parts: list[str] = []
        if p.vx is not None or p.vy is not None:
            dvx = 0.0 if p.vx is None else float(p.vx)
            dvy = 0.0 if p.vy is None else float(p.vy)
            label_parts.append(f"dir=({dvx:.3f},{dvy:.3f})")
        if p.speed is not None:
            label_parts.append(f"spd={float(p.speed):.3f}")
        if p.vw is not None:
            label_parts.append(f"vw={float(p.vw):.3f}")
        constraint_line = "  ".join(label_parts) if label_parts else ""

        label = (
            f"P{point_idx + 1}\n"
            f"({p.x:.2f}, {p.y:.2f}, {p.theta:.3f})"
        )
        if constraint_line:
            label += f"\n{constraint_line}"
        if self._hover_text is None:
            self._hover_text = self.ax.annotate(
                label,
                xy=(hx, hy),
                xytext=(12, 10),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="goldenrod", alpha=0.92),
                zorder=10,
            )
        else:
            self._hover_text.xy = (hx, hy)
            self._hover_text.set_text(label)
            self._hover_text.set_visible(True)
        self._hover_text_mode = "point"

    def _draw_path(self):
        if not self.show_path or self.path_samples.x.size < 2:
            return
        if self._path_data_x.size < 2 or self._path_data_y.size < 2:
            return
        pts = np.column_stack([self._path_data_x, self._path_data_y])
        segments = np.stack([pts[:-1], pts[1:]], axis=1)

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

    def _refresh_path_data_cache(self):
        if self.path_samples.x.size == 0:
            self._path_data_x = np.array([], dtype=float)
            self._path_data_y = np.array([], dtype=float)
            return
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        data_x = np.empty(self.path_samples.x.size, dtype=float)
        data_y = np.empty(self.path_samples.y.size, dtype=float)
        for i, (gx, gy) in enumerate(zip(self.path_samples.x, self.path_samples.y)):
            dx, dy = grid_to_data(float(gx), float(gy), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            data_x[i] = dx
            data_y[i] = dy
        self._path_data_x = data_x
        self._path_data_y = data_y

    def _rebuild_path(self):
        self.path_samples = build_path(
            self.points,
            density=self.path_density,
            speed_limits=self.speed_limits,
            solver=self.solver,
        )
        self._refresh_path_data_cache()

    def _path_insert_index_from_sample(self, sample_idx: int) -> int:
        waypoint_sample_indices = self.path_samples.meta.get("waypoint_sample_indices", None)
        if not waypoint_sample_indices:
            return len(self.points)
        try:
            pos = bisect_right([int(i) for i in waypoint_sample_indices], int(sample_idx)) - 1
        except TypeError:
            return len(self.points)
        pos = max(0, min(pos, len(self.points) - 1))
        return min(len(self.points), pos + 1)

    def _project_point_to_path(self, gx: float, gy: float):
        if self.path_samples.x.size < 2 or self.path_samples.y.size < 2:
            return None

        xs = np.asarray(self.path_samples.x, dtype=float)
        ys = np.asarray(self.path_samples.y, dtype=float)
        ths = np.asarray(self.path_samples.theta, dtype=float)
        ts = np.asarray(self.path_samples.t, dtype=float) if self.path_samples.t.size == xs.size else None
        v_lin = np.asarray(self.path_samples.v_lin, dtype=float) if self.path_samples.v_lin.size == xs.size else None
        xdot = np.asarray(self.path_samples.xdot, dtype=float) if self.path_samples.xdot.size == xs.size else None
        ydot = np.asarray(self.path_samples.ydot, dtype=float) if self.path_samples.ydot.size == xs.size else None
        w = np.asarray(self.path_samples.w, dtype=float) if self.path_samples.w.size == xs.size else None

        px = float(gx)
        py = float(gy)
        best = None
        best_dist2 = float("inf")

        waypoint_sample_indices = self.path_samples.meta.get("waypoint_sample_indices", None)
        waypoint_sample_indices = [int(i) for i in waypoint_sample_indices] if waypoint_sample_indices else []

        for i in range(xs.size - 1):
            x0 = float(xs[i])
            y0 = float(ys[i])
            x1 = float(xs[i + 1])
            y1 = float(ys[i + 1])
            dx = x1 - x0
            dy = y1 - y0
            denom = dx * dx + dy * dy
            if denom <= 1e-12:
                alpha = 0.0
            else:
                alpha = ((px - x0) * dx + (py - y0) * dy) / denom
                alpha = float(min(1.0, max(0.0, alpha)))

            qx = x0 + alpha * dx
            qy = y0 + alpha * dy
            dist2 = (px - qx) ** 2 + (py - qy) ** 2
            if dist2 >= best_dist2:
                continue

            theta0 = float(ths[i])
            theta1 = float(ths[i + 1])
            dtheta = ((theta1 - theta0) + np.pi) % (2.0 * np.pi) - np.pi
            theta_q = theta0 + alpha * dtheta

            t_q = None
            if ts is not None:
                t_q = float(ts[i] + alpha * (ts[i + 1] - ts[i]))

            v_q = None
            if v_lin is not None:
                v_q = float(v_lin[i] + alpha * (v_lin[i + 1] - v_lin[i]))

            xdot_q = None
            if xdot is not None:
                xdot_q = float(xdot[i] + alpha * (xdot[i + 1] - xdot[i]))

            ydot_q = None
            if ydot is not None:
                ydot_q = float(ydot[i] + alpha * (ydot[i + 1] - ydot[i]))

            w_q = None
            if w is not None:
                w_q = float(w[i] + alpha * (w[i + 1] - w[i]))

            if waypoint_sample_indices:
                seg_idx = bisect_right(waypoint_sample_indices, i) - 1
                seg_idx = max(0, min(seg_idx, len(self.points) - 2))
                insert_idx = seg_idx + 1
            else:
                insert_idx = len(self.points)

            best_dist2 = dist2
            best = {
                "x": float(qx),
                "y": float(qy),
                "theta": float((theta_q + np.pi) % (2.0 * np.pi) - np.pi),
                "t": t_q,
                "v_lin": v_q,
                "xdot": xdot_q,
                "ydot": ydot_q,
                "w": w_q,
                "segment_sample_idx": i,
                "alpha": alpha,
                "insert_idx": insert_idx,
                "dist2": dist2,
            }

        return best

    def _on_key_press(self, event):
        key = str(getattr(event, "key", "") or "").lower()
        if key not in ("a", "i"):
            return
        if event.inaxes != self.ax:
            return

        if event.xdata is not None and event.ydata is not None and event.inaxes == self.ax:
            bx0, by0, bx1, by1 = self._grid_bounds_tuple()
            gx, gy = data_to_grid(float(event.xdata), float(event.ydata), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
            if gx is None or gy is None:
                return
            self._last_mouse_grid_xy = (float(gx), float(gy))
        elif self._last_mouse_grid_xy is not None:
            gx, gy = self._last_mouse_grid_xy
        else:
            return

        hover_theta = 0.0
        insert_idx = len(self.points)

        if key == "i":
            if self._hover_path_sample_idx is None:
                print("请先将鼠标悬停在路径上，再按 i。")
                return
            proj = self._project_point_to_path(float(gx), float(gy))
            if proj is None:
                print("当前没有可插入的路径，请先执行 plan。")
                return
            insert_idx = int(proj["insert_idx"])
            gx = float(proj["x"])
            gy = float(proj["y"])
            hover_theta = float(proj["theta"])
            new_idx = self._insert_waypoint_at(insert_idx, gx, gy, hover_theta)
            print(
                f"已将鼠标在路径上的投影点插入为关键点："
                f"P{new_idx} = ({gx:.3f}, {gy:.3f}, {hover_theta:.3f})"
            )
            return

        if self._hover_waypoint_idx is not None and 0 <= self._hover_waypoint_idx < len(self.points):
            hover_point = self.points[self._hover_waypoint_idx]
            hover_theta = float(hover_point.theta)
            insert_idx = self._hover_waypoint_idx + 1
        elif self._hover_path_sample_idx is not None and self.path_samples.theta.size > self._hover_path_sample_idx:
            hover_theta = float(self.path_samples.theta[int(self._hover_path_sample_idx)])
            insert_idx = self._path_insert_index_from_sample(int(self._hover_path_sample_idx))

        new_idx = self._insert_waypoint_at(insert_idx, float(gx), float(gy), hover_theta)
        print(f"已在鼠标位置新增点：P{new_idx} = ({float(gx):.3f}, {float(gy):.3f}, {hover_theta:.3f})")

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        gx, gy = data_to_grid(float(event.xdata), float(event.ydata), self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
        self._last_mouse_grid_xy = None if gx is None or gy is None else (float(gx), float(gy))
        point_hover_idx = None
        self._hover_path_sample_idx = None
        if self.points:
            point_data = np.array(
                [
                    grid_to_data(p.x, p.y, self._has_image, self._img_w, self._img_h, bx0, by0, bx1, by1)
                    for p in self.points
                ],
                dtype=float,
            )
            point_pixels = self.ax.transData.transform(point_data)
            point_dist2 = (point_pixels[:, 0] - float(event.x)) ** 2 + (point_pixels[:, 1] - float(event.y)) ** 2
            nearest_point_idx = int(np.argmin(point_dist2))
            if float(np.sqrt(point_dist2[nearest_point_idx])) <= 12.0:
                point_hover_idx = nearest_point_idx

        if point_hover_idx != self._hover_waypoint_idx:
            self._hover_waypoint_idx = point_hover_idx
            if self._hover_waypoint_idx is None:
                self._clear_waypoint_hover_visuals()
                if self._hover_text_mode == "point" and self._hover_text is not None:
                    self._hover_text.set_visible(False)
                    self._hover_text_mode = None
            else:
                if self._hover_marker is not None and self._hover_marker.get_visible():
                    self._hover_marker.set_visible(False)
                self._show_waypoint_hover(int(self._hover_waypoint_idx))
            self.fig.canvas.draw_idle()

        hover_hit = False
        if self._hover_waypoint_idx is not None:
            hover_hit = True
        elif self.show_path and self.path_samples.x.size > 0 and self._path_data_x.size == self.path_samples.x.size:
            path_data = np.column_stack([self._path_data_x, self._path_data_y])
            path_pixels = self.ax.transData.transform(path_data)
            dist2 = (path_pixels[:, 0] - float(event.x)) ** 2 + (path_pixels[:, 1] - float(event.y)) ** 2
            nearest_idx = int(np.argmin(dist2))
            nearest_px = float(np.sqrt(dist2[nearest_idx]))
            hover_threshold_px = 12.0
            if nearest_px <= hover_threshold_px:
                hover_hit = True
                self._hover_path_sample_idx = nearest_idx
                gx_i = float(self.path_samples.x[nearest_idx])
                gy_i = float(self.path_samples.y[nearest_idx])
                theta_i = float(self.path_samples.theta[nearest_idx])
                xdot_i = float(self.path_samples.xdot[nearest_idx]) if self.path_samples.xdot.size > nearest_idx else 0.0
                ydot_i = float(self.path_samples.ydot[nearest_idx]) if self.path_samples.ydot.size > nearest_idx else 0.0
                w_i = float(self.path_samples.w[nearest_idx]) if self.path_samples.w.size > nearest_idx else 0.0
                v_lin_i = float(self.path_samples.v_lin[nearest_idx]) if self.path_samples.v_lin.size > nearest_idx else 0.0
                t_i = float(self.path_samples.t[nearest_idx]) if self.path_samples.t.size > nearest_idx else 0.0
                hx = self._path_data_x[nearest_idx]
                hy = self._path_data_y[nearest_idx]
                self._draw_hover_body(gx_i, gy_i, theta_i)
                if self._hover_marker is None:
                    marker, = self.ax.plot([hx], [hy], marker="o", markersize=7, markerfacecolor="none",
                                           markeredgecolor="gold", markeredgewidth=1.6, zorder=9)
                    self._hover_marker = marker
                else:
                    self._hover_marker.set_data([hx], [hy])
                    self._hover_marker.set_visible(True)

                label = (
                    f"Path[{nearest_idx}]\n"
                    f"({gx_i:.2f}, {gy_i:.2f}, {theta_i:.3f})\n"
                    f"|v|={v_lin_i:.3f}  t={t_i:.3f}\n"
                    f"({xdot_i:.3f}, {ydot_i:.3f}, {w_i:.3f})"
                )
                if self._hover_text is None:
                    self._hover_text = self.ax.annotate(
                        label,
                        xy=(hx, hy),
                        xytext=(12, 10),
                        textcoords="offset points",
                        fontsize=8,
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="goldenrod", alpha=0.92),
                        zorder=10,
                    )
                else:
                    self._hover_text.xy = (hx, hy)
                    self._hover_text.set_text(label)
                    self._hover_text.set_visible(True)
                self._hover_text_mode = "path"
                self.fig.canvas.draw_idle()

        if (not hover_hit) and self._hover_marker is not None and self._hover_marker.get_visible():
            self._hover_marker.set_visible(False)
            if self._hover_body_patch is not None:
                self._hover_body_patch.remove()
                self._hover_body_patch = None
            if self._hover_text is not None and self._hover_text_mode == "path":
                self._hover_text.set_visible(False)
                self._hover_text_mode = None
            self.fig.canvas.draw_idle()
        elif (not hover_hit) and self._hover_text is not None and self._hover_text.get_visible() and self._hover_text_mode == "path":
            if self._hover_body_patch is not None:
                self._hover_body_patch.remove()
                self._hover_body_patch = None
            self._hover_text.set_visible(False)
            self._hover_text_mode = None
            self.fig.canvas.draw_idle()

        self.coord_text.set_text("")

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
        self._hover_marker = None
        self._hover_text = None
        self._hover_text_mode = None
        self._hover_heading_arrow = None
        self._hover_velocity_arrow = None
        self._hover_body_patch = None
        self._hover_waypoint_idx = None
        self._hover_path_sample_idx = None
        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._draw_coordinate_axes()
        self._rebuild_path()
        self._draw_path()
        self._draw_points()
        self.fig.canvas.draw_idle()
