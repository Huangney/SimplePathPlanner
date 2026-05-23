from __future__ import annotations

import os
import numpy as np
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Polygon

from app_config import GRID_HEIGHT, GRID_WIDTH
from coord_utils import (
    format_coord_status,
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

        label = (
            f"P{point_idx + 1}\n"
            f"({p.x:.2f}, {p.y:.2f}, {p.theta:.3f})\n"
            f"({0.0 if p.vx is None else float(p.vx):.3f}, "
            f"{0.0 if p.vy is None else float(p.vy):.3f}, "
            f"{0.0 if p.vw is None else float(p.vw):.3f})"
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

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        bx0, by0, bx1, by1 = self._grid_bounds_tuple()
        point_hover_idx = None
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
                gx_i = float(self.path_samples.x[nearest_idx])
                gy_i = float(self.path_samples.y[nearest_idx])
                theta_i = float(self.path_samples.theta[nearest_idx])
                xdot_i = float(self.path_samples.xdot[nearest_idx]) if self.path_samples.xdot.size > nearest_idx else 0.0
                ydot_i = float(self.path_samples.ydot[nearest_idx]) if self.path_samples.ydot.size > nearest_idx else 0.0
                w_i = float(self.path_samples.w[nearest_idx]) if self.path_samples.w.size > nearest_idx else 0.0
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
        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._draw_coordinate_axes()
        self._rebuild_path()
        self._draw_path()
        self._draw_points()
        self.fig.canvas.draw_idle()
