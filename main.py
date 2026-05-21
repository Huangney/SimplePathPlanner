import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
import os

BACKGROUND_IMAGE_PATH = "bkgrd.png"

GRID_WIDTH = 6
GRID_HEIGHT = 12

# 网格在数据坐标系中的位置（与工具栏坐标一致：左下角为原点，y 向上增大）
# 四个值全为 0 时退化为整张图=网格区域
GRID_X0 = 2    # 网格左边界
GRID_Y0 = 796    # 网格下边界（较小的 y）
GRID_X1 = 372    # 网格右边界
GRID_Y1 = 54    # 网格上边界（较大的 y）


class GridCanvas:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("SimplePathPlanner")
        self._has_image = False
        self._img_w = 0
        self._img_h = 0
        self._running = True
        self.points = []  # (gx, gy, theta)

        self.coord_text = self.fig.text(
            0.01, 0.01, "", fontsize=9, va="bottom", ha="left",
            family="monospace", transform=self.fig.transFigure)

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._print_grid_info()
        self.fig.tight_layout()

    def _setup_view(self):
        self.ax.set_aspect("equal")
        self.ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax.format_coord = self._format_coord_status

    def _format_coord_status(self, x, y):
        # 右下角状态栏：同时显示像素坐标（图像索引）和数据坐标（数学坐标）
        data_part = f"Data: ({x:.1f}, {y:.1f})"
        if not self._has_image:
            return data_part

        px = int(np.floor(x))
        py = int(np.floor(self._img_h - 1 - y))
        if 0 <= px < self._img_w and 0 <= py < self._img_h:
            pixel_part = f"Pixel: ({px}, {py})"
        else:
            pixel_part = "Pixel: (out)"
        return f"{pixel_part}  |  {data_part}"

    def _apply_limits(self):
        if self._has_image:
            self.ax.set_xlim(0, self._img_w)
            self.ax.set_ylim(0, self._img_h)
        else:
            self.ax.set_xlim(0, GRID_WIDTH)
            self.ax.set_ylim(0, GRID_HEIGHT)

    def _grid_data_bounds(self):
        """返回网格在数据坐标系中的 (x_left, y_bottom, x_right, y_top)"""
        if not self._has_image:
            return 0.0, 0.0, float(GRID_WIDTH), float(GRID_HEIGHT)
        if GRID_X0 == 0 and GRID_Y0 == 0 and GRID_X1 == 0 and GRID_Y1 == 0:
            return 0.0, 0.0, float(self._img_w), float(self._img_h)
        return float(GRID_X0), float(GRID_Y0), float(GRID_X1), float(GRID_Y1)

    def _load_background(self):
        if not BACKGROUND_IMAGE_PATH:
            return
        if not os.path.exists(BACKGROUND_IMAGE_PATH):
            print(f"[WARN] background image not found: {BACKGROUND_IMAGE_PATH}")
            return
        self._img = mpimg.imread(BACKGROUND_IMAGE_PATH)
        self._img_h, self._img_w = self._img.shape[:2]
        self._has_image = True
        self._img_artist = self.ax.imshow(
            self._img,
            extent=[0, self._img_w, 0, self._img_h],
            origin="upper",
            aspect="equal",
            zorder=0,
        )
        # 禁用 Matplotlib 默认的像素颜色显示，避免覆盖我们自定义坐标文本
        self._img_artist.format_cursor_data = lambda data: ""
        print(f"[INFO] loaded image: {self._img_w}x{self._img_h}  <-  {BACKGROUND_IMAGE_PATH}")

    def _draw_grid_lines(self):
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()

        for i in range(GRID_WIDTH + 1):
            dx, _ = self._grid_to_data(i, 0)
            self.ax.plot([dx, dx], [dy0, dy1], color="black", linewidth=0.5, zorder=2)

        for j in range(GRID_HEIGHT + 1):
            _, dy = self._grid_to_data(0, j)
            self.ax.plot([dx0, dx1], [dy, dy], color="black", linewidth=0.5, zorder=2)

    def _draw_points(self):
        for idx, (gx, gy, theta) in enumerate(self.points, start=1):
            dx, dy = self._grid_to_data(gx, gy)
            self.ax.plot(dx, dy, marker="o", markersize=6, color="red", zorder=5)
            self.ax.text(
                dx + 3, dy + 3, f"P{idx} ({gx:.1f}, {gy:.1f}, {theta:.2f})",
                color="red", fontsize=8, zorder=6
            )

    def _grid_to_data(self, gx, gy):
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        dx = dx0 + gx / GRID_WIDTH * (dx1 - dx0)
        dy = dy0 + gy / GRID_HEIGHT * (dy1 - dy0)
        return dx, dy

    def _data_to_grid(self, dx, dy):
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        if dx1 == dx0 or dy1 == dy0:
            return None, None
        gx = (dx - dx0) / (dx1 - dx0) * GRID_WIDTH
        gy = (dy - dy0) / (dy1 - dy0) * GRID_HEIGHT
        return gx, gy

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        dx, dy = event.xdata, event.ydata
        gx, gy = self._data_to_grid(dx, dy)

        parts = []
        if gx is not None and gy is not None and 0 <= gx <= GRID_WIDTH and 0 <= gy <= GRID_HEIGHT:
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
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        print(f"[INFO] grid in data coords: "
              f"x=[{dx0:.1f}, {dx1:.1f}]  y=[{dy0:.1f}, {dy1:.1f}]  "
              f"|  size={dx1-dx0:.0f}x{dy1-dy0:.0f}")

    def redraw(self):
        self.ax.cla()
        self._setup_view()
        self._load_background()
        self._apply_limits()
        self._draw_grid_lines()
        self._draw_points()
        self.fig.canvas.draw_idle()

    def start_terminal_loop(self):
        print("\n========== SimplePathPlanner ==========")
        self._print_help()
        while self._running:
            try:
                cmd = input("\n>>> ").strip().split()
                if not cmd:
                    continue
                self._handle_command(cmd)
            except (EOFError, KeyboardInterrupt):
                break

    def _print_help(self):
        print("Commands:")
        print("  help      Show this help")
        print("  exit/q    Exit the program")
        print("  grid      Redraw the grid")
        print("  addpoint x, y, theta   Add a point in grid coords and draw it")

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
        else:
            print(f"Unknown command: {op}. Type 'help' for available commands.")

    def _cmd_addpoint(self, args):
        if not args:
            print("Usage: addpoint x, y, theta")
            return
        raw = " ".join(args).replace(" ", "")
        parts = raw.split(",")
        if len(parts) != 3:
            print("Usage: addpoint x, y, theta")
            return
        try:
            gx, gy, theta = map(float, parts)
        except ValueError:
            print("Invalid number format. Example: addpoint 1.0, 1.0, 1.57")
            return
        if not (0.0 <= gx <= GRID_WIDTH and 0.0 <= gy <= GRID_HEIGHT):
            print(f"Point out of grid range. x in [0,{GRID_WIDTH}], y in [0,{GRID_HEIGHT}]")
            return
        self.points.append((gx, gy, theta))
        self.redraw()
        print(f"Point added: ({gx:.3f}, {gy:.3f}, {theta:.3f})")


def main():
    app = GridCanvas()
    terminal_thread = threading.Thread(target=app.start_terminal_loop, daemon=True)
    terminal_thread.start()
    plt.show(block=True)


if __name__ == "__main__":
    main()
