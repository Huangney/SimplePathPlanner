# -*- coding: utf-8 -*-
"""
SimplePathPlanner —— 简单路径规划器
=====================================

本模块实现了基于 Matplotlib 的 GUI 画布与终端命令交互。

主要功能：
  - 加载背景图片，并在其上叠加逻辑网格（Grid）
  - 在终端通过命令添加路径点（网格坐标 + 朝向角）
  - 实时显示鼠标所在的网格坐标 / 数据坐标 / 像素坐标

坐标系统说明：
  - Grid 坐标：用户定义的逻辑网格坐标系，原点在左下，范围 [0, GRID_WIDTH] x [0, GRID_HEIGHT]
  - Data 坐标：Matplotlib 数据坐标系（也是图像像素坐标系），原点在左下，y 轴向上
  - Pixel 坐标：图像像素索引坐标系，原点在左上，y 轴向下（仅在状态栏显示用）

坐标转换流程：
  Grid 坐标 <── _grid_to_data() / _data_to_grid() ──> Data 坐标
  Data 坐标 <── 手动计算 ──> Pixel 坐标（仅右下角状态栏）

运行方式：
  python main.py
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")                                    # 强制使用 TkAgg 后端，保证跨平台兼容性
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Arc
import os

# ============================================================
# 全局配置常量
# ============================================================

# 背景图片路径（如果文件不存在，程序会给出警告并继续以纯网格模式运行）
BACKGROUND_IMAGE_PATH = "bkgrd.png"

# 逻辑网格的行列数
# GRID_WIDTH 为网格的列数（水平方向格子数）
# GRID_HEIGHT 为网格的行数（垂直方向格子数）
GRID_WIDTH = 6
GRID_HEIGHT = 12
# 设计语义：
#   x 轴对应原先“纵向 12 格”方向，取值范围 [0, GRID_HEIGHT]
#   y 轴对应原先“横向 6 格”方向，取值范围 [0, GRID_WIDTH]

# -------------------------------------------------------------------
# 网格在数据坐标系（图像像素坐标系）中的边界位置
# 注意：Matplotlib 数据坐标原点在图像左下角，y 轴方向向上
#       ┌──────────────────────┐  ← y 上边界（图像顶部，y 较小 ── 因为 origin="upper"）
#       │                      │
#       │    （图像区域）       │
#       │                      │
#       └──────────────────────┘  ← y 下边界（图像底部，y 较大）
#
# 当四个值全为 0 时，退化为"整张图即网格"，网格区域自动填满整张图片
# -------------------------------------------------------------------
GRID_X0 = 2    # 网格左边界（x 较小）
GRID_Y0 = 796  # 网格下边界（图像底部附近，y 较大）
GRID_X1 = 372  # 网格右边界（x 较大）
GRID_Y1 = 54   # 网格上边界（图像顶部附近，y 较小）


# ============================================================
# GridCanvas —— 核心画布类
# ============================================================
class GridCanvas:
    """
    路径规划画布类。

    职责：
      - 创建并管理 Matplotlib 图形窗口
      - 加载背景图片并叠加显示
      - 在图片上绘制逻辑网格线
      - 管理用户添加的路径点列表，并在图上渲染
      - 提供终端命令行交互接口（输入命令控制程序行为）
      - 处理坐标映射与实时鼠标坐标显示

    属性：
      fig, ax               : Matplotlib Figure 和 Axes 对象
      _has_image            : 是否成功加载了背景图片
      _img_w, _img_h        : 背景图片的宽和高（像素）
      _running              : 终端命令循环的运行标志
      points                : 用户添加的路径点列表，每项为 (gx, gy, theta)
                               gx/gy 为网格坐标，theta 为朝向角（弧度）
      coord_text            : 左下角实时坐标文本的 Artist 对象
      _img_artist           : imshow 返回的图像 Artist，用于控制显示行为
    """

    def __init__(self):
        """
        初始化画布：创建窗口 → 绑定事件 → 加载背景图 → 画网格 → 打印信息。
        """

        # ------------------------------------------------------------------
        # 步骤 1：创建 Matplotlib 图形与坐标轴
        # ------------------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("SimplePathPlanner")

        # ------------------------------------------------------------------
        # 步骤 2：初始化内部状态
        # ------------------------------------------------------------------
        self._has_image = False                             # 标记背景图是否加载成功
        self._img_w = 0                                     # 背景图宽度（像素）
        self._img_h = 0                                     # 背景图高度（像素）
        self._running = True                                # 控制终端命令循环是否继续运行
        self.points = []                                    # 存储用户添加的路径点 (gx, gy, theta)

        # ------------------------------------------------------------------
        # 步骤 3：在图形左下角创建坐标信息文本
        # 使用 figure 坐标 (0.01, 0.01) 定位在最左下角
        # transFigure 表示坐标值是相对于整个 figure 的比例（0~1）
        # ------------------------------------------------------------------
        self.coord_text = self.fig.text(
            0.01, 0.01, "", fontsize=9, va="bottom", ha="left",
            family="monospace", transform=self.fig.transFigure)

        # ------------------------------------------------------------------
        # 步骤 4：绑定鼠标移动事件，实现实时坐标显示
        # ------------------------------------------------------------------
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # ------------------------------------------------------------------
        # 步骤 5：依次执行初始化的各个子步骤
        # ------------------------------------------------------------------
        self._setup_view()                                  # 配置坐标轴外观（隐藏刻度、等比例缩放等）
        self._load_background()                             # 尝试加载背景图片
        self._apply_limits()                                # 根据图片尺寸或网格尺寸设置坐标轴范围
        self._draw_grid_lines()                             # 绘制逻辑网格线
        self._draw_coordinate_axes()                        # 绘制原点处坐标系正方向（+x, +y, +w）
        self._draw_points()                                 # 绘制已有路径点
        self._print_grid_info()                             # 在终端打印网格映射信息
        self.fig.tight_layout()                             # 自动调整子图边距，使布局紧凑

    # ================================================================
    # 视图与外观配置
    # ================================================================

    def _setup_view(self):
        """
        配置坐标轴的外观：
          - 设置等比例缩放（1 个数据单位在水平和垂直方向上的物理长度相同）
          - 隐藏坐标轴刻度和刻度标签
          - 隐藏坐标轴边框线
          - 覆盖 format_coord 回调，使右下角工具栏显示自定义坐标信息
        """
        self.ax.set_aspect("equal")
        # bottom=False, left=False 表示去掉底部和左侧的刻度线
        # labelbottom=False, labelleft=False 表示去掉底部和左侧的刻度标签数字
        self.ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)                        # 隐藏上下左右四条边框线

        # 覆盖 Matplotlib 默认的右下角状态栏回调函数
        # 当鼠标在图上移动时，工具栏右下角会显示 _format_coord_status 的返回值
        self.ax.format_coord = self._format_coord_status

    def _format_coord_status(self, x, y):
        """
        Matplotlib 右下角状态栏的文字格式化回调。
        输入 x, y 为鼠标在 Data 坐标系中的位置。
        返回拼接好的状态字符串（显示像素坐标和数据坐标）。
        """
        # 右下角状态栏：同时显示像素坐标（图像索引）和数据坐标（数学坐标）
        data_part = f"Data: ({x:.1f}, {y:.1f})"
        if not self._has_image:
            return data_part

        # 将 Data 坐标转换为 Pixel 坐标
        # Data 坐标原点在左下、y 向上，Pixel 坐标原点在左上、y 向下
        # 所以需要：py = img_h - 1 - y
        px = int(np.floor(x))
        py = int(np.floor(self._img_h - 1 - y))
        if 0 <= px < self._img_w and 0 <= py < self._img_h:
            pixel_part = f"Pixel: ({px}, {py})"
        else:
            pixel_part = "Pixel: (out)"
        return f"{pixel_part}  |  {data_part}"

    # ================================================================
    # 坐标轴范围
    # ================================================================

    def _apply_limits(self):
        """
        设置 Axes 的可视范围（xlim / ylim）。
        - 有背景图时：范围 = [0, img_w] x [0, img_h]
        - 无背景图时：范围 = [0, GRID_WIDTH] x [0, GRID_HEIGHT]
        """
        if self._has_image:
            self.ax.set_xlim(0, self._img_w)
            self.ax.set_ylim(0, self._img_h)
        else:
            self.ax.set_xlim(0, GRID_WIDTH)
            self.ax.set_ylim(0, GRID_HEIGHT)

    # ================================================================
    # 网格数据边界（Grid ↔ Data 坐标映射的核心）
    # ================================================================

    def _grid_data_bounds(self):
        """
        返回网格在 Data 坐标系中的矩形边界。

        返回值：
          (x_left, y_bottom, x_right, y_top)
          - x_left / y_bottom 是左下角
          - x_right / y_top 是右上角

        规则：
          1. 无背景图时：网格边界 = [0, 0, GRID_WIDTH, GRID_HEIGHT]（1:1 映射）
          2. 有背景图且四个 GRID_* 常量全为 0 时：网格边界 = 整张图的范围
          3. 有背景图且 GRID_* 已设定时：使用用户指定的边界值
        """
        # 规则 1：无法加载图片时直接退化
        if not self._has_image:
            return 0.0, 0.0, float(GRID_WIDTH), float(GRID_HEIGHT)
        # 规则 2：全零即全图
        if GRID_X0 == 0 and GRID_Y0 == 0 and GRID_X1 == 0 and GRID_Y1 == 0:
            return 0.0, 0.0, float(self._img_w), float(self._img_h)
        # 规则 3：返回用户配置的边界
        return float(GRID_X0), float(GRID_Y0), float(GRID_X1), float(GRID_Y1)

    # ================================================================
    # 背景图片加载
    # ================================================================

    def _load_background(self):
        """
        尝试加载并显示背景图片。
        - 如果路径为空字符串，跳过加载
        - 如果文件不存在，打印警告并跳过
        - 加载成功后，将图片以 "origin=upper" 的方式显示（像素 y=0 在顶部）
          extent 参数确保图片左下角对齐 Data 坐标的 (0, 0)
        """
        if not BACKGROUND_IMAGE_PATH:
            return
        if not os.path.exists(BACKGROUND_IMAGE_PATH):
            print(f"[WARN] background image not found: {BACKGROUND_IMAGE_PATH}")
            return

        # 使用 matplotlib.image.imread 读取图片（支持 PNG / JPEG 等格式）
        self._img = mpimg.imread(BACKGROUND_IMAGE_PATH)
        # shape 为 (height, width) 或 (height, width, channels)
        self._img_h, self._img_w = self._img.shape[:2]
        self._has_image = True

        # imshow 将图片以图像的形式渲染到 Axes 上
        # extent 参数：指定图片在 Data 坐标系中的四角范围 [x_min, x_max, y_min, y_max]
        # origin="upper"：强制图片的第 0 行像素在 y 轴最大值处（图片顶部 = y 最大值）
        # zorder=0：确保图片在所有其他元素的最底层
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

    # ================================================================
    # 网格线绘制
    # ================================================================

    def _draw_grid_lines(self):
        """
        在 Data 坐标系中绘制逻辑网格的横线和竖线。
        - 竖线：每个整数 gx 处画一条从上到下的线
        - 横线：每个整数 gy 处画一条从左到右的线
        网格线条颜色为黑色，线宽 0.5，位于 zorder=2（在图片之上）
        """
        # 获取网格在 Data 坐标中的边界
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()

        # 画 y=常量 的竖线（y 方向共 GRID_WIDTH 段）
        for gy in range(GRID_WIDTH + 1):
            dx, _ = self._grid_to_data(0, gy)               # 计算第 gy 列在 Data 坐标中的 x 位置
            self.ax.plot([dx, dx], [dy0, dy1], color="black", linewidth=0.5, zorder=2)

        # 画 x=常量 的横线（x 方向共 GRID_HEIGHT 段）
        for gx in range(GRID_HEIGHT + 1):
            _, dy = self._grid_to_data(gx, 0)               # 计算第 gx 行在 Data 坐标中的 y 位置
            self.ax.plot([dx0, dx1], [dy, dy], color="black", linewidth=0.5, zorder=2)

    # ================================================================
    # 路径点绘制
    # ================================================================

    def _draw_points(self):
        """
        遍历 self.points 中的所有路径点，在图上绘制红色圆点和标签。
        每个点显示其编号（P1, P2,...）和完整的网格坐标与朝向角。
        """
        for idx, (gx, gy, theta) in enumerate(self.points, start=1):
            # 将网格坐标转换为 Data 坐标，用于在图上定位
            dx, dy = self._grid_to_data(gx, gy)
            self.ax.plot(dx, dy, marker="o", markersize=6, color="red", zorder=5)
            # 在点右上偏移 (3, 3) 的位置显示标签
            self.ax.text(
                dx + 3, dy + 3, f"P{idx} ({gx:.1f}, {gy:.1f}, {theta:.2f})",
                color="red", fontsize=8, zorder=6
            )

    def _draw_coordinate_axes(self):
        """
        在网格原点 (0,0) 处绘制右手系方向标识：
          - +x：屏幕向下
          - +y：屏幕向右
          - +w：绕原点逆时针旋转方向
        """
        ox, oy = self._grid_to_data(0.0, 0.0)
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        span_x = abs(dx1 - dx0)
        span_y = abs(dy1 - dy0)

        # 按当前数据范围取固定比例，避免坐标语义调整后箭头过短
        len_y = 0.14 * span_y  # +x（屏幕向下）使用竖向长度
        len_x = 0.28 * span_x  # +y（屏幕向右）使用横向长度

        # 按“屏幕方向”定义：+x 向下（Data y 减小），+y 向右（Data x 增大）
        axis_lw = 3.2
        self.ax.annotate(
            "", xy=(ox, oy - len_y), xytext=(ox, oy),
            arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=axis_lw),
            zorder=7
        )
        self.ax.annotate(
            "", xy=(ox + len_x, oy), xytext=(ox, oy),
            arrowprops=dict(arrowstyle="->", color="seagreen", lw=axis_lw),
            zorder=7
        )
        self.ax.text(ox, oy - len_y, " +x", color="dodgerblue", fontsize=9, va="top", ha="center", zorder=8)
        self.ax.text(ox + len_x, oy, " +y", color="seagreen", fontsize=9, va="center", ha="left", zorder=8)

        # +w（omega）用劣弧（90°）表示逆时针方向：从“下”转到“右”
        radius = 0.38 * min(len_x, len_y)
        arc = Arc((ox, oy), 2 * radius, 2 * radius, angle=0, theta1=-90, theta2=0,
                  color="darkorange", lw=1.8, zorder=7)
        self.ax.add_patch(arc)

        # 在弧末端添加箭头头部，明确逆时针方向（劣弧）
        theta_tail = np.deg2rad(-20)
        theta_head = np.deg2rad(0)
        tail = (ox + radius * np.cos(theta_tail), oy + radius * np.sin(theta_tail))
        head = (ox + radius * np.cos(theta_head), oy + radius * np.sin(theta_head))
        self.ax.annotate(
            "", xy=head, xytext=tail,
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.8),
            zorder=8
        )
        self.ax.text(ox + 0.55 * radius, oy - 0.55 * radius, " +w", color="darkorange", fontsize=9, zorder=8)

    # ================================================================
    # 坐标映射（Grid ↔ Data）
    # ================================================================

    def _grid_to_data(self, gx, gy):
        """
        将网格坐标 (gx, gy) 线性映射为数据坐标 (dx, dy)。

        设计语义（已互换）：
          - x 走纵向 12 格（GRID_HEIGHT）
          - y 走横向 6 格（GRID_WIDTH）

        映射公式：
          dx = dx0 + (gy / GRID_WIDTH) * (dx1 - dx0)
          dy = dy0 + (gx / GRID_HEIGHT) * (dy1 - dy0)

        其中 (dx0, dy0, dx1, dy1) 由 _grid_data_bounds() 返回。
        """
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        dx = dx0 + gy / GRID_WIDTH * (dx1 - dx0)
        dy = dy0 + gx / GRID_HEIGHT * (dy1 - dy0)
        return dx, dy

    def _data_to_grid(self, dx, dy):
        """
        将数据坐标 (dx, dy) 线性映射为网格坐标 (gx, gy)。

        映射公式（_grid_to_data 的逆变换）：
          gx = (dy - dy0) / (dy1 - dy0) * GRID_HEIGHT
          gy = (dx - dx0) / (dx1 - dx0) * GRID_WIDTH

        返回值：
          (gx, gy) 或 (None, None)（当分母为零时）
        """
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        if dx1 == dx0 or dy1 == dy0:
            return None, None
        gx = (dy - dy0) / (dy1 - dy0) * GRID_HEIGHT
        gy = (dx - dx0) / (dx1 - dx0) * GRID_WIDTH
        return gx, gy

    # ================================================================
    # 鼠标事件处理
    # ================================================================

    def _on_mouse_move(self, event):
        """
        鼠标在 Axes 上移动时的回调函数。

        功能：
          - 将鼠标所在的 Data 坐标转换为 Grid 坐标
          - 更新左下角文本，实时显示 Grid / Data / Image 信息
          - 如果鼠标移出 Axes 范围，不做任何处理（return）
        """
        # 判断鼠标是否在当前 Axes 内，且坐标数据有效
        if event.inaxes != self.ax or event.xdata is None:
            return
        dx, dy = event.xdata, event.ydata
        gx, gy = self._data_to_grid(dx, dy)

        # 拼接左下角显示的文本内容
        parts = []
        # 新语义范围：x in [0, GRID_HEIGHT], y in [0, GRID_WIDTH]
        if gx is not None and gy is not None and 0 <= gx <= GRID_HEIGHT and 0 <= gy <= GRID_WIDTH:
            parts.append(f"Grid: ({gx:.1f}, {gy:.1f})")
        if self._has_image:
            parts.append(f"Data: ({dx:.1f}, {dy:.1f})")
            parts.append(f"Image: {self._img_w}x{self._img_h}")
        else:
            parts.append(f"Pos: ({dx:.1f}, {dy:.1f})")
        self.coord_text.set_text("  |  ".join(parts))

    # ================================================================
    # 终端信息打印
    # ================================================================

    def _print_grid_info(self):
        """
        在终端打印网格配置信息，方便用户确认网格映射是否正确。
        """
        if not self._has_image:
            print(f"[INFO] no image, grid maps directly: {GRID_WIDTH}x{GRID_HEIGHT}")
            return
        dx0, dy0, dx1, dy1 = self._grid_data_bounds()
        print(f"[INFO] grid in data coords: "
              f"x=[{dx0:.1f}, {dx1:.1f}]  y=[{dy0:.1f}, {dy1:.1f}]  "
              f"|  size={dx1-dx0:.0f}x{dy1-dy0:.0f}")

    # ================================================================
    # 重绘
    # ================================================================

    def redraw(self):
        """
        完全清除当前 Axes 并重新绘制所有元素。
        执行顺序：清除 → 重新设置视图 → 重新加载背景 → 重新设置范围 → 重画网格 → 重画点。
        每次 addpoint 后都会调用此方法以反映最新的点列表。
        """
        self.ax.cla()                                       # 清除 Axes 上的所有内容
        self._setup_view()                                  # 重新配置外观（cla 会清除之前的设置）
        self._load_background()                             # 重新显示背景图片
        self._apply_limits()                                # 重新设置坐标轴范围
        self._draw_grid_lines()                             # 重新绘制网格线
        self._draw_coordinate_axes()                        # 重新绘制原点坐标系方向
        self._draw_points()                                 # 重新绘制所有路径点
        self.fig.canvas.draw_idle()                         # 请求 GUI 后端异步刷新窗口

    # ================================================================
    # 终端命令循环
    # ================================================================

    def start_terminal_loop(self):
        """
        在独立线程中运行终端命令行交互循环。
        不断读取用户输入的命令并分发处理，直到用户输入 exit/q 或触发中断。

        注意：此方法通过 daemon 线程运行，因此不会阻塞 GUI 主线程。
        """
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
        """打印所有可用命令的帮助信息。"""
        print("Commands:")
        print("  help      Show this help")
        print("  exit/q    Exit the program")
        print("  grid      Redraw the grid")
        print("  addpoint x, y, theta   Add a point in grid coords and draw it")
        print(f"           range: x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]")

    def _handle_command(self, cmd):
        """
        解析并执行用户输入的命令。

        参数：
          cmd : list[str]，例如 ["addpoint", "1.0,", "1.0,", "1.57"]
                索引 0 为命令名，后续元素为参数
        """
        op = cmd[0].lower()
        if op in ("exit", "q"):
            self._running = False                           # 退出终端循环
            plt.close("all")                                # 关闭所有 Matplotlib 窗口
            print("Bye.")
        elif op == "help":
            self._print_help()
        elif op == "grid":
            self.redraw()                                   # 调取重绘函数刷新画面
            print("Grid redrawn.")
        elif op == "addpoint":
            self._cmd_addpoint(cmd[1:])                     # 将剩余参数传给 addpoint 处理函数
        else:
            print(f"Unknown command: {op}. Type 'help' for available commands.")

    # ================================================================
    # addpoint 命令的具体实现
    # ================================================================

    def _cmd_addpoint(self, args):
        """
        处理 addpoint 命令：解析用户输入的 x, y, theta 参数并添加路径点。

        输入格式：
          命令参数 args 为 list[str]，用户可能以 "x, y, theta" 或 "x,y,theta" 等格式输入。
          本函数会将所有参数拼接后按逗号分割，提取三个浮点数。

        验证规则：
          - 必须有恰好 3 个逗号分隔的值
          - 全部能被解析为浮点数
          - gx 在 [0, GRID_HEIGHT] 范围内，gy 在 [0, GRID_WIDTH] 范围内

        执行成功后将点添加到 self.points 并触发重绘。
        """
        if not args:
            print("Usage: addpoint x, y, theta")
            return

        # 将命令行参数全部拼接为一个字符串，去除所有空格
        # 例如 ["1.0,", "1.0,", "1.57"] → "1.0,1.0,1.57"
        raw = " ".join(args).replace(" ", "")
        parts = raw.split(",")
        if len(parts) != 3:
            print("Usage: addpoint x, y, theta")
            return

        # 尝试将三个字符串解析为浮点数
        try:
            gx, gy, theta = map(float, parts)
        except ValueError:
            print("Invalid number format. Example: addpoint 1.0, 1.0, 1.57")
            return

        # 检查网格坐标是否在有效范围内
        if not (0.0 <= gx <= GRID_HEIGHT and 0.0 <= gy <= GRID_WIDTH):
            print(f"Point out of grid range. x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]")
            return

        # 验证通过，添加到列表并刷新画面
        self.points.append((gx, gy, theta))
        self.redraw()
        print(f"Point added: ({gx:.3f}, {gy:.3f}, {theta:.3f})")
