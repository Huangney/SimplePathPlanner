# SimplePathPlanner — 项目框架

基于 Python + Matplotlib 的终端交互式路径规划工具。
用户在终端中输入路径点，GUI 画布实时绘制路径点及 Hermite 插值生成的平滑路径曲线。

---

## 文件职责

### `main.py` — 程序入口
- 创建 `GridCanvas` 实例（初始化窗口、加载背景图、首次绘制）
- 在 daemon 线程中运行 `GridCanvas.start_terminal_loop()`（终端命令循环）
- 主线程进入 `plt.show(block=True)` 响应 GUI 事件
- 关闭窗口即退出程序

### `app_config.py` — 全局常量
- `BACKGROUND_IMAGE_PATH` — 背景图文件名 (默认 `"bkgrd.png"`)
- `GRID_WIDTH=6`, `GRID_HEIGHT=12` — 逻辑网格尺寸
- `GRID_X0/Y0/X1/Y1` — 网格到图像像素坐标的映射边界
- `BACKGROUND_ALPHA=0.35`, `DEFAULT_PATH_DENSITY=20.0`

### `coord_utils.py` — 坐标转换
- `grid_to_data()` / `data_to_grid()` — 网格坐标 ↔ 数据/图像坐标
- `grid_vec_to_data_vec()` — 向量转换（如 heading、velocity 矢量）
- `grid_data_bounds()` — 获取当前数据坐标边界（有无背景图两种模式）
- `format_coord_status()` — 状态栏坐标格式化

> **注意**：代码中 `gx` 对应行（沿图像 y 方向），`gy` 对应列（沿图像 x 方向）。`grid_to_data` 返回 `(dx, dy)` 时，`dy` 由 `gx` 计算。

### `canvas.py` — 渲染与命令交互
核心类 `GridCanvas`：
- **初始化** (`__init__`)：创建 Figure/Axes、加载背景图、绘制网格线、坐标轴（+x / +y / +w 箭头）、路径点和路径曲线
- **`redraw()`**：清空 Axes 后完整重绘（背景 → 网格 → 路径 → 路径点）
- **`_draw_points()`**：绘制每个 Waypoint（红点 + 绿色 heading 箭头 + 紫色 velocity 箭头 + 标签）
- **`_draw_path()`**：将 `path_samples` 转为 data 坐标后绘制为蓝色曲线
- **`_rebuild_path()`**：调用 `path_planner.build_path()` 根据当前 points 和 density 重新生成路径
- **终端命令循环** (`start_terminal_loop`) → `_handle_command()` → 各 `_cmd_*` 方法

#### 命令列表
| 命令 | 说明 |
|------|------|
| `addpoint x,y,theta[,vx,vy,vw]` | 添加路径点（grid 坐标），可选速度分量 |
| `plan` | 重新生成路径并打印摘要 |
| `density <float>` | 设置采样密度 (>=1.0) |
| `showpath on/off` | 切换路径曲线可见性 |
| `grid` | 重绘画布 |
| `save <file>` | 保存当前路径点和设置到 JSON |
| `load <file>` | 从 JSON 加载路径点和设置 |
| `help` | 打印帮助 |
| `exit` / `q` | 退出 |

### `path_planner.py` — 路径规划核心（纯计算，无 UI 依赖）
- **`Waypoint`** (dataclass)：`x, y, theta, vx?, vy?, vw?`
- **`PathSamples`** (dataclass)：`x[], y[], theta[], s[]` (弧长), `meta` 字典
- **`build_path(waypoints, density)`** — 核心函数：
  1. 将 waypoints 的 theta 做 shortest-angle unwrap
  2. 按弦长累计参数 t，估计各点导数（用户可覆盖 vx/vy/vw）
  3. 逐段 Hermite 三次插值采样，采样数 = `max(8, ceil(弦长 * density)) + 1`
  4. 最终 theta 重新 wrap 到 [-π, π]
  5. 计算弧长 `s[]`
- `wrap_angle()` / `unwrap_shortest()` — 角度工具
- `dump_session()` / `load_session()` — JSON 存取（兼容 sequence 和 `Waypoint` 对象）

---

## 数据流

```
终端输入命令
    ↓
canvas.py: GridCanvas._handle_command()
    ↓ (addpoint 修改 self.points / density 修改 self.path_density)
    ↓
canvas.py: redraw()
    ├→ _rebuild_path()  →  path_planner.build_path()  →  PathSamples
    ├→ _draw_path()     →  绘制蓝色曲线
    └→ _draw_points()   →  绘制红点 + 箭头 + 标签
    ↓
fig.canvas.draw_idle()  →  Matplotlib GUI 更新
```

---

## 坐标系统

- **Grid 坐标** (gx, gy)：逻辑网格，范围 `gx∈[0,12]`, `gy∈[0,6]`（对应 HEIGHT×WIDTH）
- **Data 坐标** (dx, dy)：有背景图时映射到图像像素范围 (~372×796)，无背景图时与 Grid 坐标一致
- 所有 waypoint 以 grid 坐标存储和输入，渲染时通过 `coord_utils` 转换为 data 坐标

---

## 关键模式
- `GridCanvas._has_image` 标志控制两套坐标映射逻辑，所有绘图函数都据此判断
- `redraw()` 采用"全量清空 + 重建"模式，而非增量更新
- 终端输入与 GUI 通过共享 `self.points` / `self.path_samples` 等属性解耦
- `path_planner.py` 完全独立于 matplotlib，可单独测试和复用

## 依赖
- Python 3.10+
- `numpy`
- `matplotlib` (TkAgg 后端)
