# SimplePathPlanner — 项目框架

基于 Python + Matplotlib 的终端交互式路径规划工具。终端输入路径点，GUI 实时绘制 Hermite 插值平滑路径。

---

## 文件职责速览

| 文件 | 职责 |
|------|------|
| `main.py` | 入口：解析 `-p/--profile` → 取 ProfileConfig → 创建 GridCanvas → daemon 线程跑终端循环 → `plt.show()` |
| `app_config.py` | 常量 (`GRID_WIDTH=6, HEIGHT=12`, alpha, 默认密度/速度约束) + `ProfileConfig` frozen dataclass + `PROFILE_CONFIGS` + `get_profile_config()` |
| `coord_utils.py` | Grid↔Data 坐标转换，所有函数接受 `grid_x0/y0/x1/y1` 参数（来自 ProfileConfig） |
| `canvas.py` | `GridCanvas(CanvasRenderMixin, CanvasCommandMixin)` — 仅 `__init__`，轻量组装 |
| `canvas_render.py` | `CanvasRenderMixin` — 所有渲染：背景加载、网格线、坐标轴、路径点、速度着色曲线、鼠标悬停、滚轮缩放、`redraw()` 全量重绘 |
| `canvas_commands.py` | `CanvasCommandMixin` — 终端循环 + 全部命令处理（见下方命令表） |
| `path_planner.py` | 纯计算：`Waypoint`, `SpeedLimits`, `PathSamples` dataclasses；`build_path()` Hermite 插值 → 弧长 → `time_parameterize()`（legacy/toppra 双求解器）；`dump/load_session()`, `export_path_cpp()` |
| `speed_solver_toppra.py` | TOPPRA 风格 reachability 时间求解（需 `pip install toppra`），由 `path_planner.time_parameterize()` 调用 |

---

## 坐标系统

- **Grid** (gx, gy)：逻辑网格，`gx∈[0,12]`, `gy∈[0,6]`。Waypoint 以 grid 坐标存储和输入。
- **Data** (dx, dy)：有背景图时由 `ProfileConfig.grid_x0/y0/x1/y1` 映射到像素，无图时与 Grid 一致。
- `gx` 对应行（沿图像 y），`gy` 对应列（沿图像 x）。

---

## 关键模式

- Mixin 多重继承：`GridCanvas(CanvasRenderMixin, CanvasCommandMixin)`
- `redraw()` = 全量清空 + 重建（背景→网格→轴线→路径→路径点）
- 终端输入 ↔ GUI 通过共享 `self.points / path_samples / speed_limits` 解耦
- `path_planner.py` + `speed_solver_toppra.py` 完全独立于 matplotlib，可单独测试
- 双求解器：`legacy`（内置前向/后向裁剪）、`toppra`（可选，reachability 分析）

---

## 命令列表

| 命令 | 说明 |
|------|------|
| `help` | 帮助 |
| `exit` / `q` | 退出 |
| `grid` | 重绘画布 |
| `addpoint x,y,theta[,vx,vy,vw]` | 添加路径点（grid 坐标） |
| `insert point_id x,y,theta` | 指定点后插入新点 |
| `editpoint idx x,y,theta[,vx,vy,vw]` | 修改指定点 |
| `set <idx> <field> <value>` | 单字段修改 (x/y/theta/vx/vy/vw) |
| `plan` | 重新规划并打印摘要 |
| `solver [legacy\|toppra]` | 查看/切换求解器 |
| `density <float>` | 采样密度 (>=1.0) |
| `spdlim vmax\|amax\|wmax\|awmax <value>` | 单独设速度约束 |
| `speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw>` | 批量设速度约束 |
| `showpath on/off` | 路径显示开关 |
| `save <file>` | 保存会话到 JSON |
| `load <file>` | 从 JSON 加载会话 |
| `exportcpp <file> [name=X] [scale=1.0]` | 导出 C++ header |

---

## 依赖
Python 3.10+, `numpy`, `matplotlib`(TkAgg), `toppra`(可选)

---

## 测试

```bash
pytest -q example/test_project_integrity.py --junitxml=example/.reports/junit.xml
```

分组：`-k core` / `-k coord` / `-k cmd`。GUI 不可用时 `cmd` 组可能 skip（非失败）。
