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
| `canvas_render.py` | `CanvasRenderMixin` — 所有渲染：背景加载、网格线、坐标轴、路径点、速度着色曲线、鼠标悬停（含线速度\|v\|和时间 t）、滚轮缩放、`redraw()` 全量重绘 |
| `canvas_commands.py` | `CanvasCommandMixin` — 终端循环 + 全部命令处理（见下方命令表） |
| `path_planner.py` | 纯计算：`Waypoint`, `SpeedLimits`, `PathSamples` dataclasses；`build_path()` Hermite 插值 → 弧长 → 曲率计算 → `time_parameterize()`（legacy/toppra 双求解器）；`dump/load_session()`；`export_path_cpp()` |
| `speed_solver_toppra.py` | TOPPRA 风格 reachability 时间求解逻辑（可选 `toppra` 依赖），由 `path_planner.time_parameterize()` 调用 |

---

## Waypoint 速度约束模型

`Waypoint` 提供三种正交的速度约束，可任意组合或全部留空：

| 字段 | 类型 | 含义 |
|------|------|------|
| `vx`, `vy` | `float\|None` | **速度方向约束**：仅方向有效，大小被忽略。用于修正 Hermite 样条在该点的切线指向。切线量级由 chord 自动估算。 |
| `speed` | `float\|None` | **线速度大小约束**：该点处的线速度上限（标量），仅影响时间参数化中的速度锚点，不影响几何形状。 |
| `vw` | `float\|None` | **角速度约束**：该点处角速度（标量），仅影响 Hermite 的 `dθ/dt` 切线量。独立于 `vx`/`vy`/`speed`。 |

三者完全解耦。例如只设 `set 2 vx 1.0; set 2 vy 0.0` 约束方向但不限速；只设 `set 2 speed 0.8` 限速但不约束方向。

---

## 速度求解流程

### Legacy 求解器 (`_time_parameterize_legacy`)

```
anchor_linear_speed_profile  (waypoint speed 约束)
  → forward_backward_speed_limit  (max_a 传播)
  → apply_curvature_constraint  (v ≤ √(lat_accel_max / κ))
  → 若有裁剪: 重新 forward_backward
  → clip to max_v
```

### TOPPRA 求解器 (`solve_toppra_profile`)

```
x_cap = (max_v / lin_gain)²           (几何增益 cap)
  → anchor waypoint speed caps
  → curvature cap: x_cap = min(x_cap, lat_accel_max / κ)
  → reachability_pass                (max_a 传播)
  → jerk smooth → fill holes → 二次 forward-backward
```

两种求解器均**不**包含角速度/角加速度对线速度的约束（ω 与 v 在全向轮上解耦）。

---

## 曲率约束 (`lat_accel_max`)

纯几何约束，与 heading / ω 无关，仅依赖路径曲线形状：

```
κ = |x'·y'' - y'·x''|          (弧长参数化曲率)
v ≤ √(lat_accel_max / κ)        (向心加速度物理模型)
```

`lat_accel_max = 0` 时完全关闭。通过 `spdlim latacc <value>` 或 `speedcfg latacc=<value>` 调节。默认值为 0。

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
- 双求解器：`legacy`（内置前向/后向裁剪 + 曲率约束）、`toppra`（可选，reachability 分析 + 曲率约束）
- `set` 命令支持 `x/y/theta/vx/vy/speed/vw`；`addpoint`/`editpoint` 仅接受 3 参数 pose
- 序列化 `dump/load_session` 兼容新旧字段（`lat_accel_max` fallback 到旧 `turn_penalty`，`speed` 不存在时 default None）

---

## 命令列表

| 命令 | 说明 |
|------|------|
| `help` | 帮助 |
| `exit` / `q` | 退出 |
| `grid` | 重绘画布 |
| `addpoint x, y, theta` | 添加路径点（grid 坐标） |
| `insert point_id x, y, theta` | 指定点后插入新点 |
| `editpoint idx x, y, theta` | 修改指定点位置（原地修改，保留已有速度约束） |
| `set <idx> <field> <value>` | 单字段修改 (x/y/theta/vx/vy/speed/vw) |
| `plan` | 重新规划并打印摘要 |
| `solver [legacy\|toppra]` | 查看/切换求解器 |
| `density <float>` | 采样密度 (>=1.0) |
| `spdlim vmax\|amax\|wmax\|awmax\|latacc <value>` | 单独设速度约束 |
| `speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw> latacc=<k>` | 批量设速度约束 |
| `showpath on/off` | 路径显示开关 |
| `body <length>, <width> \| off` | 设置/关闭悬停车体矩形 |
| `save <file>` | 保存会话到 JSON |
| `load <file>` | 从 JSON 加载会话 |
| `exportcpp <file> [name=X] [scale=1.0]` | 导出 C++ header |

---

## 依赖
Python 3.10+, `numpy`, `matplotlib`(TkAgg), `toppra`(可选，只有 `toppra` 求解器分支才需要)

---

## 测试

```bash
pytest -q example/test_project_integrity.py --junitxml=example/.reports/junit.xml
```

分组：`-k core` / `-k coord` / `-k cmd`。GUI 不可用时 `cmd` 组可能 skip（非失败）。
