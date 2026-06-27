from __future__ import annotations

import matplotlib.pyplot as plt

from app_config import DEFAULT_PATH_DENSITY, GRID_HEIGHT, GRID_WIDTH
from path_planner import SpeedLimits, Waypoint, dump_session, load_session


class CanvasCommandMixin:
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
        print("  addpoint x, y, theta   添加路径点（首尾点必须带 theta）")
        print("  insert point_id x, y[, theta]   在指定路径点后插入新点（中间点 theta 可省略）")
        print("  editpoint idx x, y[, theta]   修改指定路径点（中间点可省略 theta）")
        print("  set <idx> <field> <value>   单独修改点约束（field: x/y/theta/vx/vy/speed/vw）")
        print("           示例: set 2 speed 0.8, set 1 vy -0.3, set 3 vx 1.0")
        print("           坐标范围：x in [0,{GRID_HEIGHT}], y in [0,{GRID_WIDTH}]")
        print("  plan      重新规划路径并打印摘要")
        print("  solver [coupled|legacy|toppra]   查看速度求解器（旧名称会映射到 coupled）")
        print("  density d 设置路径采样密度 (d >= 1.0)")
        print("  spdlim <param> <value>   单独设置全局速度约束 (param: vmax/amax/wmax/awmax/latacc)")
        print("  ispdlim <segment_id> <vmax|off>   设置/清除单段限速，例如 ispdlim 1 2.5 表示 P1→P2")
        print("  speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw> latacc=<k>   设置全局速度约束")
        print("  showpath on/off   切换路径曲线显示")
        print("  body <length>, <width> | off   设置/关闭悬停车体矩形（单位同网格）")
        print("  save <文件>   保存当前路径点和设置到 JSON")
        print("  load <文件>   从 JSON 加载路径点和设置")
        print("  exportcpp <文件> [name=PathName] [scale=1.0]   导出 MCU C++ 路径头文件")
        print("  鼠标悬停在画布上按 a   在当前鼠标位置新增一个点")
        print("  鼠标悬停在路径上按 i   将当前路径采样点插入为关键点")
        print("  鼠标悬停在已有关键点上按 d   删除该关键点")
        print("  快速双击已有关键点    弹出窗口编辑 x,y,theta / vx,vy / w,velo")
        print("  快速双击路径非关键点  弹出窗口编辑该关键点段的 vmax")

    def _validate_grid_pose(self, gx: float, gy: float) -> tuple[float, float] | None:
        if not (0.0 <= gx <= GRID_HEIGHT and 0.0 <= gy <= GRID_WIDTH):
            print(f"路径点超出网格范围。x 在 [0,{GRID_HEIGHT}]，y 在 [0,{GRID_WIDTH}]")
            return None
        return float(gx), float(gy)

    def _format_theta_label(self, theta: float | None) -> str:
        return "-" if theta is None else f"{float(theta):.3f}"

    def _parse_pose_parts(self, args, usage: str) -> tuple[float, float, float | None] | None:
        parts = " ".join(args).replace(" ", "").split(",")
        if len(parts) not in (2, 3):
            print(usage)
            return None
        try:
            gx = float(parts[0])
            gy = float(parts[1])
            theta = None if len(parts) == 2 or parts[2].lower() in ("", "none", "null") else float(parts[2])
        except ValueError:
            print("数值格式无效。示例: addpoint 1.0, 1.0, 1.57 或 insert 2 1.0, 2.0")
            return None
        pose = self._validate_grid_pose(gx, gy)
        if pose is None:
            return None
        gx, gy = pose
        return gx, gy, theta

    def _theta_allowed_at_insert(self, insert_idx: int, theta: float | None) -> bool:
        if theta is not None:
            return True
        # A theta-less waypoint is only valid when it remains an intermediate point.
        if insert_idx <= 0 or insert_idx >= len(self.points):
            print("首尾路径点必须设置 theta；只有中间过渡点可以省略 theta。")
            return False
        return True

    def _theta_allowed_at_existing_index(self, idx0: int, theta: float | None) -> bool:
        if theta is not None:
            return True
        if idx0 <= 0 or idx0 >= len(self.points) - 1:
            print("首尾路径点必须设置 theta；只有中间过渡点可以省略 theta。")
            return False
        return True

    def _insert_waypoint_at(self, insert_idx: int, gx: float, gy: float, theta: float | None) -> int:
        if not self._theta_allowed_at_insert(insert_idx, theta):
            return 0
        self._shift_interval_speed_limits_for_insert(insert_idx)
        self.points.insert(insert_idx, Waypoint(x=float(gx), y=float(gy), theta=None if theta is None else float(theta)))
        self.redraw()
        return insert_idx + 1

    def _replace_speed_limits(self, **updates):
        values = {
            "max_v": self.speed_limits.max_v,
            "max_a": self.speed_limits.max_a,
            "max_w": self.speed_limits.max_w,
            "max_aw": self.speed_limits.max_aw,
            "max_jk": self.speed_limits.max_jk,
            "lat_accel_max": self.speed_limits.lat_accel_max,
            "interval_speed_limits": self.speed_limits.interval_speed_limits,
        }
        values.update(updates)
        self.speed_limits = SpeedLimits(**values)

    def _interval_speed_limit_dict(self) -> dict[int, float]:
        return {int(seg_idx): float(vmax) for seg_idx, vmax in self.speed_limits.interval_speed_limits}

    def _set_interval_speed_limit(self, segment_idx0: int, vmax: float | None):
        limits = self._interval_speed_limit_dict()
        if vmax is None:
            limits.pop(int(segment_idx0), None)
        else:
            limits[int(segment_idx0)] = max(float(vmax), 0.0)
        self._replace_speed_limits(interval_speed_limits=tuple(sorted(limits.items())))

    def _shift_interval_speed_limits_for_insert(self, insert_idx: int):
        if not self.speed_limits.interval_speed_limits:
            return
        shifted: dict[int, float] = {}
        split_seg = int(insert_idx) - 1
        for seg_idx, vmax in self.speed_limits.interval_speed_limits:
            seg_idx = int(seg_idx)
            if 0 <= split_seg == seg_idx < len(self.points) - 1:
                shifted[seg_idx] = float(vmax)
                shifted[seg_idx + 1] = float(vmax)
            elif seg_idx >= int(insert_idx):
                shifted[seg_idx + 1] = float(vmax)
            else:
                shifted[seg_idx] = float(vmax)
        self._replace_speed_limits(interval_speed_limits=tuple(sorted(shifted.items())))

    def _shift_interval_speed_limits_for_delete(self, delete_idx: int):
        if not self.speed_limits.interval_speed_limits:
            return
        if len(self.points) < 2:
            self._replace_speed_limits(interval_speed_limits=())
            return

        shifted: dict[int, float] = {}
        left_seg = int(delete_idx) - 1
        right_seg = int(delete_idx)
        for seg_idx, vmax in self.speed_limits.interval_speed_limits:
            seg_idx = int(seg_idx)
            if seg_idx == left_seg or seg_idx == right_seg:
                continue
            if seg_idx > right_seg:
                shifted[seg_idx - 1] = float(vmax)
            else:
                shifted[seg_idx] = float(vmax)
        self._replace_speed_limits(interval_speed_limits=tuple(sorted(shifted.items())))

    def _delete_waypoint_at(self, point_idx: int) -> bool:
        if point_idx < 0 or point_idx >= len(self.points):
            return False
        if len(self.points) <= 1:
            print("[错误] 至少需要保留一个路径点。")
            return False

        deleted_idx = point_idx + 1
        deleted_point = self.points[point_idx]
        self._shift_interval_speed_limits_for_delete(point_idx)
        del self.points[point_idx]
        self.redraw()

        theta_label = "-" if deleted_point.theta is None else f"{float(deleted_point.theta):.3f}"
        print(
            f"已删除路径点 P{deleted_idx} = "
            f"({float(deleted_point.x):.3f}, {float(deleted_point.y):.3f}, {theta_label})"
        )
        return True

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
        elif op == "insert":
            self._cmd_insert(cmd[1:])
        elif op == "editpoint":
            self._cmd_editpoint(cmd[1:])
        elif op == "set":
            self._cmd_set(cmd[1:])
        elif op == "plan":
            self._cmd_plan()
        elif op == "solver":
            self._cmd_solver(cmd[1:])
        elif op == "density":
            self._cmd_density(cmd[1:])
        elif op == "spdlim":
            self._cmd_spdlim(cmd[1:])
        elif op == "ispdlim":
            self._cmd_ispdlim(cmd[1:])
        elif op == "speedcfg":
            self._cmd_speedcfg(cmd[1:])
        elif op == "showpath":
            self._cmd_showpath(cmd[1:])
        elif op == "body":
            self._cmd_body(cmd[1:])
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
        parsed = self._parse_pose_parts(args, "用法: addpoint x, y, theta")
        if parsed is None:
            return
        gx, gy, theta = parsed
        idx = self._insert_waypoint_at(len(self.points), gx, gy, theta)
        if idx:
            print(f"路径点已添加：P{idx} = ({gx:.3f}, {gy:.3f}, {self._format_theta_label(theta)})")

    def _cmd_insert(self, args):
        if len(args) < 2:
            print("用法: insert <point_id> x, y[, theta]")
            return
        if not self.points:
            print("[错误] 当前没有路径点，无法执行插入。请先使用 addpoint。")
            return
        try:
            point_id = int(args[0])
        except ValueError:
            print("point_id 格式无效。示例: insert 2 1.0, 2.0, 0.5")
            return
        if point_id < 1 or point_id > len(self.points):
            print(f"[错误] point_id 越界：{point_id}（当前共有 {len(self.points)} 个点，合法范围 1~{len(self.points)}）")
            return

        parsed = self._parse_pose_parts(args[1:], "用法: insert <point_id> x, y[, theta]")
        if parsed is None:
            return
        gx, gy, theta = parsed

        idx = self._insert_waypoint_at(point_id, gx, gy, theta)
        if idx:
            print(f"已在 P{point_id} 后插入新点：P{idx} = ({gx:.3f}, {gy:.3f}, {self._format_theta_label(theta)})")

    def _cmd_editpoint(self, args):
        if len(args) < 2:
            print("用法: editpoint idx x, y[, theta]")
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

        parsed = self._parse_pose_parts(args[1:], "用法: editpoint idx x, y[, theta]")
        if parsed is None:
            return
        gx, gy, theta = parsed

        if not self._theta_allowed_at_existing_index(idx - 1, theta):
            return

        p = self.points[idx - 1]
        p.x = gx
        p.y = gy
        p.theta = theta
        self.redraw()
        print(f"路径点 P{idx} 已修改为：({gx:.3f}, {gy:.3f}, {self._format_theta_label(theta)})")

    def _cmd_set(self, args):
        if len(args) != 3:
            print("用法: set <idx> <field> <value>  (field: x/y/theta/vx/vy/speed/vw)")
            return
        if not self.points:
            print("[错误] 当前没有可修改的路径点。")
            return

        field: str
        value_token: str = args[2]
        try:
            idx = int(args[0])
            field = args[1].lower()
        except ValueError:
            field = args[0].lower()
            try:
                idx = int(args[1])
            except ValueError:
                print("索引格式无效。示例: set 2 x 3.5")
                return

        if field not in ("x", "y", "theta", "vx", "vy", "speed", "vw"):
            print("字段无效。可用字段: x, y, theta, vx, vy, speed, vw")
            return

        if idx < 1 or idx > len(self.points):
            print(f"[错误] 路径点索引越界：{idx}（当前共有 {len(self.points)} 个点，合法范围 1~{len(self.points)}）")
            return

        if field == "theta" and value_token.strip().lower() in ("none", "null", ""):
            value = None
        else:
            try:
                value = float(value_token)
            except ValueError:
                print("数值格式无效。示例: set 1 vy -0.3 或 set 2 theta none")
                return

        p = self.points[idx - 1]
        if field == "x":
            if not (0.0 <= value <= GRID_HEIGHT):
                print(f"x 超出网格范围。x 在 [0,{GRID_HEIGHT}]")
                return
            p.x = value
        elif field == "y":
            if not (0.0 <= value <= GRID_WIDTH):
                print(f"y 超出网格范围。y 在 [0,{GRID_WIDTH}]")
                return
            p.y = value
        elif field == "theta":
            if not self._theta_allowed_at_existing_index(idx - 1, value):
                return
            p.theta = value
        elif field == "vx":
            p.vx = value
        elif field == "vy":
            p.vy = value
        elif field == "speed":
            p.speed = value
        elif field == "vw":
            p.vw = value

        self.redraw()
        value_label = "none" if value is None else f"{float(value):.6f}"
        print(f"路径点 P{idx} 的 {field} 已设置为 {value_label}")

    def _cmd_plan(self):
        self.redraw()
        meta = self.path_samples.meta
        print(
            f"[规划] 段数={meta.get('segments', 0)}  采样点={meta.get('sample_count', 0)}  "
            f"总长度={meta.get('total_length', 0.0):.3f}  总时长={meta.get('total_time', 0.0):.3f}s  "
            f"峰值线速={meta.get('peak_v', 0.0):.3f}  峰值角速={meta.get('peak_w', 0.0):.3f}  "
            f"约束裁剪={'是' if meta.get('constraint_clipped', False) else '否'}  "
            f"求解器={meta.get('solver', self.solver)}"
        )

    def _cmd_solver(self, args):
        if len(args) == 0:
            print(f"当前求解器：{self.solver}")
            return
        if len(args) != 1:
            print("用法: solver [coupled|legacy|toppra]")
            return
        target = args[0].strip().lower()
        if target not in ("coupled", "legacy", "toppra"):
            print("求解器无效。可用: coupled, legacy, toppra")
            return
        old = self.solver
        self.solver = "coupled"
        try:
            self.redraw()
        except Exception as e:
            self.solver = old
            print(f"[错误] 切换求解器失败: {e}")
            return
        if target == "coupled":
            print("求解器已切换为：coupled")
        else:
            print(f"求解器 {target} 已映射为：coupled")

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
            print("用法: speedcfg vmax=<v> amax=<a> wmax=<w> awmax=<aw> latacc=<k>")
            return
        mapping = {
            "vmax": "max_v",
            "amax": "max_a",
            "wmax": "max_w",
            "awmax": "max_aw",
            "latacc": "lat_accel_max",
        }
        updates = {
            "max_v": self.speed_limits.max_v,
            "max_a": self.speed_limits.max_a,
            "max_w": self.speed_limits.max_w,
            "max_aw": self.speed_limits.max_aw,
            "lat_accel_max": self.speed_limits.lat_accel_max,
        }
        for token in args:
            if "=" not in token:
                print(f"参数格式无效: {token}")
                return
            key, val = token.split("=", 1)
            key = key.lower().strip()
            if key not in mapping:
                print(f"未知参数: {key}（可用: vmax, amax, wmax, awmax, latacc）")
                return
            try:
                fval = float(val)
            except ValueError:
                print(f"参数值无效: {token}")
                return
            if fval < 0.0:
                print(f"参数必须 >= 0: {token}")
                return
            updates[mapping[key]] = fval

        self._replace_speed_limits(
            max_v=updates["max_v"],
            max_a=updates["max_a"],
            max_w=updates["max_w"],
            max_aw=updates["max_aw"],
            lat_accel_max=updates["lat_accel_max"],
        )
        self.redraw()
        print(
            "速度约束已更新："
            f"vmax={self.speed_limits.max_v:.3f}, "
            f"amax={self.speed_limits.max_a:.3f}, "
            f"wmax={self.speed_limits.max_w:.3f}, "
            f"awmax={self.speed_limits.max_aw:.3f}, "
            f"latacc={self.speed_limits.lat_accel_max:.3f}"
        )

    def _cmd_spdlim(self, args):
        if len(args) != 2:
            print("用法: spdlim <param> <value>  (param: vmax/amax/wmax/awmax/latacc)")
            return
        mapping = {
            "vmax": "max_v",
            "amax": "max_a",
            "wmax": "max_w",
            "awmax": "max_aw",
            "latacc": "lat_accel_max",
        }
        key = args[0].lower().strip()
        attr = mapping.get(key)
        if attr is None:
            print(f"未知参数: {key}（可用: vmax, amax, wmax, awmax, latacc）")
            return
        try:
            value = float(args[1])
        except ValueError:
            print(f"参数值无效: {args[1]}")
            return
        if value < 0.0:
            print("参数必须 >= 0")
            return

        updates = {
            "max_v": self.speed_limits.max_v,
            "max_a": self.speed_limits.max_a,
            "max_w": self.speed_limits.max_w,
            "max_aw": self.speed_limits.max_aw,
            "lat_accel_max": self.speed_limits.lat_accel_max,
        }
        updates[attr] = value
        self._replace_speed_limits(
            max_v=updates["max_v"],
            max_a=updates["max_a"],
            max_w=updates["max_w"],
            max_aw=updates["max_aw"],
            lat_accel_max=updates["lat_accel_max"],
        )
        self.redraw()
        print(
            f"速度约束 {key} 已设置为 {value:.3f}；"
            f"当前 vmax={self.speed_limits.max_v:.3f}, "
            f"amax={self.speed_limits.max_a:.3f}, "
            f"wmax={self.speed_limits.max_w:.3f}, "
            f"awmax={self.speed_limits.max_aw:.3f}, "
            f"latacc={self.speed_limits.lat_accel_max:.3f}"
        )

    def _cmd_ispdlim(self, args):
        if len(args) != 2:
            print("用法: ispdlim <segment_id> <vmax|off>  (例如: ispdlim 1 2.5 表示 P1→P2)")
            return
        try:
            seg_id = int(args[0])
        except ValueError:
            print("segment_id 格式无效。示例: ispdlim 1 2.5")
            return
        if seg_id < 1 or seg_id >= len(self.points):
            print(f"[错误] 段索引越界：{seg_id}（当前合法范围 1~{max(0, len(self.points) - 1)}）")
            return

        value_token = args[1].strip().lower()
        if value_token in ("off", "none", "null", "clear"):
            self._set_interval_speed_limit(seg_id - 1, None)
            self.redraw()
            print(f"已清除 P{seg_id}→P{seg_id + 1} 的单段限速。")
            return
        try:
            value = float(args[1])
        except ValueError:
            print("vmax 数值无效。示例: ispdlim 1 2.5 或 ispdlim 1 off")
            return
        if value < 0.0:
            print("vmax 必须 >= 0")
            return
        self._set_interval_speed_limit(seg_id - 1, value)
        self.redraw()
        print(f"P{seg_id}→P{seg_id + 1} 段 vmax 已设置为 {value:.3f} m/s")

    def _cmd_showpath(self, args):
        if len(args) != 1 or args[0].lower() not in ("on", "off"):
            print("用法: showpath on/off")
            return
        self.show_path = args[0].lower() == "on"
        self.redraw()
        print(f"路径显示：{'开启' if self.show_path else '关闭'}")

    def _cmd_body(self, args):
        if not args:
            if self.body_length is None or self.body_width is None:
                print("当前车体矩形：关闭")
            else:
                print(f"当前车体矩形：length={self.body_length:.3f}, width={self.body_width:.3f}")
            return

        joined = " ".join(args).strip().lower()
        if joined == "off":
            self.body_length = None
            self.body_width = None
            self.redraw()
            print("车体矩形悬停显示：关闭")
            return

        parts = joined.replace(" ", "").split(",")
        if len(parts) != 2:
            print("用法: body <length>, <width>  或  body off")
            return
        try:
            length = float(parts[0])
            width = float(parts[1])
        except ValueError:
            print("数值格式无效。示例: body 1.0, 1.0")
            return
        if length <= 0.0 or width <= 0.0:
            print("length 和 width 必须 > 0")
            return
        self.body_length = length
        self.body_width = width
        self.redraw()
        print(f"车体矩形已设置：length={self.body_length:.3f}, width={self.body_width:.3f}")

    def _cmd_save(self, args):
        if len(args) != 1:
            print("用法: save <文件>")
            return
        try:
            out = dump_session(
                args[0],
                self.points,
                self.path_density,
                self.show_path,
                self.speed_limits,
                solver=self.solver,
                body_size=(
                    None
                    if self.body_length is None or self.body_width is None
                    else (self.body_length, self.body_width)
                ),
            )
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
        loaded_solver = str(settings.get("solver", "coupled")).strip().lower()
        self.solver = "coupled" if loaded_solver in ("coupled", "legacy", "toppra") else "coupled"
        body_size = settings.get("body_size", None)
        if (
            isinstance(body_size, tuple)
            and len(body_size) == 2
            and body_size[0] is not None
            and body_size[1] is not None
        ):
            self.body_length = float(body_size[0])
            self.body_width = float(body_size[1])
        else:
            self.body_length = None
            self.body_width = None
        loaded_limits = settings.get("speed_limits", SpeedLimits())
        if not isinstance(loaded_limits, SpeedLimits):
            loaded_limits = SpeedLimits()
        self.speed_limits = loaded_limits
        self.redraw()
        print(
            f"[加载] 会话已加载: {payload.get('path')}  (路径点数={len(self.points)}, "
            f"密度={self.path_density:.2f}, 显示路径={self.show_path}, 求解器={self.solver}, "
            f"vmax={self.speed_limits.max_v:.3f}, amax={self.speed_limits.max_a:.3f}, "
            f"wmax={self.speed_limits.max_w:.3f}, awmax={self.speed_limits.max_aw:.3f}, "
            f"latacc={self.speed_limits.lat_accel_max:.3f}, "
            f"车体={'off' if self.body_length is None else f'{self.body_length:.3f}x{self.body_width:.3f}'})"
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
            import canvas as canvas_module
            out = canvas_module.export_path_cpp(
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
