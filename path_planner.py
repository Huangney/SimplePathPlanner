# -*- coding: utf-8 -*-
"""
Path planner core module.

This module is UI-agnostic and handles:
  - Geometric path generation by Hermite interpolation
  - Arc-length computation
  - Time parameterization under global speed/acceleration limits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
from pathlib import Path
from datetime import datetime
import json
import math
import re
import numpy as np


@dataclass
class Waypoint:
    x: float
    y: float
    theta: float
    vx: float | None = None
    vy: float | None = None
    vw: float | None = None


@dataclass(frozen=True)
class SpeedLimits:
    max_v: float = 1.0
    max_a: float = 1.0
    max_w: float = 1.0
    max_aw: float = 1.0
    max_jk: float = 5.0


@dataclass
class PathSamples:
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    s: np.ndarray
    t: np.ndarray
    xdot: np.ndarray
    ydot: np.ndarray
    w: np.ndarray
    v_lin: np.ndarray
    meta: dict


def _empty_samples() -> PathSamples:
    return PathSamples(
        x=np.array([], dtype=float),
        y=np.array([], dtype=float),
        theta=np.array([], dtype=float),
        s=np.array([], dtype=float),
        t=np.array([], dtype=float),
        xdot=np.array([], dtype=float),
        ydot=np.array([], dtype=float),
        w=np.array([], dtype=float),
        v_lin=np.array([], dtype=float),
        meta={
            "segments": 0,
            "total_length": 0.0,
            "sample_count": 0,
            "total_time": 0.0,
            "peak_v": 0.0,
            "peak_w": 0.0,
            "constraint_clipped": False,
            "solver": "legacy",
        },
    )


def _normalize_solver_name(solver: str | None) -> str:
    raw = "legacy" if solver is None else str(solver).strip().lower()
    if raw not in ("legacy", "toppra"):
        raise ValueError(f"unknown solver: {solver}; expected one of: legacy, toppra")
    return raw


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_shortest(theta: Sequence[float]) -> np.ndarray:
    if len(theta) == 0:
        return np.array([], dtype=float)
    out = np.zeros(len(theta), dtype=float)
    out[0] = float(theta[0])
    for i in range(1, len(theta)):
        prev = out[i - 1]
        raw = float(theta[i])
        delta = ((raw - prev) + np.pi) % (2.0 * np.pi) - np.pi
        out[i] = prev + delta
    return out


def _estimate_derivatives(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(values)
    d = np.zeros(n, dtype=float)
    if n < 2:
        return d
    if n == 2:
        denom = max(t[1] - t[0], 1e-9)
        slope = (values[1] - values[0]) / denom
        d[0] = slope
        d[1] = slope
        return d
    d[0] = (values[1] - values[0]) / max(t[1] - t[0], 1e-9)
    d[-1] = (values[-1] - values[-2]) / max(t[-1] - t[-2], 1e-9)
    for i in range(1, n - 1):
        denom = max(t[i + 1] - t[i - 1], 1e-9)
        d[i] = (values[i + 1] - values[i - 1]) / denom
    return d


def _hermite(p0: float, p1: float, m0: float, m1: float, dt: float, u: np.ndarray) -> np.ndarray:
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * p0 + h10 * dt * m0 + h01 * p1 + h11 * dt * m1


def _coerce_waypoints(waypoints: Iterable[Waypoint | Sequence[float]]) -> List[Waypoint]:
    out: List[Waypoint] = []
    for p in waypoints:
        if isinstance(p, Waypoint):
            out.append(p)
            continue
        vals = list(p)
        if len(vals) < 3:
            raise ValueError("waypoint must contain at least (x, y, theta)")
        out.append(
            Waypoint(
                x=float(vals[0]),
                y=float(vals[1]),
                theta=float(vals[2]),
                vx=float(vals[3]) if len(vals) > 3 and vals[3] is not None else None,
                vy=float(vals[4]) if len(vals) > 4 and vals[4] is not None else None,
                vw=float(vals[5]) if len(vals) > 5 and vals[5] is not None else None,
            )
        )
    return out


def _anchor_linear_speed_profile(
    pts: List[Waypoint],
    s: np.ndarray,
    limits: SpeedLimits,
    waypoint_sample_indices: Sequence[int] | None = None,
) -> np.ndarray:
    n = len(s)
    v = np.full(n, float(max(limits.max_v, 1e-6)), dtype=float)
    if n > 0:
        v[0] = 0.0
        v[-1] = 0.0

    if len(pts) < 2:
        return v

    if waypoint_sample_indices is not None and len(waypoint_sample_indices) == len(pts):
        sample_idx = [int(max(0, min(n - 1, i))) for i in waypoint_sample_indices]
    else:
        # Fallback for compatibility; precise indices should be provided by build_path.
        x = np.array([p.x for p in pts], dtype=float)
        y = np.array([p.y for p in pts], dtype=float)
        seg_chord = np.hypot(np.diff(x), np.diff(y))
        density_proxy = max(len(s) / max(float(s[-1]), 1e-6), 1.0)
        sample_idx = [0]
        cursor = 0
        for i in range(len(pts) - 1):
            count = max(8, int(math.ceil(max(seg_chord[i], 1e-6) * density_proxy)) + 1)
            cursor += count if i == len(pts) - 2 else (count - 1)
            sample_idx.append(min(cursor, n - 1))

    for i, p in enumerate(pts):
        idx = sample_idx[i]
        if p.vx is not None or p.vy is not None:
            vx = 0.0 if p.vx is None else float(p.vx)
            vy = 0.0 if p.vy is None else float(p.vy)
            target = min(float(np.hypot(vx, vy)), float(limits.max_v))
            v[idx] = min(v[idx], target)

    return np.clip(v, 0.0, float(limits.max_v))


def _forward_backward_speed_limit(v_cap: np.ndarray, s: np.ndarray, max_a: float) -> np.ndarray:
    v = np.clip(v_cap.copy(), 0.0, None)
    n = len(v)
    if n == 0:
        return v
    max_a = max(float(max_a), 1e-6)

    for i in range(1, n):
        ds = max(float(s[i] - s[i - 1]), 0.0)
        v[i] = min(v[i], math.sqrt(max(v[i - 1] * v[i - 1] + 2.0 * max_a * ds, 0.0)))

    for i in range(n - 2, -1, -1):
        ds = max(float(s[i + 1] - s[i]), 0.0)
        v[i] = min(v[i], math.sqrt(max(v[i + 1] * v[i + 1] + 2.0 * max_a * ds, 0.0)))

    return v


def _apply_angular_constraints(v: np.ndarray, s: np.ndarray, theta_unwrapped: np.ndarray, limits: SpeedLimits) -> tuple[np.ndarray, bool]:
    if len(v) < 2:
        return v, False

    clipped = False
    dtheta_ds = np.gradient(theta_unwrapped, s, edge_order=1)
    max_w = max(float(limits.max_w), 1e-6)

    for i in range(len(v)):
        gain = abs(float(dtheta_ds[i]))
        if gain > 1e-9:
            v_cap = max_w / gain
            if v[i] > v_cap:
                v[i] = v_cap
                clipped = True

    v = _forward_backward_speed_limit(v, s, max(float(limits.max_a), 1e-6))

    # Conservative angular acceleration check via finite differences on omega.
    t = np.zeros_like(v)
    for i in range(1, len(v)):
        ds = max(float(s[i] - s[i - 1]), 0.0)
        v_avg = max(float(0.5 * (v[i] + v[i - 1])), 1e-6)
        t[i] = t[i - 1] + ds / v_avg

    omega = dtheta_ds * v
    max_aw = max(float(limits.max_aw), 1e-6)
    for i in range(1, len(v)):
        dt = max(float(t[i] - t[i - 1]), 1e-6)
        aw = abs(float((omega[i] - omega[i - 1]) / dt))
        if aw > max_aw:
            scale = max_aw / aw
            v[i] *= scale
            clipped = True

    return np.clip(v, 0.0, float(limits.max_v)), clipped


def _time_parameterize_legacy(samples: PathSamples, waypoints: Iterable[Waypoint | Sequence[float]], limits: SpeedLimits) -> PathSamples:
    if samples.x.size == 0:
        return samples

    pts = _coerce_waypoints(waypoints)
    x = samples.x
    y = samples.y
    s = samples.s
    th_unwrapped = unwrap_shortest(samples.theta.tolist())

    ds = np.diff(s)
    dx_ds = np.gradient(x, s, edge_order=1)
    dy_ds = np.gradient(y, s, edge_order=1)
    dth_ds = np.gradient(th_unwrapped, s, edge_order=1)

    wp_indices = samples.meta.get("waypoint_sample_indices", None)
    v_cap = _anchor_linear_speed_profile(pts, s, limits, waypoint_sample_indices=wp_indices)
    v = _forward_backward_speed_limit(v_cap, s, limits.max_a)
    v, ang_clipped = _apply_angular_constraints(v, s, th_unwrapped, limits)

    t = np.zeros_like(s)
    for i in range(1, len(s)):
        local_ds = max(float(ds[i - 1]), 0.0)
        v_avg = max(float(0.5 * (v[i] + v[i - 1])), 1e-6)
        t[i] = t[i - 1] + local_ds / v_avg

    xdot = dx_ds * v
    ydot = dy_ds * v
    w = dth_ds * v
    v_lin = np.hypot(xdot, ydot)

    meta = dict(samples.meta)
    meta.update(
        {
            "total_time": float(t[-1]) if len(t) else 0.0,
            "peak_v": float(np.max(v_lin)) if len(v_lin) else 0.0,
            "peak_w": float(np.max(np.abs(w))) if len(w) else 0.0,
            "constraint_clipped": bool(ang_clipped or np.any(v < (v_cap - 1e-9))),
            "solver": "legacy",
        }
    )

    return PathSamples(
        x=samples.x,
        y=samples.y,
        theta=samples.theta,
        s=samples.s,
        t=t,
        xdot=xdot,
        ydot=ydot,
        w=w,
        v_lin=v_lin,
        meta=meta,
    )


def time_parameterize(
    samples: PathSamples,
    waypoints: Iterable[Waypoint | Sequence[float]],
    limits: SpeedLimits,
    solver: str = "legacy",
) -> PathSamples:
    solver_name = _normalize_solver_name(solver)
    if solver_name == "legacy":
        return _time_parameterize_legacy(samples, waypoints, limits)

    if samples.x.size == 0:
        return samples
    pts = _coerce_waypoints(waypoints)

    from speed_solver_toppra import solve_toppra_profile

    wp_indices = samples.meta.get("waypoint_sample_indices", None)
    wp_targets: list[float | None] = []
    for p in pts:
        if p.vx is None and p.vy is None:
            wp_targets.append(None)
        else:
            vx = 0.0 if p.vx is None else float(p.vx)
            vy = 0.0 if p.vy is None else float(p.vy)
            wp_targets.append(float(np.hypot(vx, vy)))

    solved = solve_toppra_profile(
        s=samples.s,
        x=samples.x,
        y=samples.y,
        theta=samples.theta,
        waypoint_sample_indices=wp_indices,
        waypoint_v_targets=wp_targets,
        max_v=limits.max_v,
        max_a=limits.max_a,
        max_w=limits.max_w,
        max_aw=limits.max_aw,
        max_jk=limits.max_jk,
    )

    meta = dict(samples.meta)
    meta.update(
        {
            "total_time": float(solved["t"][-1]) if len(solved["t"]) else 0.0,
            "peak_v": float(np.max(solved["v_lin"])) if len(solved["v_lin"]) else 0.0,
            "peak_w": float(np.max(np.abs(solved["w"]))) if len(solved["w"]) else 0.0,
            "constraint_clipped": bool(solved["meta"].get("constraint_clipped", False)),
            "peak_jerk": float(solved["meta"].get("peak_jerk", 0.0)),
            "jerk_clipped": bool(solved["meta"].get("jerk_clipped", False)),
            "solver": "toppra",
        }
    )

    return PathSamples(
        x=samples.x,
        y=samples.y,
        theta=samples.theta,
        s=samples.s,
        t=np.asarray(solved["t"], dtype=float),
        xdot=np.asarray(solved["xdot"], dtype=float),
        ydot=np.asarray(solved["ydot"], dtype=float),
        w=np.asarray(solved["w"], dtype=float),
        v_lin=np.asarray(solved["v_lin"], dtype=float),
        meta=meta,
    )


def build_path(
    waypoints: Iterable[Waypoint | Sequence[float]],
    density: float = 20.0,
    speed_limits: SpeedLimits | None = None,
    solver: str = "legacy",
) -> PathSamples:
    pts = _coerce_waypoints(waypoints)
    n = len(pts)
    if n < 2:
        return _empty_samples()

    density = max(float(density), 1.0)
    x = np.array([p.x for p in pts], dtype=float)
    y = np.array([p.y for p in pts], dtype=float)
    th = unwrap_shortest([p.theta for p in pts])

    seg_chord = np.hypot(np.diff(x), np.diff(y))
    t = np.zeros(n, dtype=float)
    t[1:] = np.cumsum(np.maximum(seg_chord, 1e-6))

    dxdt = _estimate_derivatives(x, t)
    dydt = _estimate_derivatives(y, t)
    dthdt = _estimate_derivatives(th, t)
    for i, p in enumerate(pts):
        # vx/vy/vw are waypoint target velocities in world frame.
        # If either linear component is provided, treat the missing one as 0.0
        # so users can set/serialize partial vectors (e.g., only vy).
        if p.vx is not None or p.vy is not None:
            dxdt[i] = 0.0 if p.vx is None else float(p.vx)
            dydt[i] = 0.0 if p.vy is None else float(p.vy)
        if p.vw is not None:
            dthdt[i] = float(p.vw)

    xs: List[float] = []
    ys: List[float] = []
    ths: List[float] = []

    waypoint_sample_indices: list[int] = [0]
    built_count = 0
    for i in range(n - 1):
        dt = max(t[i + 1] - t[i], 1e-9)
        approx_len = max(seg_chord[i], 1e-6)
        count = max(8, int(math.ceil(approx_len * density)) + 1)
        if i < n - 2:
            u = np.linspace(0.0, 1.0, count, endpoint=False)
        else:
            u = np.linspace(0.0, 1.0, count, endpoint=True)

        xi = _hermite(x[i], x[i + 1], dxdt[i], dxdt[i + 1], dt, u)
        yi = _hermite(y[i], y[i + 1], dydt[i], dydt[i + 1], dt, u)
        thi = _hermite(th[i], th[i + 1], dthdt[i], dthdt[i + 1], dt, u)

        xs.extend(xi.tolist())
        ys.extend(yi.tolist())
        ths.extend(thi.tolist())
        # Index mapping notes:
        # - Non-last segment uses endpoint=False, so waypoint(i+1) is NOT included yet.
        #   It appears as the first sample of the next segment -> index equals current length.
        # - Last segment uses endpoint=True, so final waypoint is the last appended sample.
        if i < n - 2:
            built_count += count
        else:
            built_count += count - 1
        waypoint_sample_indices.append(max(0, built_count))

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    th_unwrapped = np.array(ths, dtype=float)
    th_arr = wrap_angle(th_unwrapped)

    s = np.zeros_like(x_arr)
    if len(x_arr) > 1:
        ds = np.hypot(np.diff(x_arr), np.diff(y_arr))
        s[1:] = np.cumsum(ds)

    base = PathSamples(
        x=x_arr,
        y=y_arr,
        theta=th_arr,
        s=s,
        t=np.array([], dtype=float),
        xdot=np.array([], dtype=float),
        ydot=np.array([], dtype=float),
        w=np.array([], dtype=float),
        v_lin=np.array([], dtype=float),
        meta={
            "segments": n - 1,
            "total_length": float(s[-1]) if len(s) else 0.0,
            "sample_count": int(len(x_arr)),
            "waypoint_sample_indices": waypoint_sample_indices,
            "solver": _normalize_solver_name(solver),
        },
    )

    limits = speed_limits if speed_limits is not None else SpeedLimits()
    return time_parameterize(base, pts, limits, solver=solver)


def waypoints_to_dict(waypoints: Iterable[Waypoint | Sequence[float]]) -> list[dict]:
    pts = _coerce_waypoints(waypoints)
    out: list[dict] = []
    for p in pts:
        out.append(
            {
                "x": float(p.x),
                "y": float(p.y),
                "theta": float(p.theta),
                "vx": None if p.vx is None else float(p.vx),
                "vy": None if p.vy is None else float(p.vy),
                "vw": None if p.vw is None else float(p.vw),
            }
        )
    return out


def waypoints_from_dict(items: Sequence[dict]) -> list[Waypoint]:
    if not isinstance(items, list):
        raise ValueError("'waypoints' must be a list")
    out: list[Waypoint] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"waypoint #{idx} must be an object")
        try:
            x = float(item["x"])
            y = float(item["y"])
            theta = float(item["theta"])
        except KeyError as e:
            raise ValueError(f"waypoint #{idx} missing key: {e.args[0]}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"waypoint #{idx} has invalid x/y/theta") from e

        def _opt(name: str):
            v = item.get(name, None)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"waypoint #{idx} has invalid {name}") from e

        out.append(Waypoint(x=x, y=y, theta=theta, vx=_opt("vx"), vy=_opt("vy"), vw=_opt("vw")))
    return out


def _normalize_json_path(file_path: str | Path) -> Path:
    p = Path(file_path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    return p


def _coerce_speed_limits(speed_limits: SpeedLimits | dict | None) -> SpeedLimits:
    if speed_limits is None:
        return SpeedLimits()
    if isinstance(speed_limits, SpeedLimits):
        return speed_limits
    if isinstance(speed_limits, dict):
        return SpeedLimits(
            max_v=float(speed_limits.get("max_v", 1.0)),
            max_a=float(speed_limits.get("max_a", 1.0)),
            max_w=float(speed_limits.get("max_w", 1.0)),
            max_aw=float(speed_limits.get("max_aw", 1.0)),
            max_jk=float(speed_limits.get("max_jk", 5.0)),
        )
    raise ValueError("speed_limits must be SpeedLimits/dict/None")


def _sanitize_cpp_identifier(name: str) -> str:
    raw = str(name).strip() if name is not None else ""
    if not raw:
        raw = "GeneratedPath"
    ident = re.sub(r"[^0-9A-Za-z_]", "_", raw)
    if not (ident[0].isalpha() or ident[0] == "_"):
        ident = "_" + ident
    return ident


def export_path_cpp(
    file_path: str | Path,
    samples: PathSamples,
    path_name: str = "GeneratedPath",
    grid_scale: float = 1.0,
    capacity: int | None = None,
) -> Path:
    out = Path(file_path)
    if out.suffix.lower() not in (".hpp", ".h"):
        out = out.with_suffix(".hpp")
    if samples.x.size < 2:
        raise ValueError("not enough path samples; please plan at least two points before export")
    if float(grid_scale) <= 0.0:
        raise ValueError("grid_scale must be > 0")
    sample_count = int(samples.x.size)
    if capacity is None:
        capacity = sample_count
    capacity = int(capacity)
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if sample_count > capacity:
        raise ValueError(f"sample_count exceeds capacity: {sample_count} > {capacity}")

    ident = _sanitize_cpp_identifier(path_name)
    total_length = float(samples.meta.get("total_length", 0.0))
    total_time = float(samples.meta.get("total_time", 0.0))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("#pragma once")
    lines.append('#include "PathChaser.hpp"')
    lines.append("")
    lines.append("// Auto-generated by SimplePathPlanner.")
    lines.append(f"// generated_at: {generated_at}")
    lines.append(f"// sample_count: {sample_count}")
    lines.append(f"// grid_scale(m/grid): {float(grid_scale):.6f}")
    lines.append(f"// total_length(grid): {total_length:.6f}")
    lines.append(f"// total_time(s): {total_time:.6f}")
    lines.append("")
    lines.append(f"static const Path<{capacity}> {ident} = {{")
    lines.append("    {")
    for i in range(sample_count):
        px = float(samples.x[i]) * float(grid_scale)
        py = float(samples.y[i]) * float(grid_scale)
        yaw = float(samples.theta[i])
        vx = float(samples.xdot[i]) * float(grid_scale)
        vy = float(samples.ydot[i]) * float(grid_scale)
        w = float(samples.w[i])
        lines.append(
            f"        {{{{{px:.6f}f, {py:.6f}f}}, {yaw:.6f}f, {{{vx:.6f}f, {vy:.6f}f}}, {w:.6f}f}},"
        )
    for _ in range(sample_count, capacity):
        lines.append("        {{ {0.000000f, 0.000000f}, 0.000000f, {0.000000f, 0.000000f}, 0.000000f }},")
    lines.append("    }")
    lines.append("};")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def dump_session(
    file_path: str | Path,
    waypoints: Iterable[Waypoint | Sequence[float]],
    density: float,
    showpath: bool,
    speed_limits: SpeedLimits | dict | None = None,
    solver: str = "legacy",
) -> Path:
    p = _normalize_json_path(file_path)
    limits = _coerce_speed_limits(speed_limits)
    payload = {
        "format_version": 1,
        "waypoints": waypoints_to_dict(waypoints),
        "settings": {
            "density": float(density),
            "showpath": bool(showpath),
            "solver": _normalize_solver_name(solver),
            "speed_limits": {
                "max_v": float(limits.max_v),
                "max_a": float(limits.max_a),
                "max_w": float(limits.max_w),
                "max_aw": float(limits.max_aw),
                "max_jk": float(limits.max_jk),
            },
        },
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def load_session(file_path: str | Path) -> dict:
    p = _normalize_json_path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("session JSON root must be an object")
    ver = payload.get("format_version", None)
    if ver != 1:
        raise ValueError(f"unsupported format_version: {ver}")
    if "waypoints" not in payload:
        raise ValueError("missing required field: waypoints")

    points = waypoints_from_dict(payload["waypoints"])
    settings = payload.get("settings", {})
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")

    density = settings.get("density", 20.0)
    showpath = settings.get("showpath", True)
    try:
        density = float(density)
    except (TypeError, ValueError):
        density = 20.0
    if density < 1.0:
        density = 20.0
    showpath = bool(showpath)

    raw_limits = settings.get("speed_limits", {})
    if not isinstance(raw_limits, dict):
        raw_limits = {}
    limits = SpeedLimits(
        max_v=float(raw_limits.get("max_v", 1.0)),
        max_a=float(raw_limits.get("max_a", 1.0)),
        max_w=float(raw_limits.get("max_w", 1.0)),
        max_aw=float(raw_limits.get("max_aw", 1.0)),
        max_jk=float(raw_limits.get("max_jk", 5.0)),
    )
    solver = _normalize_solver_name(settings.get("solver", "legacy"))

    return {
        "path": p,
        "waypoints": points,
        "settings": {
            "density": density,
            "showpath": showpath,
            "solver": solver,
            "speed_limits": limits,
        },
    }
