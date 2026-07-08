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
    theta: float | None
    vx: float | None = None
    vy: float | None = None
    speed: float | None = None
    vw: float | None = None


@dataclass
class Obstacle:
    kind: str
    x: float
    y: float
    theta: float = 0.0
    w: float = 1.0
    h: float = 1.0
    r: float = 0.5


@dataclass(frozen=True)
class SpeedLimits:
    max_v: float = 1.0
    max_a: float = 1.0
    max_w: float = 1.0
    max_aw: float = 1.0
    max_jk: float = 5.0
    lat_accel_max: float = 0.0
    interval_speed_limits: tuple[tuple[int, float], ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            "interval_speed_limits",
            _coerce_interval_speed_limits(self.interval_speed_limits),
        )


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
            "solver": "coupled",
        },
    )


def _normalize_solver_name(solver: str | None) -> str:
    raw = "coupled" if solver is None else str(solver).strip().lower()
    if raw not in ("legacy", "toppra", "coupled"):
        raise ValueError(f"unknown solver: {solver}; expected one of: coupled, legacy, toppra")
    return "coupled"


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


def _format_optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _validate_endpoint_theta(pts: Sequence[Waypoint]) -> None:
    if len(pts) >= 2 and (pts[0].theta is None or pts[-1].theta is None):
        raise ValueError("start and end waypoints must have theta; only intermediate waypoint theta can be empty")


def _interpolate_waypoint_theta(pts: Sequence[Waypoint], chord_t: np.ndarray) -> np.ndarray:
    anchors = [i for i, p in enumerate(pts) if p.theta is not None]
    if not anchors:
        return np.zeros(len(pts), dtype=float)
    if anchors[0] != 0:
        anchors.insert(0, 0)
    if anchors[-1] != len(pts) - 1:
        anchors.append(len(pts) - 1)

    theta_vals = np.zeros(len(pts), dtype=float)
    anchor_theta = unwrap_shortest([0.0 if pts[i].theta is None else float(pts[i].theta) for i in anchors])
    for k, idx in enumerate(anchors):
        theta_vals[idx] = anchor_theta[k]
    for k in range(len(anchors) - 1):
        i0 = anchors[k]
        i1 = anchors[k + 1]
        dt = max(float(chord_t[i1] - chord_t[i0]), 1e-9)
        for i in range(i0 + 1, i1):
            alpha = (float(chord_t[i] - chord_t[i0]) / dt)
            theta_vals[i] = theta_vals[i0] + alpha * (theta_vals[i1] - theta_vals[i0])
    return theta_vals


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
        if len(vals) < 2:
            raise ValueError("waypoint must contain at least (x, y)")
        out.append(
            Waypoint(
                x=float(vals[0]),
                y=float(vals[1]),
                theta=float(vals[2]) if len(vals) > 2 and vals[2] is not None else None,
                vx=float(vals[3]) if len(vals) > 3 and vals[3] is not None else None,
                vy=float(vals[4]) if len(vals) > 4 and vals[4] is not None else None,
                speed=float(vals[5]) if len(vals) > 5 and vals[5] is not None else None,
                vw=float(vals[6]) if len(vals) > 6 and vals[6] is not None else None,
            )
        )
    return out


def time_parameterize(
    samples: PathSamples,
    waypoints: Iterable[Waypoint | Sequence[float]],
    limits: SpeedLimits,
    solver: str = "coupled",
) -> PathSamples:
    if samples.x.size == 0:
        return samples
    pts = _coerce_waypoints(waypoints)

    _normalize_solver_name(solver)
    from speed_solver_coupled import solve_coupled_profile

    solved = solve_coupled_profile(
        s=samples.s,
        x=samples.x,
        y=samples.y,
        waypoints=pts,
        waypoint_sample_indices=samples.meta.get("waypoint_sample_indices", None),
        max_v=limits.max_v,
        max_a=limits.max_a,
        max_w=limits.max_w,
        max_aw=limits.max_aw,
        lat_accel_max=limits.lat_accel_max,
        interval_speed_limits=limits.interval_speed_limits,
    )

    meta = dict(samples.meta)
    meta.update(
        {
            "total_time": float(solved["t"][-1]) if len(solved["t"]) else 0.0,
            "peak_v": float(np.max(solved["v_lin"])) if len(solved["v_lin"]) else 0.0,
            "peak_w": float(np.max(np.abs(solved["w"]))) if len(solved["w"]) else 0.0,
            "peak_aw": float(solved["meta"].get("peak_aw", 0.0)),
            "angular_constrained": bool(solved["meta"].get("angular_constrained", False)),
            "constraint_clipped": bool(solved["meta"].get("constraint_clipped", False)),
            "segment_min_heading_times": solved["meta"].get("segment_min_heading_times", []),
            "segment_actual_times": solved["meta"].get("segment_actual_times", []),
            "heading_anchor_waypoint_indices": solved["meta"].get("heading_anchor_waypoint_indices", []),
            "solver": "coupled",
        }
    )

    return PathSamples(
        x=samples.x,
        y=samples.y,
        theta=np.asarray(solved["theta"], dtype=float),
        s=samples.s,
        t=np.asarray(solved["t"], dtype=float),
        xdot=np.asarray(solved["xdot"], dtype=float),
        ydot=np.asarray(solved["ydot"], dtype=float),
        w=np.asarray(solved["w"], dtype=float),
        v_lin=np.asarray(solved["v_lin"], dtype=float),
        meta=meta,
    )


def resample_path_by_max_dt(samples: PathSamples, max_dt: float | None) -> PathSamples:
    if max_dt is None:
        return samples
    max_dt = float(max_dt)
    if max_dt <= 0.0:
        raise ValueError("max_dt must be > 0 or None")
    if samples.x.size < 2 or samples.t.size != samples.x.size:
        return samples

    old_n = int(samples.x.size)
    new_values: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "theta": [],
        "s": [],
        "t": [],
        "xdot": [],
        "ydot": [],
        "w": [],
        "v_lin": [],
    }
    old_to_new: dict[int, int] = {0: 0}

    def append_sample(i: int, alpha: float):
        alpha = float(alpha)
        theta0 = float(samples.theta[i])
        theta1 = float(samples.theta[i + 1])
        dtheta = float(wrap_angle(theta1 - theta0))
        new_values["x"].append(float(samples.x[i] + alpha * (samples.x[i + 1] - samples.x[i])))
        new_values["y"].append(float(samples.y[i] + alpha * (samples.y[i + 1] - samples.y[i])))
        new_values["theta"].append(float(wrap_angle(theta0 + alpha * dtheta)))
        new_values["s"].append(float(samples.s[i] + alpha * (samples.s[i + 1] - samples.s[i])))
        new_values["t"].append(float(samples.t[i] + alpha * (samples.t[i + 1] - samples.t[i])))
        new_values["xdot"].append(float(samples.xdot[i] + alpha * (samples.xdot[i + 1] - samples.xdot[i])))
        new_values["ydot"].append(float(samples.ydot[i] + alpha * (samples.ydot[i + 1] - samples.ydot[i])))
        new_values["w"].append(float(samples.w[i] + alpha * (samples.w[i + 1] - samples.w[i])))
        new_values["v_lin"].append(float(samples.v_lin[i] + alpha * (samples.v_lin[i + 1] - samples.v_lin[i])))

    for i in range(old_n - 1):
        if i == 0:
            append_sample(i, 0.0)
        dt = max(float(samples.t[i + 1] - samples.t[i]), 0.0)
        pieces = max(1, int(math.ceil(dt / max_dt)))
        for k in range(1, pieces + 1):
            append_sample(i, k / pieces)
        old_to_new[i + 1] = len(new_values["t"]) - 1

    meta = dict(samples.meta)
    waypoint_indices = meta.get("waypoint_sample_indices", None)
    if waypoint_indices:
        meta["waypoint_sample_indices"] = [
            old_to_new.get(int(idx), min(len(new_values["t"]) - 1, max(0, int(idx))))
            for idx in waypoint_indices
        ]
    meta.update(
        {
            "sample_count": len(new_values["t"]),
            "max_dt": max_dt,
            "max_dt_enabled": True,
            "pre_maxdt_sample_count": old_n,
        }
    )

    return PathSamples(
        x=np.asarray(new_values["x"], dtype=float),
        y=np.asarray(new_values["y"], dtype=float),
        theta=np.asarray(new_values["theta"], dtype=float),
        s=np.asarray(new_values["s"], dtype=float),
        t=np.asarray(new_values["t"], dtype=float),
        xdot=np.asarray(new_values["xdot"], dtype=float),
        ydot=np.asarray(new_values["ydot"], dtype=float),
        w=np.asarray(new_values["w"], dtype=float),
        v_lin=np.asarray(new_values["v_lin"], dtype=float),
        meta=meta,
    )


def build_path(
    waypoints: Iterable[Waypoint | Sequence[float]],
    density: float = 20.0,
    speed_limits: SpeedLimits | None = None,
    solver: str = "coupled",
    max_dt: float | None = None,
) -> PathSamples:
    pts = _coerce_waypoints(waypoints)
    n = len(pts)
    if n < 2:
        return _empty_samples()
    _validate_endpoint_theta(pts)

    density = max(float(density), 1.0)
    x = np.array([p.x for p in pts], dtype=float)
    y = np.array([p.y for p in pts], dtype=float)

    seg_chord = np.hypot(np.diff(x), np.diff(y))
    t = np.zeros(n, dtype=float)
    t[1:] = np.cumsum(np.maximum(seg_chord, 1e-6))
    th = _interpolate_waypoint_theta(pts, t)

    dxdt = _estimate_derivatives(x, t)
    dydt = _estimate_derivatives(y, t)
    dthdt = _estimate_derivatives(th, t)
    for i, p in enumerate(pts):
        if p.vx is not None or p.vy is not None:
            vx_val = 0.0 if p.vx is None else float(p.vx)
            vy_val = 0.0 if p.vy is None else float(p.vy)
            v_norm = math.hypot(vx_val, vy_val)
            if v_norm > 1e-9:
                auto_mag = math.hypot(dxdt[i], dydt[i])
                if auto_mag < 1e-9:
                    auto_mag = 0.5
                dxdt[i] = (vx_val / v_norm) * auto_mag
                dydt[i] = (vy_val / v_norm) * auto_mag
        if p.theta is not None and p.vw is not None:
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
    timed = time_parameterize(base, pts, limits, solver=solver)
    return resample_path_by_max_dt(timed, max_dt)


def _coerce_optional_positive_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "off"):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out <= 0.0:
        return default
    return out


def waypoints_to_dict(waypoints: Iterable[Waypoint | Sequence[float]]) -> list[dict]:
    pts = _coerce_waypoints(waypoints)
    out: list[dict] = []
    for p in pts:
        out.append(
            {
                "x": float(p.x),
                "y": float(p.y),
                "theta": _format_optional_float(p.theta),
                "vx": None if p.vx is None else float(p.vx),
                "vy": None if p.vy is None else float(p.vy),
                "speed": None if p.speed is None else float(p.speed),
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
            theta_raw = item.get("theta", None)
            theta = None if theta_raw is None else float(theta_raw)
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

        out.append(Waypoint(x=x, y=y, theta=theta, vx=_opt("vx"), vy=_opt("vy"), speed=_opt("speed"), vw=_opt("vw")))
    return out


def obstacles_to_dict(obstacles: Iterable[Obstacle | dict]) -> list[dict]:
    out: list[dict] = []
    for obs in obstacles:
        if isinstance(obs, Obstacle):
            kind = str(obs.kind).strip().lower()
            x = float(obs.x)
            y = float(obs.y)
            theta = float(obs.theta)
            w = float(obs.w)
            h = float(obs.h)
            r = float(obs.r)
        elif isinstance(obs, dict):
            kind = str(obs.get("kind", obs.get("type", "rect"))).strip().lower()
            x = float(obs["x"])
            y = float(obs["y"])
            theta = float(obs.get("theta", 0.0))
            w = float(obs.get("w", obs.get("width", 1.0)))
            h = float(obs.get("h", obs.get("height", 1.0)))
            r = float(obs.get("r", obs.get("radius", 0.5)))
        else:
            raise ValueError("obstacle must be Obstacle or dict")

        if kind in ("rectangle", "box"):
            kind = "rect"
        elif kind in ("circle", "round"):
            kind = "circle"
        if kind not in ("rect", "circle"):
            raise ValueError(f"unknown obstacle kind: {kind}")

        item = {"kind": kind, "x": x, "y": y}
        if kind == "rect":
            item.update({"theta": theta, "w": w, "h": h})
        else:
            item.update({"r": r})
        out.append(item)
    return out


def obstacles_from_dict(items: Sequence[dict] | None) -> list[Obstacle]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("'obstacles' must be a list")
    out: list[Obstacle] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"obstacle #{idx} must be an object")
        try:
            kind = str(item.get("kind", item.get("type", "rect"))).strip().lower()
            if kind in ("rectangle", "box"):
                kind = "rect"
            elif kind in ("circle", "round"):
                kind = "circle"
            x = float(item["x"])
            y = float(item["y"])
            theta = float(item.get("theta", 0.0))
            w = float(item.get("w", item.get("width", 1.0)))
            h = float(item.get("h", item.get("height", 1.0)))
            r = float(item.get("r", item.get("radius", 0.5)))
        except KeyError as e:
            raise ValueError(f"obstacle #{idx} missing key: {e.args[0]}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"obstacle #{idx} has invalid numeric field") from e
        if kind not in ("rect", "circle"):
            raise ValueError(f"obstacle #{idx} has invalid kind: {kind}")
        if kind == "rect" and (w <= 0.0 or h <= 0.0):
            raise ValueError(f"obstacle #{idx} rectangle w/h must be > 0")
        if kind == "circle" and r <= 0.0:
            raise ValueError(f"obstacle #{idx} circle r must be > 0")
        out.append(Obstacle(kind=kind, x=x, y=y, theta=theta, w=w, h=h, r=r))
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
            lat_accel_max=float(speed_limits.get("lat_accel_max", speed_limits.get("turn_penalty", 0.0))),
            interval_speed_limits=_coerce_interval_speed_limits(speed_limits.get("interval_speed_limits", ())),
        )
    raise ValueError("speed_limits must be SpeedLimits/dict/None")


def _coerce_interval_speed_limits(raw) -> tuple[tuple[int, float], ...]:
    if raw is None:
        return ()
    out: list[tuple[int, float]] = []
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, (list, tuple)):
        iterator = raw
    else:
        return ()

    for item in iterator:
        try:
            if isinstance(item, dict):
                seg_idx = int(item.get("segment", item.get("segment_index", item.get("start", 0))))
                vmax = float(item.get("vmax", item.get("max_v")))
            else:
                seg_idx = int(item[0])
                vmax = float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if seg_idx >= 0 and vmax >= 0.0:
            out.append((seg_idx, vmax))
    return tuple(sorted(out, key=lambda pair: pair[0]))


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
    if samples.meta.get("max_dt_enabled", False):
        lines.append(f"// max_dt(s): {float(samples.meta.get('max_dt', 0.0)):.6f}")
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
        t_sec = float(samples.t[i])
        lines.append(
            f"        {{{{{px:.6f}f, {py:.6f}f}}, {yaw:.6f}f, {{{vx:.6f}f, {vy:.6f}f}}, {w:.6f}f, {t_sec:.6f}f}},"
        )
    for _ in range(sample_count, capacity):
        lines.append("        {{ {0.000000f, 0.000000f}, 0.000000f, {0.000000f, 0.000000f}, 0.000000f, 0.000000f }},")
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
    solver: str = "coupled",
    body_size: tuple[float, float] | None = None,
    max_dt: float | None = None,
    obstacles: Iterable[Obstacle | dict] | None = None,
) -> Path:
    p = _normalize_json_path(file_path)
    limits = _coerce_speed_limits(speed_limits)
    payload = {
        "format_version": 1,
        "waypoints": waypoints_to_dict(waypoints),
        "settings": {
            "density": float(density),
            "max_dt": _coerce_optional_positive_float(max_dt, None),
            "showpath": bool(showpath),
            "solver": _normalize_solver_name(solver),
            "speed_limits": {
                "max_v": float(limits.max_v),
                "max_a": float(limits.max_a),
                "max_w": float(limits.max_w),
                "max_aw": float(limits.max_aw),
                "max_jk": float(limits.max_jk),
                "lat_accel_max": float(limits.lat_accel_max),
                "interval_speed_limits": [
                    {"segment": int(seg_idx), "vmax": float(vmax)}
                    for seg_idx, vmax in limits.interval_speed_limits
                ],
            },
        },
    }
    if obstacles is not None:
        payload["obstacles"] = obstacles_to_dict(obstacles)
    if body_size is not None:
        payload["settings"]["body_size"] = {
            "length": float(body_size[0]),
            "width": float(body_size[1]),
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
    obstacles = obstacles_from_dict(payload.get("obstacles", []))
    settings = payload.get("settings", {})
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")

    density = settings.get("density", 20.0)
    max_dt = _coerce_optional_positive_float(settings.get("max_dt", None), None)
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
        lat_accel_max=float(raw_limits.get("lat_accel_max", raw_limits.get("turn_penalty", 0.0))),
        interval_speed_limits=_coerce_interval_speed_limits(raw_limits.get("interval_speed_limits", ())),
    )
    solver = _normalize_solver_name(settings.get("solver", "legacy"))
    body_cfg = settings.get("body_size", None)
    body_size = None
    if isinstance(body_cfg, dict):
        try:
            body_l = float(body_cfg.get("length", 0.0))
            body_w = float(body_cfg.get("width", 0.0))
            if body_l > 0.0 and body_w > 0.0:
                body_size = (body_l, body_w)
        except (TypeError, ValueError):
            body_size = None

    return {
        "path": p,
        "waypoints": points,
        "obstacles": obstacles,
        "settings": {
            "density": density,
            "max_dt": max_dt,
            "showpath": showpath,
            "solver": solver,
            "speed_limits": limits,
            "body_size": body_size,
        },
    }
