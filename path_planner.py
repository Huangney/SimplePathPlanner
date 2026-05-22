# -*- coding: utf-8 -*-
"""
Path planner core module.

This module is UI-agnostic and only handles geometric path generation:
  - Waypoint definition
  - Decoupled position/theta interpolation
  - Shortest-angle unwrap/wrap
  - Adaptive sampling by segment length
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
from pathlib import Path
import json
import math
import numpy as np


@dataclass
class Waypoint:
    x: float
    y: float
    theta: float
    vx: float | None = None
    vy: float | None = None
    vw: float | None = None


@dataclass
class PathSamples:
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    s: np.ndarray
    meta: dict


def _empty_samples() -> PathSamples:
    return PathSamples(
        x=np.array([], dtype=float),
        y=np.array([], dtype=float),
        theta=np.array([], dtype=float),
        s=np.array([], dtype=float),
        meta={"segments": 0, "total_length": 0.0, "sample_count": 0},
    )


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


def build_path(waypoints: Iterable[Waypoint | Sequence[float]], density: float = 20.0) -> PathSamples:
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
        if p.vx is not None:
            dxdt[i] = float(p.vx)
        if p.vy is not None:
            dydt[i] = float(p.vy)
        if p.vw is not None:
            dthdt[i] = float(p.vw)

    xs: List[float] = []
    ys: List[float] = []
    ths: List[float] = []

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

    x_arr = np.array(xs, dtype=float)
    y_arr = np.array(ys, dtype=float)
    th_arr = wrap_angle(np.array(ths, dtype=float))

    s = np.zeros_like(x_arr)
    if len(x_arr) > 1:
        ds = np.hypot(np.diff(x_arr), np.diff(y_arr))
        s[1:] = np.cumsum(ds)

    meta = {
        "segments": n - 1,
        "total_length": float(s[-1]) if len(s) else 0.0,
        "sample_count": int(len(x_arr)),
    }
    return PathSamples(x=x_arr, y=y_arr, theta=th_arr, s=s, meta=meta)


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


def dump_session(file_path: str | Path, waypoints: Iterable[Waypoint | Sequence[float]], density: float, showpath: bool) -> Path:
    p = _normalize_json_path(file_path)
    payload = {
        "format_version": 1,
        "waypoints": waypoints_to_dict(waypoints),
        "settings": {
            "density": float(density),
            "showpath": bool(showpath),
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

    return {
        "path": p,
        "waypoints": points,
        "settings": {
            "density": density,
            "showpath": showpath,
        },
    }
