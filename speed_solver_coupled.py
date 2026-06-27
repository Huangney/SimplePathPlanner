# -*- coding: utf-8 -*-
"""Coupled translational/heading time parameterization."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _unwrap_shortest(theta: Sequence[float]) -> np.ndarray:
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


def _forward_backward_speed_limit(v_cap: np.ndarray, s: np.ndarray, max_a: float) -> np.ndarray:
    v = np.clip(v_cap.copy(), 0.0, None)
    if v.size == 0:
        return v
    max_a = max(float(max_a), 1e-9)
    for i in range(1, v.size):
        ds = max(float(s[i] - s[i - 1]), 0.0)
        v[i] = min(v[i], math.sqrt(max(v[i - 1] * v[i - 1] + 2.0 * max_a * ds, 0.0)))
    for i in range(v.size - 2, -1, -1):
        ds = max(float(s[i + 1] - s[i]), 0.0)
        v[i] = min(v[i], math.sqrt(max(v[i + 1] * v[i + 1] + 2.0 * max_a * ds, 0.0)))
    return v


def _integrate_time(s: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = np.zeros_like(s, dtype=float)
    for i in range(1, s.size):
        ds = max(float(s[i] - s[i - 1]), 0.0)
        v_avg = max(float(0.5 * (v[i] + v[i - 1])), 1e-6)
        t[i] = t[i - 1] + ds / v_avg
    return t


def _compute_curvature(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> np.ndarray:
    dx_ds = np.gradient(x, s, edge_order=1)
    dy_ds = np.gradient(y, s, edge_order=1)
    d2x_ds2 = np.gradient(dx_ds, s, edge_order=1)
    d2y_ds2 = np.gradient(dy_ds, s, edge_order=1)
    num = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    denom = np.maximum((dx_ds * dx_ds + dy_ds * dy_ds) ** 1.5, 1e-9)
    return num / denom


def _angular_min_duration(delta: float, max_w: float, max_aw: float) -> float:
    dist = abs(float(delta))
    if dist <= 1e-12:
        return 0.0
    if max_w <= 0.0 or max_aw <= 0.0:
        raise ValueError("nonzero heading change requires max_w > 0 and max_aw > 0")
    ramp_dist = max_w * max_w / max_aw
    if dist <= ramp_dist:
        return 2.0 * math.sqrt(dist / max_aw)
    return dist / max_w + max_w / max_aw


def _sample_angular_zero_endpoint(
    delta: float,
    duration: float,
    max_w: float,
    max_aw: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sign = 1.0 if float(delta) >= 0.0 else -1.0
    dist = abs(float(delta))
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, max(float(duration), 0.0))
    if dist <= 1e-12:
        return np.zeros_like(tau), np.zeros_like(tau)

    active_t = _angular_min_duration(dist, max_w, max_aw)
    active_tau = np.minimum(tau, active_t)
    ramp_t = min(math.sqrt(dist / max_aw), max_w / max_aw)
    ramp_dist = max_aw * ramp_t * ramp_t
    if dist <= ramp_dist + 1e-12:
        cruise_t = 0.0
        peak_w = max_aw * ramp_t
    else:
        peak_w = max_w
        cruise_t = (dist - ramp_dist) / max_w
    cruise_end = ramp_t + cruise_t

    q = np.zeros_like(active_tau)
    w = np.zeros_like(active_tau)
    accel = active_tau < ramp_t
    cruise = (active_tau >= ramp_t) & (active_tau <= cruise_end)
    decel = active_tau > cruise_end

    q[accel] = 0.5 * max_aw * active_tau[accel] ** 2
    w[accel] = max_aw * active_tau[accel]

    q_ramp = 0.5 * max_aw * ramp_t * ramp_t
    q[cruise] = q_ramp + peak_w * (active_tau[cruise] - ramp_t)
    w[cruise] = peak_w

    rem = active_t - active_tau[decel]
    q[decel] = dist - 0.5 * max_aw * rem * rem
    w[decel] = max_aw * rem

    hold = tau >= active_t
    q[hold] = dist
    w[hold] = 0.0
    return sign * q, sign * w


def _sample_angular_cubic(
    delta: float,
    duration: float,
    w0: float,
    w1: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = max(float(duration), 1e-9)
    u = np.clip(np.asarray(tau, dtype=float) / duration, 0.0, 1.0)
    m0 = float(w0) * duration
    m1 = float(w1) * duration
    d = float(delta)

    h10 = u ** 3 - 2.0 * u ** 2 + u
    h01 = -2.0 * u ** 3 + 3.0 * u ** 2
    h11 = u ** 3 - u ** 2
    q = h10 * m0 + h01 * d + h11 * m1

    dq_du = (3.0 * u ** 2 - 4.0 * u + 1.0) * m0
    dq_du += (-6.0 * u ** 2 + 6.0 * u) * d
    dq_du += (3.0 * u ** 2 - 2.0 * u) * m1
    w = dq_du / duration

    d2q_du2 = (6.0 * u - 4.0) * m0 + (-12.0 * u + 6.0) * d + (6.0 * u - 2.0) * m1
    aw = d2q_du2 / (duration * duration)
    return q, w, aw


def _angular_cubic_min_duration(delta: float, w0: float, w1: float, max_w: float, max_aw: float) -> float:
    if abs(w0) > max_w + 1e-9 or abs(w1) > max_w + 1e-9:
        raise ValueError("waypoint vw exceeds max_w")
    if max_w <= 0.0 or max_aw <= 0.0:
        if abs(delta) > 1e-12 or abs(w0) > 1e-12 or abs(w1) > 1e-12:
            raise ValueError("heading motion requires max_w > 0 and max_aw > 0")
        return 0.0

    low = 1e-6
    high = max(_angular_min_duration(delta, max_w, max_aw), abs(delta) / max_w, 1e-3)
    high = max(high, abs(w0) / max_aw, abs(w1) / max_aw)
    probe = np.linspace(0.0, 1.0, 120)

    def ok(duration: float) -> bool:
        tau = probe * duration
        _, wi, awi = _sample_angular_cubic(delta, duration, w0, w1, tau)
        return bool(np.max(np.abs(wi)) <= max_w + 1e-8 and np.max(np.abs(awi)) <= max_aw + 1e-8)

    while not ok(high):
        high *= 1.5
        if high > 1e6:
            raise ValueError("unable to find feasible heading duration")
    for _ in range(60):
        mid = 0.5 * (low + high)
        if ok(mid):
            high = mid
        else:
            low = mid
    return high


def _segment_heading_min_time(delta: float, w0: float, w1: float, max_w: float, max_aw: float) -> float:
    if abs(w0) <= 1e-12 and abs(w1) <= 1e-12:
        return _angular_min_duration(delta, max_w, max_aw)
    return _angular_cubic_min_duration(delta, w0, w1, max_w, max_aw)


def _sample_heading_segment(
    delta: float,
    duration: float,
    w0: float,
    w1: float,
    max_w: float,
    max_aw: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(w0) <= 1e-12 and abs(w1) <= 1e-12:
        q, w = _sample_angular_zero_endpoint(delta, duration, max_w, max_aw, tau)
        aw = np.gradient(w, np.maximum(tau, 0.0), edge_order=1) if tau.size > 2 else np.zeros_like(w)
        return q, w, aw
    q, w, aw = _sample_angular_cubic(delta, duration, w0, w1, tau)
    return q, w, aw


def _build_initial_speed_cap(
    *,
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    waypoints: Sequence[object],
    waypoint_sample_indices: Sequence[int],
    max_v: float,
    lat_accel_max: float,
    interval_speed_limits: Sequence[tuple[int, float]] = (),
) -> tuple[np.ndarray, bool]:
    v_cap = np.full(s.size, max(float(max_v), 1e-9), dtype=float)
    v_cap[0] = 0.0
    v_cap[-1] = 0.0

    for p, idx in zip(waypoints, waypoint_sample_indices):
        target = getattr(p, "speed", None)
        if target is not None:
            v_cap[int(idx)] = min(v_cap[int(idx)], max(float(target), 0.0), float(max_v))

    for seg_idx, target in interval_speed_limits:
        try:
            seg_idx = int(seg_idx)
            target_v = max(float(target), 0.0)
        except (TypeError, ValueError):
            continue
        if seg_idx < 0 or seg_idx >= len(waypoint_sample_indices) - 1:
            continue
        i0 = int(waypoint_sample_indices[seg_idx])
        i1 = int(waypoint_sample_indices[seg_idx + 1])
        if i1 < i0:
            i0, i1 = i1, i0
        v_cap[i0:i1 + 1] = np.minimum(v_cap[i0:i1 + 1], min(target_v, float(max_v)))
        v_cap[0] = 0.0
        v_cap[-1] = 0.0

    curvature_clipped = False
    if lat_accel_max > 0.0:
        kappa = _compute_curvature(x, y, s)
        valid = kappa > 1e-9
        curv_cap = np.full_like(v_cap, np.inf)
        curv_cap[valid] = np.sqrt(max(float(lat_accel_max), 1e-9) / kappa[valid])
        curvature_clipped = bool(np.any(curv_cap[valid] + 1e-9 < v_cap[valid]))
        v_cap = np.minimum(v_cap, curv_cap)
        v_cap[0] = 0.0
        v_cap[-1] = 0.0
    return np.clip(v_cap, 0.0, float(max_v)), curvature_clipped


def solve_coupled_profile(
    *,
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    waypoints: Sequence[object],
    waypoint_sample_indices: Sequence[int] | None,
    max_v: float,
    max_a: float,
    max_w: float,
    max_aw: float,
    lat_accel_max: float = 0.0,
    interval_speed_limits: Sequence[tuple[int, float]] = (),
) -> dict:
    if s.size < 2:
        z = np.zeros_like(s)
        return {"t": z, "xdot": z, "ydot": z, "theta": z, "w": z, "v_lin": z, "meta": {}}
    if waypoint_sample_indices is None or len(waypoint_sample_indices) != len(waypoints):
        raise ValueError("coupled solver requires waypoint_sample_indices for every waypoint")
    if len(waypoints) < 2:
        raise ValueError("coupled solver requires at least two waypoints")
    if getattr(waypoints[0], "theta") is None or getattr(waypoints[-1], "theta") is None:
        raise ValueError("start and end waypoints must have theta")

    wp_indices = [int(max(0, min(s.size - 1, i))) for i in waypoint_sample_indices]
    anchor_wp_indices = [i for i, p in enumerate(waypoints) if getattr(p, "theta") is not None]
    if anchor_wp_indices[0] != 0:
        anchor_wp_indices.insert(0, 0)
    if anchor_wp_indices[-1] != len(waypoints) - 1:
        anchor_wp_indices.append(len(waypoints) - 1)
    anchor_sample_indices = [wp_indices[i] for i in anchor_wp_indices]
    theta_wp = _unwrap_shortest([float(getattr(waypoints[i], "theta")) for i in anchor_wp_indices])
    w_wp = np.array(
        [
            0.0 if getattr(waypoints[i], "vw", None) is None else float(getattr(waypoints[i], "vw"))
            for i in anchor_wp_indices
        ],
        dtype=float,
    )
    if np.any(np.abs(w_wp) > max(float(max_w), 0.0) + 1e-9):
        raise ValueError("waypoint vw exceeds max_w")

    seg_min_times = np.zeros(len(anchor_wp_indices) - 1, dtype=float)
    for seg in range(len(anchor_wp_indices) - 1):
        dtheta = float(theta_wp[seg + 1] - theta_wp[seg])
        seg_min_times[seg] = _segment_heading_min_time(dtheta, w_wp[seg], w_wp[seg + 1], float(max_w), float(max_aw))

    v_cap, curvature_clipped = _build_initial_speed_cap(
        s=s,
        x=x,
        y=y,
        waypoints=waypoints,
        waypoint_sample_indices=wp_indices,
        max_v=max_v,
        lat_accel_max=lat_accel_max,
        interval_speed_limits=interval_speed_limits,
    )

    angular_constrained = False
    for _ in range(80):
        v = _forward_backward_speed_limit(v_cap, s, max_a)
        t = _integrate_time(s, v)
        changed = False
        for seg, min_dt in enumerate(seg_min_times):
            if min_dt <= 1e-12:
                continue
            i0 = anchor_sample_indices[seg]
            i1 = anchor_sample_indices[seg + 1]
            if i1 <= i0:
                continue
            actual_dt = max(float(t[i1] - t[i0]), 0.0)
            if actual_dt + 1e-7 >= min_dt:
                continue
            seg_len = max(float(s[i1] - s[i0]), 1e-9)
            cap = max(seg_len / min_dt * 0.96, 1e-6)
            old = v_cap[i0:i1 + 1].copy()
            v_cap[i0:i1 + 1] = np.minimum(v_cap[i0:i1 + 1], cap)
            v_cap[0] = 0.0
            v_cap[-1] = 0.0
            changed = changed or bool(np.any(v_cap[i0:i1 + 1] < old - 1e-12))
            angular_constrained = True
        if not changed:
            break
    else:
        raise ValueError("coupled retiming failed to satisfy heading durations")

    v = _forward_backward_speed_limit(v_cap, s, max_a)
    t = _integrate_time(s, v)

    theta_u = np.zeros_like(s, dtype=float)
    w = np.zeros_like(s, dtype=float)
    aw = np.zeros_like(s, dtype=float)
    for seg in range(len(anchor_wp_indices) - 1):
        i0 = anchor_sample_indices[seg]
        i1 = anchor_sample_indices[seg + 1]
        if i1 <= i0:
            continue
        duration = max(float(t[i1] - t[i0]), 1e-9)
        tau = t[i0:i1 + 1] - t[i0]
        dtheta = float(theta_wp[seg + 1] - theta_wp[seg])
        q, wi, awi = _sample_heading_segment(dtheta, duration, w_wp[seg], w_wp[seg + 1], max_w, max_aw, tau)
        theta_u[i0:i1 + 1] = theta_wp[seg] + q
        w[i0:i1 + 1] = wi
        aw[i0:i1 + 1] = awi

    dx_ds = np.gradient(x, s, edge_order=1)
    dy_ds = np.gradient(y, s, edge_order=1)
    xdot = dx_ds * v
    ydot = dy_ds * v
    v_lin = np.hypot(xdot, ydot)

    return {
        "t": t,
        "xdot": xdot,
        "ydot": ydot,
        "theta": _wrap_angle(theta_u),
        "w": w,
        "v_lin": v_lin,
        "meta": {
            "solver": "coupled",
            "constraint_clipped": bool(curvature_clipped or angular_constrained or np.any(v + 1e-9 < max(float(max_v), 1e-9))),
            "angular_constrained": bool(angular_constrained),
            "segment_min_heading_times": seg_min_times.tolist(),
            "segment_actual_times": [
                float(t[anchor_sample_indices[i + 1]] - t[anchor_sample_indices[i]])
                for i in range(len(anchor_sample_indices) - 1)
            ],
            "heading_anchor_waypoint_indices": anchor_wp_indices,
            "peak_aw": float(np.max(np.abs(aw))) if aw.size else 0.0,
        },
    }
