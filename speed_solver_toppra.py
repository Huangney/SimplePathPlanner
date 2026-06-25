# -*- coding: utf-8 -*-
"""
TOPPRA-style time parameterization helper.

This module is intentionally UI-agnostic and only performs timing solve on a
pre-sampled geometric path.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Sequence

import numpy as np


def _ensure_toppra_available() -> None:
    if importlib.util.find_spec("toppra") is None:
        raise RuntimeError(
            "solver 'toppra' requires package 'toppra'. Install with: pip install toppra"
        )


def _anchor_speed_caps(
    x_cap: np.ndarray,
    waypoint_sample_indices: Sequence[int] | None,
    waypoint_v_targets: Sequence[float | None],
) -> np.ndarray:
    out = x_cap.copy()
    if waypoint_sample_indices is None:
        return out
    for i, tgt in enumerate(waypoint_v_targets):
        if tgt is None:
            continue
        if i < 0 or i >= len(waypoint_sample_indices):
            continue
        idx = int(waypoint_sample_indices[i])
        if idx < 0 or idx >= out.size:
            continue
        # x_cap is the squared speed domain, so convert the waypoint speed
        # target to v^2 before applying it.
        out[idx] = min(out[idx], max(float(tgt), 0.0) ** 2)
    return out


def _reachability_pass(
    s: np.ndarray,
    x_cap: np.ndarray,
    a_up: np.ndarray,
    a_lo: np.ndarray,
    x0: float,
    xN: float,
) -> np.ndarray:
    n = len(s)
    eps = 1e-9
    x_lo_fwd = np.zeros(n, dtype=float)
    x_hi_fwd = np.zeros(n, dtype=float)
    x_lo_fwd[0] = max(0.0, min(float(x_cap[0]), float(x0)))
    x_hi_fwd[0] = x_lo_fwd[0]

    for i in range(n - 1):
        ds = max(float(s[i + 1] - s[i]), eps)
        lo_next = max(0.0, x_lo_fwd[i] + 2.0 * a_lo[i] * ds)
        hi_next = min(float(x_cap[i + 1]), x_hi_fwd[i] + 2.0 * a_up[i] * ds)
        if hi_next + eps < lo_next:
            mid = max(0.0, min(float(x_cap[i + 1]), 0.5 * (lo_next + hi_next)))
            lo_next = mid
            hi_next = mid
        x_lo_fwd[i + 1] = lo_next
        x_hi_fwd[i + 1] = hi_next

    x_lo = x_lo_fwd.copy()
    x_hi = x_hi_fwd.copy()
    x_lo[-1] = max(0.0, min(float(x_cap[-1]), float(xN)))
    x_hi[-1] = x_lo[-1]

    for i in range(n - 2, -1, -1):
        ds = max(float(s[i + 1] - s[i]), eps)
        low = max(x_lo_fwd[i], max(0.0, x_lo[i + 1] - 2.0 * a_up[i] * ds))
        high = min(x_hi_fwd[i], min(float(x_cap[i]), x_hi[i + 1] - 2.0 * a_lo[i] * ds))
        if high + eps < low:
            mid = max(0.0, min(float(x_cap[i]), 0.5 * (low + high)))
            low = mid
            high = mid
        x_lo[i] = low
        x_hi[i] = high

    return 0.5 * (x_lo + x_hi)


def _smooth_jerk(v: np.ndarray, t: np.ndarray, max_jk: float) -> np.ndarray:
    if len(v) < 4 or max_jk <= 0.0:
        return v
    out = v.copy()
    max_jk = float(max_jk)
    for _ in range(3):
        a = np.gradient(out, t, edge_order=1)
        j = np.gradient(a, t, edge_order=1)
        over = np.where(np.abs(j) > max_jk)[0]
        if over.size == 0:
            break
        for idx in over:
            i0 = max(0, idx - 1)
            i1 = min(len(out) - 1, idx + 1)
            out[idx] = 0.5 * (out[i0] + out[i1])
        out = np.clip(out, 0.0, None)
    return out


def _fill_internal_speed_holes(v: np.ndarray, *, floor: float = 1e-6) -> np.ndarray:
    if len(v) < 3:
        return v
    out = v.copy()
    i = 1
    while i < len(out) - 1:
        if out[i] > floor:
            i += 1
            continue
        j = i
        while j < len(out) - 1 and out[j] <= floor:
            j += 1
        if out[i - 1] > floor and out[j] > floor:
            out[i:j] = np.linspace(out[i - 1], out[j], j - i + 2)[1:-1]
        i = j + 1
    return out


def solve_toppra_profile(
    *,
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    waypoint_sample_indices: Sequence[int] | None,
    waypoint_v_targets: Sequence[float | None],
    max_v: float,
    max_a: float,
    max_w: float,
    max_aw: float,
    max_jk: float,
    turn_penalty: float = 1.0,
    lat_accel_max: float = 0.0,
) -> dict:
    _ensure_toppra_available()
    if s.size < 2:
        return {
            "t": np.zeros_like(s),
            "xdot": np.zeros_like(s),
            "ydot": np.zeros_like(s),
            "w": np.zeros_like(s),
            "v_lin": np.zeros_like(s),
            "meta": {"constraint_clipped": False, "peak_jerk": 0.0, "jerk_clipped": False},
        }

    eps = 1e-9
    th_u = np.unwrap(theta)
    dx_ds = np.gradient(x, s, edge_order=1)
    dy_ds = np.gradient(y, s, edge_order=1)
    dth_ds = np.gradient(th_u, s, edge_order=1)

    lin_gain = np.hypot(dx_ds, dy_ds)
    lin_gain = np.maximum(lin_gain, eps)
    x_cap = (max(float(max_v), eps) / lin_gain) ** 2

    x_cap = _anchor_speed_caps(x_cap, waypoint_sample_indices, waypoint_v_targets)
    x_cap = np.maximum(x_cap, 0.0)

    if lat_accel_max > 0.0:
        d2x_ds2 = np.gradient(dx_ds, s, edge_order=1)
        d2y_ds2 = np.gradient(dy_ds, s, edge_order=1)
        num = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
        denom = (dx_ds ** 2 + dy_ds ** 2) ** 1.5
        denom = np.maximum(denom, eps)
        kappa = num / denom
        valid = kappa > eps
        cap_k2 = np.full_like(x_cap, np.inf)
        cap_k2[valid] = max(float(lat_accel_max), eps) / kappa[valid]
        x_cap = np.minimum(x_cap, cap_k2)

    max_a = max(float(max_a), eps)
    a_up = np.full_like(s, max_a)
    a_lo = np.full_like(s, -max_a)

    x_prof = _reachability_pass(s, x_cap, a_up, a_lo, x0=0.0, xN=0.0)
    sdot = np.sqrt(np.maximum(x_prof, 0.0))

    t = np.zeros_like(sdot)
    for i in range(1, len(sdot)):
        ds = max(float(s[i] - s[i - 1]), 0.0)
        v_avg = max(float(0.5 * (sdot[i] + sdot[i - 1])), 1e-6)
        t[i] = t[i - 1] + ds / v_avg

    sdot_smoothed = _smooth_jerk(sdot, t, float(max_jk))
    sdot_smoothed = _fill_internal_speed_holes(sdot_smoothed)
    for i in range(1, len(sdot_smoothed)):
        sdot_smoothed[i] = min(
            sdot_smoothed[i],
            math.sqrt(max(sdot_smoothed[i - 1] ** 2 + 2.0 * max_a * (s[i] - s[i - 1]), 0.0)),
        )
    for i in range(len(sdot_smoothed) - 2, -1, -1):
        sdot_smoothed[i] = min(
            sdot_smoothed[i],
            math.sqrt(max(sdot_smoothed[i + 1] ** 2 + 2.0 * max_a * (s[i + 1] - s[i]), 0.0)),
        )
    sdot_smoothed[0] = 0.0
    sdot_smoothed[-1] = 0.0

    t2 = np.zeros_like(sdot_smoothed)
    for i in range(1, len(sdot_smoothed)):
        ds = max(float(s[i] - s[i - 1]), 0.0)
        v_avg = max(float(0.5 * (sdot_smoothed[i] + sdot_smoothed[i - 1])), 1e-6)
        t2[i] = t2[i - 1] + ds / v_avg

    xdot = dx_ds * sdot_smoothed
    ydot = dy_ds * sdot_smoothed
    w = dth_ds * sdot_smoothed
    v_lin = np.hypot(xdot, ydot)

    a_est = np.gradient(v_lin, t2, edge_order=1) if len(v_lin) > 1 else np.zeros_like(v_lin)
    j_est = np.gradient(a_est, t2, edge_order=1) if len(a_est) > 2 else np.zeros_like(a_est)

    return {
        "t": t2,
        "xdot": xdot,
        "ydot": ydot,
        "w": w,
        "v_lin": v_lin,
        "meta": {
            "constraint_clipped": bool(np.any(v_lin + 1e-9 < np.sqrt(np.maximum(x_cap, 0.0)))),
            "peak_jerk": float(np.max(np.abs(j_est))) if j_est.size else 0.0,
            "jerk_clipped": bool(j_est.size and np.any(np.abs(j_est) > max(float(max_jk), 1e-9) + 1e-6)),
        },
    }
