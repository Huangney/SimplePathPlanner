import math
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coord_utils import (
    data_to_grid,
    format_coord_status,
    grid_data_bounds,
    grid_to_data,
    grid_vec_to_data_vec,
)
from app_config import DEFAULT_PROFILE, get_profile_config
import main as main_module
from main import parse_args
from path_planner import Waypoint, SpeedLimits, build_path, dump_session, load_session, export_path_cpp


def _assert_monotonic_non_decreasing(arr: np.ndarray, label: str):
    diffs = np.diff(arr)
    ok = np.all(diffs >= -1e-9)
    assert ok, (
        f"{label} must be monotonic non-decreasing; "
        f"min_diff={float(diffs.min()) if diffs.size else 'n/a'}"
    )


# =========================
# core: path_planner tests
# =========================

def test_core_build_path_generates_valid_samples():
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(4.0, 2.0, 0.6),
        Waypoint(8.0, 4.0, 1.0),
    ]
    samples = build_path(points, density=20.0)

    assert samples.meta["segments"] == 2, (
        f"segments mismatch; expected=2 actual={samples.meta['segments']}"
    )
    assert samples.x.size > 0, "sample_count must be >0 for 3 waypoints"
    assert samples.meta["sample_count"] == int(samples.x.size), (
        "meta.sample_count must equal samples.x size; "
        f"meta={samples.meta['sample_count']} actual={samples.x.size}"
    )
    _assert_monotonic_non_decreasing(samples.s, "arc-length s")


def test_core_density_increase_produces_more_samples():
    points = [Waypoint(0.0, 0.0, 0.0), Waypoint(10.0, 3.0, 0.8)]
    low = build_path(points, density=2.0)
    high = build_path(points, density=25.0)

    assert high.meta["sample_count"] > low.meta["sample_count"], (
        "higher density must produce more samples; "
        f"low={low.meta['sample_count']} high={high.meta['sample_count']}"
    )


def test_core_max_dt_resampling_caps_time_gap():
    points = [Waypoint(0.0, 0.0, 0.0), Waypoint(1.0, 0.0, 0.0)]
    limits = SpeedLimits(max_v=0.2, max_a=2.0, max_w=2.0, max_aw=2.0)
    base = build_path(points, density=1.0, speed_limits=limits)
    capped = build_path(points, density=1.0, speed_limits=limits, max_dt=0.05)

    assert capped.meta["sample_count"] > base.meta["sample_count"]
    assert capped.meta["pre_maxdt_sample_count"] == base.meta["sample_count"]
    assert capped.meta["max_dt_enabled"] is True
    assert float(np.max(np.diff(capped.t))) <= 0.05 + 1e-9
    assert abs(float(capped.t[-1]) - float(base.t[-1])) < 1e-9
    assert capped.meta["waypoint_sample_indices"][-1] == capped.meta["sample_count"] - 1


def test_core_theta_output_is_wrapped_to_pi_range():
    points = [
        Waypoint(0.0, 0.0, math.pi - 0.05),
        Waypoint(3.0, 2.0, -math.pi + 0.05),
        Waypoint(6.0, 2.5, -math.pi + 0.15),
    ]
    samples = build_path(points, density=15.0)

    lo = float(np.min(samples.theta))
    hi = float(np.max(samples.theta))
    assert lo >= -math.pi - 1e-9 and hi <= math.pi + 1e-9, (
        "theta must stay in [-pi, pi] after wrapping; "
        f"min={lo:.6f} max={hi:.6f}"
    )


def test_core_intermediate_theta_can_be_unconstrained():
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(1.0, 0.5, None, vw=1.0),
        Waypoint(2.0, 0.0, math.pi),
    ]
    samples = build_path(
        points,
        density=20.0,
        speed_limits=SpeedLimits(max_v=1.0, max_a=1.0, max_w=2.0, max_aw=2.0),
    )

    assert samples.meta.get("heading_anchor_waypoint_indices") == [0, 2]
    mid_idx = int(samples.meta["waypoint_sample_indices"][1])
    assert abs(float(samples.w[mid_idx]) - 1.0) > 1e-3, "vw on theta-less waypoint should not be a hard anchor"
    assert samples.theta.size == samples.x.size


def test_core_anchor_velocity_and_constraints_respected():
    limits = SpeedLimits(max_v=1.2, max_a=0.8, max_w=1.1, max_aw=1.5)
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(4.0, 3.0, 0.6, vx=0.8, vy=0.0, vw=0.2),
        Waypoint(9.0, 5.0, 1.0),
    ]
    samples = build_path(points, density=15.0, speed_limits=limits)

    assert samples.t.size == samples.x.size, "t should align with path sample size"
    assert samples.meta["total_time"] > 0.0, "total_time should be positive"
    assert np.max(samples.v_lin) <= limits.max_v + 1e-6, "linear speed must not exceed max_v"
    assert np.max(np.abs(samples.w)) <= limits.max_w + 1e-6, "angular speed must not exceed max_w"
    assert samples.meta.get("solver") == "coupled"

    assert samples.v_lin[0] <= 1e-6 and samples.v_lin[-1] <= 1e-6, "endpoints should default to zero speed"
    mid_idx = int(samples.meta["waypoint_sample_indices"][1])
    assert abs(float(samples.w[mid_idx]) - 0.2) < 1e-6, "vw should be enforced at waypoint samples"


def test_core_angular_limits_affect_time_parameterization():
    points = [
        Waypoint(1.3, 4.5, 1.57),
        Waypoint(1.77, 3.34, 0.0, vw=0.0),
        Waypoint(2.3, 3.0, 0.0, vx=1.0, vy=0.0),
        Waypoint(2.7, 3.0, 0.0),
    ]

    loose = build_path(
        points,
        density=40.0,
        speed_limits=SpeedLimits(max_v=2.0, max_a=2.5, max_w=3.14, max_aw=3.14, lat_accel_max=2.0),
    )
    tight_w = build_path(
        points,
        density=40.0,
        speed_limits=SpeedLimits(max_v=2.0, max_a=2.5, max_w=0.5, max_aw=3.14, lat_accel_max=2.0),
    )
    tight_aw = build_path(
        points,
        density=40.0,
        speed_limits=SpeedLimits(max_v=2.0, max_a=2.5, max_w=3.14, max_aw=0.5, lat_accel_max=2.0),
    )

    loose_aw = np.gradient(loose.w, loose.t, edge_order=1)
    tight_aw_est = np.gradient(tight_aw.w, tight_aw.t, edge_order=1)

    assert float(np.max(np.abs(tight_w.w))) <= 0.5 + 1e-6
    assert tight_w.meta["total_time"] > loose.meta["total_time"]
    assert float(np.max(np.abs(tight_aw_est))) <= 0.5 + 1e-6
    assert float(np.max(np.abs(tight_aw_est))) < float(np.max(np.abs(loose_aw)))


def test_core_coupled_solver_smooths_stt_regression():
    payload = json.loads((ROOT / "stt.json").read_text(encoding="utf-8"))
    points = [Waypoint(**item) for item in payload["waypoints"]]
    limits = SpeedLimits(**payload["settings"]["speed_limits"])

    samples = build_path(points, density=payload["settings"]["density"], speed_limits=limits, solver="legacy")
    aw = np.gradient(samples.w, samples.t, edge_order=1)
    dt = np.diff(samples.t)
    dv = np.diff(samples.v_lin)
    accel = np.divide(dv, dt, out=np.zeros_like(dv), where=dt > 1e-9)

    assert samples.meta.get("solver") == "coupled"
    assert float(np.max(np.abs(accel))) <= limits.max_a + 5e-3
    assert float(np.max(np.abs(samples.w))) <= limits.max_w + 1e-6
    assert float(np.max(np.abs(aw))) <= limits.max_aw + 1e-6

    for idx in samples.meta["waypoint_sample_indices"][1:-1]:
        lo = max(0, int(idx) - 3)
        hi = min(samples.v_lin.size, int(idx) + 4)
        assert float(np.max(np.abs(np.diff(samples.v_lin[lo:hi])))) < 0.2


def test_core_waypoint_velocity_direction_matches_local_tangent():
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(4.0, 2.0, 0.2, vx=-0.5, vy=0.0, vw=0.0),
        Waypoint(8.0, 2.0, 0.4),
    ]
    samples = build_path(points, density=20.0, speed_limits=SpeedLimits(max_v=2.0, max_a=2.0, max_w=2.0, max_aw=2.0))

    idx = int(samples.meta["waypoint_sample_indices"][1])
    i0 = max(0, idx - 1)
    i1 = min(samples.x.size - 1, idx + 1)
    tx = float(samples.x[i1] - samples.x[i0])
    ty = float(samples.y[i1] - samples.y[i0])
    tangent = np.array([tx, ty], dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    assert tangent_norm > 1e-9, "local tangent norm too small"
    tangent /= tangent_norm

    vel = np.array([-0.5, 0.0], dtype=float)
    vel /= float(np.linalg.norm(vel))
    cosang = float(np.dot(tangent, vel))
    assert cosang > 0.95, f"waypoint tangent should align with velocity direction; cos={cosang:.6f}"


def test_core_partial_waypoint_velocity_treats_missing_component_as_zero():
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(4.0, 2.0, 0.2, vx=None, vy=1.0, vw=0.0),
        Waypoint(8.0, 2.0, 0.4),
    ]
    samples = build_path(points, density=20.0, speed_limits=SpeedLimits(max_v=2.0, max_a=2.0, max_w=2.0, max_aw=2.0))

    idx = int(samples.meta["waypoint_sample_indices"][1])
    i0 = max(0, idx - 1)
    i1 = min(samples.x.size - 1, idx + 1)
    tx = float(samples.x[i1] - samples.x[i0])
    ty = float(samples.y[i1] - samples.y[i0])
    tangent = np.array([tx, ty], dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    assert tangent_norm > 1e-9, "local tangent norm too small"
    tangent /= tangent_norm

    vel = np.array([0.0, 1.0], dtype=float)
    cosang = float(np.dot(tangent, vel))
    assert cosang > 0.95, f"partial velocity anchor should use missing component as zero; cos={cosang:.6f}"


def test_core_dump_and_load_roundtrip(tmp_path: Path):
    points = [
        Waypoint(1.0, 2.0, 0.3),
        Waypoint(2.0, 3.0, None, vx=0.5, vy=0.0, vw=0.2),
        Waypoint(3.0, 4.0, 0.7),
    ]
    limits = SpeedLimits(max_v=1.3, max_a=0.9, max_w=1.4, max_aw=1.6, lat_accel_max=0.75)
    out = dump_session(
        tmp_path / "session_case",
        points,
        density=18.5,
        showpath=False,
        speed_limits=limits,
        solver="legacy",
        max_dt=0.04,
    )
    payload = load_session(out)

    loaded_points = payload["waypoints"]
    settings = payload["settings"]

    assert len(loaded_points) == 3, (
        f"waypoint count mismatch; expected=3 actual={len(loaded_points)}"
    )
    assert abs(settings["density"] - 18.5) < 1e-9, (
        "density mismatch after load; "
        f"expected=18.5 actual={settings['density']}"
    )
    assert settings["showpath"] is False, (
        f"showpath mismatch; expected=False actual={settings['showpath']}"
    )
    assert settings.get("solver") == "coupled", f"solver mismatch after load: {settings.get('solver')}"
    assert abs(settings["max_dt"] - 0.04) < 1e-9
    assert isinstance(settings["speed_limits"], SpeedLimits), "speed_limits should deserialize to SpeedLimits"
    assert abs(settings["speed_limits"].max_v - 1.3) < 1e-9
    assert abs(settings["speed_limits"].lat_accel_max - 0.75) < 1e-9
    assert loaded_points[1].theta is None


def test_core_export_cpp_generates_header_and_applies_scale(tmp_path: Path):
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(2.0, 1.0, 0.4),
        Waypoint(3.0, 2.0, 0.8),
    ]
    samples = build_path(points, density=10.0, speed_limits=SpeedLimits(max_v=1.5, max_a=1.0, max_w=1.0, max_aw=1.0, lat_accel_max=0.0))
    out = export_path_cpp(tmp_path / "path_data", samples, path_name="R2_Path", grid_scale=0.5)

    text = out.read_text(encoding="utf-8")
    assert '#include "PathChaser.hpp"' in text
    assert f"static const Path<{samples.x.size}> R2_Path" in text
    assert "grid_scale(m/grid): 0.500000" in text
    assert "{0.000000f, 0.000000f}" in text


def test_core_export_cpp_rejects_when_capacity_too_small(tmp_path: Path):
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(12.0, 6.0, 0.0),
    ]
    samples = build_path(points, density=60.0)
    with pytest.raises(ValueError) as exc:
        export_path_cpp(tmp_path / "too_long.hpp", samples, capacity=32)
    assert "exceeds capacity" in str(exc.value)


# ========================
# coord_utils mapping tests
# ========================

def test_coord_roundtrip_grid_data_consistency_with_image_mapping():
    has_image = True
    img_w, img_h = 400, 800
    gx_in, gy_in = 7.25, 1.75
    cfg = get_profile_config(DEFAULT_PROFILE)

    dx, dy = grid_to_data(gx_in, gy_in, has_image, img_w, img_h, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)
    gx_out, gy_out = data_to_grid(dx, dy, has_image, img_w, img_h, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)

    assert gx_out is not None and gy_out is not None, "grid roundtrip returned None unexpectedly"
    assert abs(gx_out - gx_in) < 1e-6, (
        "gx roundtrip mismatch; "
        f"input={gx_in} output={gx_out}"
    )
    assert abs(gy_out - gy_in) < 1e-6, (
        "gy roundtrip mismatch; "
        f"input={gy_in} output={gy_out}"
    )


def test_coord_vector_transform_matches_bounds_scale():
    has_image = True
    img_w, img_h = 400, 800
    vx, vy = 2.0, 3.0
    cfg = get_profile_config(DEFAULT_PROFILE)

    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)
    vdx, vdy = grid_vec_to_data_vec(vx, vy, has_image, img_w, img_h, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)

    expected_vdx = vy / 6.0 * (dx1 - dx0)
    expected_vdy = vx / 12.0 * (dy1 - dy0)

    assert abs(vdx - expected_vdx) < 1e-9, (
        "vdx scale mismatch; "
        f"expected={expected_vdx} actual={vdx}"
    )
    assert abs(vdy - expected_vdy) < 1e-9, (
        "vdy scale mismatch; "
        f"expected={expected_vdy} actual={vdy}"
    )


def test_coord_format_status_covers_modes_and_out_of_bounds():
    cfg = get_profile_config(DEFAULT_PROFILE)
    no_image = format_coord_status(1.2, 3.4, False, 0, 0, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)
    in_image = format_coord_status(10.2, 20.3, True, 100, 200, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)
    out_image = format_coord_status(-2.0, 50.0, True, 100, 200, cfg.grid_x0, cfg.grid_y0, cfg.grid_x1, cfg.grid_y1)

    assert "Data:" in no_image and "Grid:" in no_image, (
        f"no-image status format unexpected: {no_image}"
    )
    assert "Grid:" in in_image and "Data:" in in_image, (
        f"in-bounds image status format unexpected: {in_image}"
    )
    assert "Grid: (out)" in out_image, (
        f"out-of-bounds image status format unexpected: {out_image}"
    )


# ========================
# cmd: canvas flow tests
# ========================

def _can_construct_canvas_without_gui_error() -> tuple[bool, str]:
    try:
        import canvas as canvas_module

        original_redraw = canvas_module.GridCanvas.redraw
        canvas_module.GridCanvas.redraw = lambda self: self._rebuild_path()
        try:
            app = canvas_module.GridCanvas()
            app._running = False
            app.fig.clf()
            return True, ""
        finally:
            canvas_module.GridCanvas.redraw = original_redraw
    except Exception as exc:
        return False, f"GUI backend unavailable: {type(exc).__name__}: {exc}"


_CANVAS_OK, _CANVAS_SKIP_REASON = _can_construct_canvas_without_gui_error()


@pytest.fixture
def cmd_canvas(monkeypatch):
    if not _CANVAS_OK:
        pytest.skip(_CANVAS_SKIP_REASON)

    import canvas as canvas_module

    monkeypatch.setattr(canvas_module.plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(canvas_module.GridCanvas, "redraw", lambda self: self._rebuild_path())

    try:
        app = canvas_module.GridCanvas()
    except Exception as exc:
        pytest.skip(f"GUI backend unavailable: {type(exc).__name__}: {exc}")
    try:
        yield app
    finally:
        app._running = False
        try:
            app.fig.clf()
        except Exception:
            pass


def test_cmd_addpoint_accepts_valid_and_rejects_out_of_range(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "1.0,", "2.0,", "0.5"])
    before = len(cmd_canvas.points)

    cmd_canvas._handle_command(["addpoint", "999,", "1.0,", "0.0"])
    after = len(cmd_canvas.points)

    assert before == 1, f"valid addpoint should add 1 point; actual={before}"
    assert after == before, (
        "out-of-range addpoint must not change point count; "
        f"before={before} after={after}"
    )


def test_gui_key_a_adds_point_without_theta_warning(cmd_canvas):
    before = len(cmd_canvas.points)
    event = types.SimpleNamespace(key="a", inaxes=cmd_canvas.ax, xdata=1.5, ydata=2.5)

    cmd_canvas._on_key_press(event)

    assert len(cmd_canvas.points) == before + 1, "pressing A should add a new waypoint"
    assert cmd_canvas.points[-1].theta is not None, "mouse-driven endpoint insert should infer theta"


def test_cmd_density_and_showpath_toggle_state(cmd_canvas):
    old_density = cmd_canvas.path_density

    cmd_canvas._handle_command(["density", "30"])
    assert abs(cmd_canvas.path_density - 30.0) < 1e-9, (
        f"density command failed; expected=30 actual={cmd_canvas.path_density}"
    )

    cmd_canvas._handle_command(["density", "0.2"])
    assert abs(cmd_canvas.path_density - 30.0) < 1e-9, (
        "density <1.0 should be ignored; "
        f"old_valid=30 actual={cmd_canvas.path_density} (initial={old_density})"
    )

    cmd_canvas._handle_command(["showpath", "off"])
    assert cmd_canvas.show_path is False, "showpath off command failed"
    cmd_canvas._handle_command(["showpath", "on"])
    assert cmd_canvas.show_path is True, "showpath on command failed"


def test_cmd_plan_builds_meta_after_points_added(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "0,0,0"])
    cmd_canvas._handle_command(["addpoint", "4,1,0.3"])
    cmd_canvas._handle_command(["plan"])

    meta = cmd_canvas.path_samples.meta
    assert meta.get("segments") == 1, (
        f"plan should produce one segment; actual={meta.get('segments')}"
    )
    assert int(meta.get("sample_count", 0)) > 0, (
        f"plan should produce samples; actual={meta.get('sample_count')}"
    )
    assert float(meta.get("total_time", 0.0)) > 0.0, "plan should compute total_time"


def test_cmd_editpoint_updates_target_and_rejects_missing_index(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "1,1,0.0"])
    cmd_canvas._handle_command(["addpoint", "2,2,0.5"])
    cmd_canvas._handle_command(["set", "2", "vx", "0.1"])
    cmd_canvas._handle_command(["set", "2", "vy", "0.2"])
    cmd_canvas._handle_command(["set", "2", "vw", "0.3"])

    cmd_canvas._handle_command(["editpoint", "2", "3,4,0.8"])
    p2 = cmd_canvas.points[1]
    assert abs(p2.x - 3.0) < 1e-9 and abs(p2.y - 4.0) < 1e-9 and abs(p2.theta - 0.8) < 1e-9, (
        "editpoint should update target waypoint's x/y/theta"
    )
    assert abs(p2.vx - 0.1) < 1e-9 and abs(p2.vy - 0.2) < 1e-9 and abs(p2.vw - 0.3) < 1e-9, (
        "editpoint should preserve existing vx/vy/vw constraints"
    )

    before = [(p.x, p.y, p.theta, p.vx, p.vy, p.speed, p.vw) for p in cmd_canvas.points]
    cmd_canvas._handle_command(["editpoint", "9", "0,0,0"])
    after = [(p.x, p.y, p.theta, p.vx, p.vy, p.speed, p.vw) for p in cmd_canvas.points]
    assert before == after, "out-of-range editpoint must not mutate existing waypoints"


def test_cmd_intermediate_theta_can_be_empty(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "0,0,0"])
    cmd_canvas._handle_command(["addpoint", "2,0,0"])
    cmd_canvas._handle_command(["insert", "1", "1,0"])

    assert len(cmd_canvas.points) == 3
    assert cmd_canvas.points[1].theta is None

    cmd_canvas._handle_command(["set", "1", "theta", "none"])
    assert cmd_canvas.points[0].theta is not None, "endpoint theta must not be cleared"

    cmd_canvas._handle_command(["set", "2", "theta", "0.5"])
    assert abs(cmd_canvas.points[1].theta - 0.5) < 1e-9
    cmd_canvas._handle_command(["editpoint", "2", "1.1,0.2"])
    assert cmd_canvas.points[1].theta is None


def test_cmd_set_velocity_after_addpoint_is_used_in_path_planning(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "0,0,0"])
    cmd_canvas._handle_command(["addpoint", "3,0,0"])
    cmd_canvas._handle_command(["addpoint", "6,0,0"])

    baseline_y = np.array(cmd_canvas.path_samples.y, copy=True)

    cmd_canvas._handle_command(["set", "2", "vx", "0.0"])
    cmd_canvas._handle_command(["set", "2", "vy", "1.2"])
    updated_y = np.array(cmd_canvas.path_samples.y, copy=True)

    assert np.max(np.abs(updated_y - baseline_y)) > 1e-6, (
        "setting vx/vy direction should change Hermite path geometry"
    )

    cmd_canvas._handle_command(["set", "2", "speed", "0.8"])
    idx = int(cmd_canvas.path_samples.meta["waypoint_sample_indices"][1])
    anchored_v = float(cmd_canvas.path_samples.v_lin[idx])
    assert anchored_v <= 0.8 + 1e-6, "speed constraint should cap local linear speed"
    assert anchored_v <= cmd_canvas.speed_limits.max_v + 1e-6, "anchored speed should still respect max_v limit"


def test_core_interval_speed_limit_caps_selected_segment():
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(4.0, 0.0, 0.0),
        Waypoint(8.0, 0.0, 0.0),
    ]
    limits = SpeedLimits(max_v=3.0, max_a=10.0, interval_speed_limits=((1, 0.8),))
    samples = build_path(points, density=25.0, speed_limits=limits)
    wp = [int(i) for i in samples.meta["waypoint_sample_indices"]]

    first_seg_peak = float(np.max(samples.v_lin[wp[0]:wp[1] + 1]))
    second_seg_peak = float(np.max(samples.v_lin[wp[1]:wp[2] + 1]))
    assert second_seg_peak <= 0.8 + 1e-6, (
        f"interval speed limit should cap P2->P3; actual={second_seg_peak:.4f}"
    )
    assert first_seg_peak > second_seg_peak + 0.2, (
        f"unlimited segment should remain faster; first={first_seg_peak:.4f} second={second_seg_peak:.4f}"
    )


def test_cmd_ispdlim_updates_and_clears_segment_limit(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "0,0,0"])
    cmd_canvas._handle_command(["addpoint", "4,0,0"])
    cmd_canvas._handle_command(["addpoint", "8,0,0"])

    cmd_canvas._handle_command(["ispdlim", "1", "0.7"])
    assert cmd_canvas.speed_limits.interval_speed_limits == ((0, 0.7),)
    wp = [int(i) for i in cmd_canvas.path_samples.meta["waypoint_sample_indices"]]
    seg_peak = float(np.max(cmd_canvas.path_samples.v_lin[wp[0]:wp[1] + 1]))
    assert seg_peak <= 0.7 + 1e-6

    cmd_canvas._handle_command(["ispdlim", "1", "off"])
    assert cmd_canvas.speed_limits.interval_speed_limits == ()


def test_cmd_maxdt_updates_and_clears_resampling(cmd_canvas):
    cmd_canvas._handle_command(["addpoint", "0,0,0"])
    cmd_canvas._handle_command(["addpoint", "1,0,0"])
    cmd_canvas._handle_command(["speedcfg", "vmax=0.2", "amax=2.0", "wmax=2.0", "awmax=2.0"])

    cmd_canvas._handle_command(["maxdt", "0.05"])
    assert abs(cmd_canvas.path_max_dt - 0.05) < 1e-9
    assert cmd_canvas.path_samples.meta.get("max_dt_enabled") is True
    assert float(np.max(np.diff(cmd_canvas.path_samples.t))) <= 0.05 + 1e-9

    cmd_canvas._handle_command(["maxdt", "off"])
    assert cmd_canvas.path_max_dt is None


def test_cmd_speedcfg_updates_limits_and_rejects_invalid(cmd_canvas):
    old = cmd_canvas.speed_limits
    cmd_canvas._handle_command(["speedcfg", "vmax=1.8", "amax=0.7", "wmax=1.3", "awmax=1.1", "latacc=0.6"])
    new = cmd_canvas.speed_limits

    assert abs(new.max_v - 1.8) < 1e-9
    assert abs(new.max_a - 0.7) < 1e-9
    assert abs(new.max_w - 1.3) < 1e-9
    assert abs(new.max_aw - 1.1) < 1e-9
    assert abs(new.lat_accel_max - 0.6) < 1e-9

    cmd_canvas._handle_command(["speedcfg", "vmax=-1"])
    assert cmd_canvas.speed_limits == new, "invalid speedcfg must not mutate limits"
    assert cmd_canvas.speed_limits != old, "speedcfg valid update should change original limits"


def test_core_curvature_constraint_reduces_corner_speed():
    points = [
        Waypoint(0.0, 0.0, 0.0),
        Waypoint(2.0, 0.0, 0.0),
        Waypoint(2.0, 2.0, 1.57),
        Waypoint(2.0, 4.0, 1.57),
    ]
    base = SpeedLimits(max_v=3.0, max_a=3.0, lat_accel_max=1.0)
    samples = build_path(points, density=20.0, speed_limits=base)
    idx = int(samples.meta["waypoint_sample_indices"][2])
    corner_v = float(np.min(samples.v_lin[max(0, idx - 3):min(samples.v_lin.size, idx + 4)]))
    assert corner_v < 2.5, f"curvature constraint should slow corner well below max_v=3.0; actual={corner_v:.4f}"
    assert samples.meta["total_time"] > 0.0


def test_core_curvature_zero_disables_constraint():
    points = [Waypoint(0, 0, 0), Waypoint(2, 2, 0.8), Waypoint(4, 3, 0.0)]
    off = build_path(points, density=15.0,
                     speed_limits=SpeedLimits(max_v=2.0, max_a=2.0, lat_accel_max=0.0))
    assert off.meta["total_time"] > 0
    assert float(np.max(off.v_lin)) <= 2.0 + 1e-6


def test_cmd_save_and_load_restores_points_settings_and_speedcfg(cmd_canvas, tmp_path: Path):
    cmd_canvas._handle_command(["addpoint", "1,1,0.0"])
    cmd_canvas._handle_command(["addpoint", "2,3,0.5"])
    cmd_canvas._handle_command(["set", "2", "vx", "0.2"])
    cmd_canvas._handle_command(["set", "2", "vy", "0.0"])
    cmd_canvas._handle_command(["set", "2", "speed", "0.8"])
    cmd_canvas._handle_command(["set", "2", "vw", "0.1"])
    cmd_canvas._handle_command(["density", "22"])
    cmd_canvas._handle_command(["showpath", "off"])
    cmd_canvas._handle_command(["solver", "legacy"])
    cmd_canvas._handle_command(["speedcfg", "vmax=1.7", "amax=0.6", "wmax=1.2", "awmax=0.9"])
    cmd_canvas._handle_command(["ispdlim", "1", "0.65"])
    cmd_canvas._handle_command(["maxdt", "0.05"])

    save_path = tmp_path / "agent_case"
    cmd_canvas._handle_command(["save", str(save_path)])

    cmd_canvas.points = []
    cmd_canvas.path_density = 7.0
    cmd_canvas.path_max_dt = None
    cmd_canvas.show_path = True
    cmd_canvas.solver = "legacy"
    cmd_canvas._handle_command(["speedcfg", "vmax=2.5", "amax=2.5", "wmax=2.5", "awmax=2.5"])

    cmd_canvas._handle_command(["load", str(save_path)])

    assert len(cmd_canvas.points) == 2, (
        f"load should restore 2 points; actual={len(cmd_canvas.points)}"
    )
    assert abs(cmd_canvas.path_density - 22.0) < 1e-9, (
        "load should restore density=22; "
        f"actual={cmd_canvas.path_density}"
    )
    assert abs(cmd_canvas.path_max_dt - 0.05) < 1e-9, "load should restore maxdt=0.05"
    assert cmd_canvas.show_path is False, (
        f"load should restore showpath=False; actual={cmd_canvas.show_path}"
    )
    assert cmd_canvas.solver == "coupled", f"load should restore solver=coupled; actual={cmd_canvas.solver}"
    assert abs(cmd_canvas.speed_limits.max_v - 1.7) < 1e-9, "load should restore speed limits"
    assert cmd_canvas.speed_limits.interval_speed_limits == ((0, 0.65),), "load should restore interval speed limits"
    p2 = cmd_canvas.points[1]
    assert abs(p2.vx - 0.2) < 1e-9, "load should restore vx direction constraint"
    assert abs(p2.vy - 0.0) < 1e-9, "load should restore vy direction constraint"
    assert abs(p2.speed - 0.8) < 1e-9, "load should restore speed constraint"
    assert abs(p2.vw - 0.1) < 1e-9, "load should restore vw constraint"


def test_cmd_solver_switch_and_reject_invalid(cmd_canvas):
    cmd_canvas._handle_command(["solver"])
    assert cmd_canvas.solver == "coupled"
    cmd_canvas._handle_command(["solver", "bad_solver"])
    assert cmd_canvas.solver == "coupled", "invalid solver should not change current solver"


def test_cmd_exportcpp_parses_options_and_invokes_export(cmd_canvas, monkeypatch, tmp_path: Path):
    called = {}

    def _fake_export(file_path, samples, path_name="GeneratedPath", grid_scale=1.0, capacity=None):
        called["file_path"] = str(file_path)
        called["path_name"] = path_name
        called["grid_scale"] = float(grid_scale)
        called["capacity"] = capacity
        return Path(file_path)

    import canvas as canvas_module
    monkeypatch.setattr(canvas_module, "export_path_cpp", _fake_export)

    cmd_canvas._handle_command(["addpoint", "0,0,0"])
    cmd_canvas._handle_command(["addpoint", "2,1,0.3"])
    out = tmp_path / "mcu_path.hpp"
    cmd_canvas._handle_command(["exportcpp", str(out), "name=My_Path", "scale=0.25"])

    assert called["file_path"].endswith("mcu_path.hpp")
    assert called["path_name"] == "My_Path"
    assert abs(called["grid_scale"] - 0.25) < 1e-9
    assert called["capacity"] is None


def test_cmd_exit_sets_running_false(cmd_canvas):
    cmd_canvas._running = True
    cmd_canvas._handle_command(["q"])
    assert cmd_canvas._running is False, "q command must set _running to False"


def test_cli_profile_args_support_short_and_long_flags():
    args_default = parse_args([])
    args_short = parse_args(["-p=1"])
    args_long = parse_args(["--profile=2"])

    assert args_default.profile == 0, f"default profile should be 0; actual={args_default.profile}"
    assert args_short.profile == 1, f"-p=1 should parse to profile 1; actual={args_short.profile}"
    assert args_long.profile == 2, f"--profile=2 should parse to profile 2; actual={args_long.profile}"


def test_profile_missing_id_raises_readable_error():
    with pytest.raises(ValueError) as exc:
        get_profile_config(999)
    msg = str(exc.value)
    assert "not found" in msg and "available profiles" in msg, (
        f"unexpected error message for missing profile: {msg}"
    )


def test_main_returns_nonzero_for_invalid_profile():
    code = main_module.main(["-p=999"])
    assert code != 0, f"invalid profile should return non-zero; actual={code}"
