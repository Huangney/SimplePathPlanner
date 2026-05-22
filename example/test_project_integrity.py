import math
import sys
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
from path_planner import Waypoint, build_path, dump_session, load_session


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


def test_core_user_velocity_override_still_generates_valid_path():
    points = [
        Waypoint(0.0, 0.0, 0.0, vx=1.0, vy=0.2, vw=0.1),
        Waypoint(4.0, 3.0, 0.6, vx=1.2, vy=0.1, vw=0.0),
        Waypoint(9.0, 5.0, 1.0, vx=0.9, vy=-0.1, vw=-0.05),
    ]
    samples = build_path(points, density=12.0)

    assert np.isfinite(samples.x).all(), "x contains non-finite values"
    assert np.isfinite(samples.y).all(), "y contains non-finite values"
    assert np.isfinite(samples.theta).all(), "theta contains non-finite values"
    assert samples.meta["sample_count"] >= 16, (
        "sample_count unexpectedly small; "
        f"actual={samples.meta['sample_count']}"
    )


def test_core_dump_and_load_roundtrip(tmp_path: Path):
    points = [
        Waypoint(1.0, 2.0, 0.3),
        Waypoint(3.0, 4.0, 0.7, vx=0.5, vy=0.0, vw=0.2),
    ]
    out = dump_session(tmp_path / "session_case", points, density=18.5, showpath=False)
    payload = load_session(out)

    loaded_points = payload["waypoints"]
    settings = payload["settings"]

    assert len(loaded_points) == 2, (
        f"waypoint count mismatch; expected=2 actual={len(loaded_points)}"
    )
    assert abs(settings["density"] - 18.5) < 1e-9, (
        "density mismatch after load; "
        f"expected=18.5 actual={settings['density']}"
    )
    assert settings["showpath"] is False, (
        f"showpath mismatch; expected=False actual={settings['showpath']}"
    )


# ========================
# coord_utils mapping tests
# ========================

def test_coord_roundtrip_grid_data_consistency_with_image_mapping():
    has_image = True
    img_w, img_h = 400, 800
    gx_in, gy_in = 7.25, 1.75

    dx, dy = grid_to_data(gx_in, gy_in, has_image, img_w, img_h)
    gx_out, gy_out = data_to_grid(dx, dy, has_image, img_w, img_h)

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

    dx0, dy0, dx1, dy1 = grid_data_bounds(has_image, img_w, img_h)
    vdx, vdy = grid_vec_to_data_vec(vx, vy, has_image, img_w, img_h)

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
    no_image = format_coord_status(1.2, 3.4, False, 0, 0)
    in_image = format_coord_status(10.2, 20.3, True, 100, 200)
    out_image = format_coord_status(-2.0, 50.0, True, 100, 200)

    assert "Data:" in no_image and "Pixel:" not in no_image, (
        f"no-image status format unexpected: {no_image}"
    )
    assert "Pixel:" in in_image and "Data:" in in_image, (
        f"in-bounds image status format unexpected: {in_image}"
    )
    assert "Pixel: (out)" in out_image, (
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

    app = canvas_module.GridCanvas()
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


def test_cmd_save_and_load_restores_points_and_settings(cmd_canvas, tmp_path: Path):
    cmd_canvas._handle_command(["addpoint", "1,1,0.0"])
    cmd_canvas._handle_command(["addpoint", "2,3,0.5,0.2,0.0,0.1"])
    cmd_canvas._handle_command(["density", "22"])
    cmd_canvas._handle_command(["showpath", "off"])

    save_path = tmp_path / "agent_case"
    cmd_canvas._handle_command(["save", str(save_path)])

    cmd_canvas.points = []
    cmd_canvas.path_density = 7.0
    cmd_canvas.show_path = True

    cmd_canvas._handle_command(["load", str(save_path)])

    assert len(cmd_canvas.points) == 2, (
        f"load should restore 2 points; actual={len(cmd_canvas.points)}"
    )
    assert abs(cmd_canvas.path_density - 22.0) < 1e-9, (
        "load should restore density=22; "
        f"actual={cmd_canvas.path_density}"
    )
    assert cmd_canvas.show_path is False, (
        f"load should restore showpath=False; actual={cmd_canvas.show_path}"
    )


def test_cmd_exit_sets_running_false(cmd_canvas):
    cmd_canvas._running = True
    cmd_canvas._handle_command(["q"])
    assert cmd_canvas._running is False, "q command must set _running to False"
