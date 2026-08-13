"""Unit tests for adaptive cost-curve helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.eval.adaptive_cost import (  # noqa: E402
    added_approach_time_h,
    added_approach_time_min,
    annulus_entry_time,
    checkpoint_times,
    ddr_by_axis,
    first_flag_along_track,
    join_ddr_sweep,
    load_adaptive_cost_config,
    load_freeze_platforms,
    platform_burst_kn,
    range_series,
)
from counterusv.eval.oracle_ddr import assert_freeze_has_no_ddr  # noqa: E402


def test_load_adaptive_cost_config():
    cfg = load_adaptive_cost_config()
    assert cfg["condition"] == "perfect_disguise_oracle"
    assert "kinematics_only" in cfg["arms"]
    assert float(cfg["cost"]["start_range_nm"]) == 12.0
    assert float(cfg["warning_time"]["stride_s"]) == 60.0


def test_added_approach_time_formula():
    # R = 12 - 2 = 10 nm; v=10, vmax=20 → 10*(0.1-0.05)=0.5 h = 30 min
    assert added_approach_time_h(
        v_mimic_kn=10.0, v_max_kn=20.0, commit_range_nm=2.0, start_range_nm=12.0
    ) == pytest.approx(0.5)
    assert added_approach_time_min(
        v_mimic_kn=10.0, v_max_kn=20.0, commit_range_nm=2.0, start_range_nm=12.0
    ) == pytest.approx(30.0)
    assert (
        added_approach_time_h(
            v_mimic_kn=10.0,
            v_max_kn=20.0,
            commit_range_nm=2.0,
            unconstrained=True,
        )
        == 0.0
    )
    assert (
        added_approach_time_h(
            v_mimic_kn=42.0, v_max_kn=42.0, commit_range_nm=0.5
        )
        == 0.0
    )


def test_platform_burst_kn():
    plats = {"magura_v5": {"burst_kn": 42.0, "cruise_kn": 22.0}}
    assert platform_burst_kn(plats, "magura_v5") == 42.0
    assert platform_burst_kn(plats, "missing", default=45.0) == 45.0


def test_join_ddr_sweep_excludes_nc():
    ddr = pd.DataFrame([
        {
            "cell_id": "a",
            "trip_id": "a",
            "arm": "kinematics_only",
            "defense_kind": "consistency",
            "action": "flag",
            "mimicked_class": "fishing",
            "platform": "magura_v5",
            "asset_id": "x",
            "negative_control": False,
            "unconstrained": False,
        },
        {
            "cell_id": "a",
            "trip_id": "a",
            "arm": "kinematics_geometry",
            "defense_kind": "consistency",
            "action": "pass",
            "mimicked_class": "fishing",
            "platform": "magura_v5",
            "asset_id": "x",
            "negative_control": False,
            "unconstrained": False,
        },
        {
            "cell_id": "nc",
            "trip_id": "nc",
            "arm": "kinematics_only",
            "defense_kind": "consistency",
            "action": "pass",
            "mimicked_class": "fishing",
            "platform": "magura_v5",
            "asset_id": "x",
            "negative_control": True,
            "unconstrained": False,
        },
        {
            "cell_id": "a",
            "trip_id": "a",
            "arm": "presence",
            "defense_kind": "presence_only",
            "action": "pass",
            "mimicked_class": "fishing",
            "platform": "magura_v5",
            "asset_id": "x",
            "negative_control": False,
            "unconstrained": False,
        },
    ])
    sweep = pd.DataFrame([
        {
            "cell_id": "a",
            "v_mimic_kn": 12.0,
            "commit_range_nm": 2.0,
            "bearing_offset_deg": 0.0,
            "peak_speed_kn": 12.0,
            "use_burst_on_commit": True,
            "in_plausibility_band": True,
        },
        {
            "cell_id": "nc",
            "v_mimic_kn": 8.0,
            "commit_range_nm": 0.5,
            "bearing_offset_deg": 0.0,
            "peak_speed_kn": 8.0,
            "use_burst_on_commit": False,
            "in_plausibility_band": True,
        },
    ])
    plats = {"magura_v5": {"burst_kn": 42.0}}
    joined = join_ddr_sweep(ddr, sweep, plats, start_range_nm=12.0)
    assert len(joined) == 2
    assert set(joined["arm"]) == {"kinematics_only", "kinematics_geometry"}
    assert "nc" not in set(joined["cell_id"])
    # R=10, v=12, vmax=42 → 10*(1/12 - 1/42) h
    expect_h = 10.0 * (1 / 12.0 - 1 / 42.0)
    assert joined.iloc[0]["delta_t_add_h"] == pytest.approx(expect_h)


def test_ddr_by_axis():
    df = pd.DataFrame([
        {
            "mimicked_class": "fishing",
            "arm": "kinematics_only",
            "v_mimic_kn": 8.0,
            "action": "flag",
        },
        {
            "mimicked_class": "fishing",
            "arm": "kinematics_only",
            "v_mimic_kn": 8.0,
            "action": "pass",
        },
        {
            "mimicked_class": "fishing",
            "arm": "kinematics_only",
            "v_mimic_kn": 25.0,
            "action": "flag",
        },
    ])
    tab = ddr_by_axis(df, "v_mimic_kn", thin_n_threshold=20)
    assert len(tab) == 2
    row8 = tab.loc[tab["v_mimic_kn"] == 8.0].iloc[0]
    assert row8["ddr"] == pytest.approx(0.5)


def test_checkpoint_and_first_flag():
    # Straight inbound-ish toy track: t=0..900 every 60s.
    t = np.arange(0.0, 901.0, 60.0)
    # Fake lat/lon near asset so ranges decrease.
    asset_lat, asset_lon = 25.0, -80.0
    # Start ~far, approach by moving lat.
    lat = 25.0 + 0.15 * (1.0 - t / 900.0)
    lon = np.full_like(t, -80.0)
    points = pd.DataFrame({"t": t, "lat": lat, "lon": lon, "sog": 10.0, "cog": 180.0})
    r = range_series(points, asset_lat, asset_lon)
    assert r.iloc[0] > r.iloc[-1]
    t_enter = annulus_entry_time(points, r, max_range_nm=6.0)
    assert t_enter is not None

    times = checkpoint_times(points, min_history_s=300.0, stride_s=60.0)
    assert times
    assert times[0] >= 300.0 - 1e-6

    calls = {"n": 0}

    def evaluate_fn(*, features_by_window, complete_windows):
        calls["n"] += 1
        # Flag on second successful checkpoint.
        action = "flag" if calls["n"] >= 2 else "pass"
        return SimpleNamespace(action=action)

    # Monkeypatch extract_windows via wrapping — first_flag calls extract_windows
    # which needs real features; for unit test, patch by providing enough
    # points that extract_windows may abstain. Instead stub evaluate only when
    # windows exist: force by_w non-empty by patching module function.
    import counterusv.eval.adaptive_cost as ac

    def fake_extract(points, windows_s, **kwargs):
        return {int(windows_s[-1]): {"sog_p50": 10.0}}, {int(windows_s[-1])}

    orig = ac.extract_windows
    ac.extract_windows = fake_extract  # type: ignore[assignment]
    try:
        out = first_flag_along_track(
            points,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            max_range_nm=6.0,
            windows_s=[120, 180, 300],
            join_geometry=False,
            annulus={"min_range_nm": 0.25, "max_range_nm": 6.0},
            inbound_leg=None,
            evaluate_fn=evaluate_fn,
            stride_s=60.0,
        )
    finally:
        ac.extract_windows = orig  # type: ignore[assignment]

    assert out["flagged"] is True
    assert out["R_flag_nm"] is not None
    assert out["t_flag_unix"] is not None
    assert calls["n"] == 2


def test_freeze_anti_circularity_if_present():
    freeze = REPO_ROOT / "results" / "adversary_motion" / "FROZEN_SWEEP.json"
    if not freeze.is_file():
        pytest.skip("motion freeze not present")
    f, plats = load_freeze_platforms(freeze)
    assert "magura_v5" in plats or plats
    assert_freeze_has_no_ddr(freeze)
    # Injecting DDR keys must fail.
    bad = dict(f)
    bad["adaptive_cost_curve"] = {"ddr": 0.5}
    tmp = REPO_ROOT / "results" / "adaptive_cost" / "_test_bad_freeze.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="anti-circularity"):
        assert_freeze_has_no_ddr(tmp)
    tmp.unlink(missing_ok=True)
