"""Tests for the eval-only adversary motion model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.kinematics import (  # noqa: E402
    EvalOnlyFirewallError,
    PlatformProfile,
    SweepCell,
    assert_eval_only,
    build_sweep_cells,
    cells_to_frame,
    digest_cells_frame,
    generate_two_phase_track,
    load_kinematics_config,
    range_nm,
    refuse_benign_corpus_write,
    thin_to_cadence,
)
from counterusv.defense.consistency import FirewallError, assert_benign_train_allowed  # noqa: E402
from counterusv.defense.geometry_features import geometry_features_from_points  # noqa: E402
from counterusv.kinematics.features import features_from_points, last_window_mask  # noqa: E402

ASSET = (25.77, -80.15)
ANNULUS = {"min_range_nm": 0.25, "max_range_nm": 6.0}
INBOUND = {"require_points_before_cpa": 3, "min_points": 4}


def _platform() -> PlatformProfile:
    return PlatformProfile(
        key="magura_v5",
        name="Magura V5",
        cruise_kn=22.0,
        burst_kn=42.0,
        range_nm=450.0,
        citations=("test",),
    )


def _cell(**kwargs) -> SweepCell:
    base = dict(
        cell_id="swp__magura_v5__fishing__v12_c2_b0__test",
        platform="magura_v5",
        mimicked_class="fishing",
        v_mimic_kn=12.0,
        commit_range_nm=2.0,
        bearing_offset_deg=0.0,
        use_burst_on_commit=True,
        unconstrained=False,
        negative_control=False,
        asset_id="test_berth",
        approach_bearing_deg=90.0,
    )
    base.update(kwargs)
    return SweepCell(**base)


def test_two_phase_commit_near_range_and_ends_toward_asset():
    cfg = {
        "synth_dt_s": 1.0,
        "thin_cadence_s": 60.0,
        "start_range_nm": 8.0,
        "terminal_range_nm": 0.3,
        "dynamics": {"max_turn_rate_dps": 8.0, "max_accel_kn_per_s": 0.5},
        "plausibility_band": {"max_in_band_kn": 40.0},
    }
    cell = _cell(commit_range_nm=2.0, bearing_offset_deg=0.0, v_mimic_kn=12.0)
    track = generate_two_phase_track(*ASSET, cell, _platform(), cfg=cfg)
    pts = track.points
    assert len(pts) >= 5
    # Terminal range inside / near annulus toward asset.
    r_end = range_nm(float(pts.iloc[-1]["lat"]), float(pts.iloc[-1]["lon"]), *ASSET)
    assert r_end <= 0.5
    # Reconstruct fine to check commit transition: regenerate and inspect meta.
    assert "commit" in track.meta["phases_present"]
    assert "mimic" in track.meta["phases_present"]
    # Peak uses burst on commit.
    assert track.meta["peak_speed_kn"] == pytest.approx(42.0)
    assert track.meta["role"] == "hostile"
    assert track.meta["source"] == "synth"


def test_negative_control_stays_at_mimic_speed():
    cfg = {
        "synth_dt_s": 1.0,
        "thin_cadence_s": 60.0,
        "start_range_nm": 6.0,
        "terminal_range_nm": 0.3,
        "dynamics": {"max_turn_rate_dps": 8.0, "max_accel_kn_per_s": 0.5},
        "plausibility_band": {"max_in_band_kn": 40.0},
    }
    cell = _cell(
        cell_id="nc__test",
        v_mimic_kn=8.0,
        commit_range_nm=0.5,
        use_burst_on_commit=False,
        negative_control=True,
    )
    track = generate_two_phase_track(*ASSET, cell, _platform(), cfg=cfg)
    assert float(track.points["sog"].max()) == pytest.approx(8.0, abs=0.05)
    assert track.meta["peak_speed_kn"] == pytest.approx(8.0)


def test_thin_to_cadence():
    t0 = 1_700_000_000.0
    fine = pd.DataFrame({
        "trip_id": ["t"] * 301,
        "t": t0 + np.arange(301),
        "lat": 25.0,
        "lon": -80.0,
        "sog": 10.0,
        "cog": 90.0,
    })
    thinned = thin_to_cadence(fine, 60.0)
    dts = np.diff(thinned["t"].to_numpy(dtype="float64"))
    assert np.median(dts) == pytest.approx(60.0, abs=1.0)
    assert len(thinned) < len(fine)


def test_firewall_train_refuses_synth():
    meta = {"role": "hostile", "source": "synth", "canonical_class": "usv"}
    assert_eval_only(meta)
    with pytest.raises(FirewallError):
        assert_benign_train_allowed(meta)
    with pytest.raises(EvalOnlyFirewallError):
        assert_eval_only({"role": "benign", "source": "ais"})
    with pytest.raises(EvalOnlyFirewallError):
        refuse_benign_corpus_write(REPO_ROOT / "data/behavior/features_window_300s.parquet")


def test_sweep_digest_stable():
    cfg = load_kinematics_config()
    assets = pd.DataFrame([{
        "asset_id": "a1",
        "lat": 25.77,
        "lon": -80.15,
        "role": "fit",
        "valid": True,
    }, {
        "asset_id": "a2",
        "lat": 29.9,
        "lon": -90.1,
        "role": "fit",
        "valid": True,
    }])
    # Deterministic bearings.
    bearings = {"a1": 10.0, "a2": 200.0}
    small = dict(cfg)
    small["sweep"] = {
        "v_mimic_kn": [8, 25],
        "commit_range_nm": [2.0],
        "bearing_offset_deg": [0.0],
        "mimicked_classes": ["fishing"],
        "platforms": ["magura_v5"],
        "include_unconstrained": False,
    }
    small["seed"] = 42
    c1 = cells_to_frame(build_sweep_cells(small, assets, approach_bearings=bearings))
    c2 = cells_to_frame(build_sweep_cells(small, assets, approach_bearings=bearings))
    assert digest_cells_frame(c1) == digest_cells_frame(c2)
    assert len(c1) == 2 * (1 + 2)  # nc + 2 sweep per asset


def test_feature_round_trip_berth_approach():
    cfg = {
        "synth_dt_s": 1.0,
        "thin_cadence_s": 60.0,
        "start_range_nm": 8.0,
        "terminal_range_nm": 0.3,
        "dynamics": {"max_turn_rate_dps": 8.0, "max_accel_kn_per_s": 0.5},
        "plausibility_band": {"max_in_band_kn": 40.0},
    }
    cell = _cell(
        v_mimic_kn=12.0,
        commit_range_nm=2.0,
        bearing_offset_deg=0.0,
        approach_bearing_deg=0.0,
    )
    track = generate_two_phase_track(*ASSET, cell, _platform(), cfg=cfg)
    pts = track.points.copy()
    # Kinematics features on last 600 s.
    mask = last_window_mask(pts, 600.0)
    win = pts.loc[mask]
    kin = features_from_points(win)
    assert not kin.empty
    assert float(kin.iloc[0]["sog_med"]) > 0
    geo = geometry_features_from_points(
        win, *ASSET, annulus=ANNULUS, inbound_leg=INBOUND,
    )
    assert geo is not None
    assert geo["geometry_usable"] is True
    assert geo["n_points_in_annulus"] >= 4
