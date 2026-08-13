"""Unit tests for asset-relative geometry feature extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.defense.geometry_features import (  # noqa: E402
    GEOMETRY_FEATURE_KEYS,
    geometry_features_from_points,
    range_bearing_nm,
)

ANNULUS = {"min_range_nm": 0.25, "max_range_nm": 6.0}
INBOUND = {"require_points_before_cpa": 3, "min_points": 4}
ASSET = (25.77, -80.15)


def _track(
    ranges_nm: list[float],
    *,
    bearing_deg: float = 90.0,
    dt_s: float = 60.0,
    asset_lat: float = ASSET[0],
    asset_lon: float = ASSET[1],
) -> pd.DataFrame:
    """Build a synthetic track along a constant bearing from the asset."""
    # Point at (range, bearing): east/north offset in nm.
    br = np.radians(bearing_deg)
    east = np.asarray(ranges_nm) * np.sin(br)
    north = np.asarray(ranges_nm) * np.cos(br)
    lat = asset_lat + north / 60.0
    lon = asset_lon + east / (60.0 * np.cos(np.radians(asset_lat)))
    t0 = 1_700_000_000.0
    t = t0 + np.arange(len(ranges_nm)) * dt_s
    return pd.DataFrame({"t": t, "lat": lat, "lon": lon, "sog": 10.0, "cog": bearing_deg + 180.0})


def test_range_bearing_basic():
    # 1 nm due north of asset.
    r, b = range_bearing_nm(ASSET[0] + 1.0 / 60.0, ASSET[1], *ASSET)
    assert float(r) == pytest.approx(1.0, abs=0.02)
    assert float(b) == pytest.approx(0.0, abs=1.0)


def test_inbound_closing_yields_features():
    # Approach from 5 nm → 1 nm (closing).
    pts = _track([5.0, 4.0, 3.0, 2.0, 1.0])
    feat = geometry_features_from_points(pts, *ASSET, annulus=ANNULUS, inbound_leg=INBOUND)
    assert feat is not None
    assert feat["geometry_usable"] is True
    assert set(GEOMETRY_FEATURE_KEYS) <= set(feat)
    assert feat["n_points_in_annulus"] == 5
    assert feat["cpa_range_nm"] == pytest.approx(1.0, abs=0.05)
    assert feat["range_min_nm"] == pytest.approx(1.0, abs=0.05)
    assert feat["closing_rate_med_kn"] > 0  # approaching
    assert feat["closing_frac"] == pytest.approx(1.0)
    assert feat["inbound_leg_persistence_s"] == pytest.approx(240.0)


def test_out_of_annulus_abstains():
    pts = _track([10.0, 9.5, 9.0, 8.5, 8.0])  # all beyond 6 nm
    assert geometry_features_from_points(pts, *ASSET, annulus=ANNULUS, inbound_leg=INBOUND) is None


def test_too_few_points_abstains():
    pts = _track([4.0, 3.0, 2.0])  # only 3 points
    assert geometry_features_from_points(pts, *ASSET, annulus=ANNULUS, inbound_leg=INBOUND) is None


def test_insufficient_history_before_cpa_abstains():
    # CPA at first in-annulus sample after jumping in from outside.
    pts = _track([8.0, 7.0, 1.5, 1.4, 1.3])
    # Points at 1.5,1.4,1.3 are in annulus; CPA is last → only 2 before CPA.
    feat = geometry_features_from_points(pts, *ASSET, annulus=ANNULUS, inbound_leg=INBOUND)
    assert feat is None


def test_transit_past_inbound_leg_ends_at_cpa():
    # Fly by: close then open. CPA in the middle; inbound leg uses only the
    # closing half (enough points before CPA to clear the history gate).
    pts = _track([5.5, 4.5, 3.5, 2.5, 1.5, 2.5, 3.5, 4.5])
    feat = geometry_features_from_points(pts, *ASSET, annulus=ANNULUS, inbound_leg=INBOUND)
    assert feat is not None
    assert feat["cpa_range_nm"] == pytest.approx(1.5, abs=0.1)
    assert feat["closing_frac"] == pytest.approx(1.0)
    assert feat["closing_rate_med_kn"] > 0


def test_never_imputes_zeros_on_abstain():
    pts = _track([12.0, 11.0, 10.0, 9.0])
    assert geometry_features_from_points(pts, *ASSET) is None
