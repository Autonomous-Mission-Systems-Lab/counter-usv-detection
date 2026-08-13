"""Tests for placement materialization (synthetic point cloud)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.defense.engagement import (  # noqa: E402
    PortRegion,
    load_engagement_geometry,
)
from counterusv.defense.placements import (  # noqa: E402
    derive_seed_placements,
    materialize_placements,
    offset_nm,
)


def _synthetic_port_cloud(
    seed_lat: float = 26.0,
    seed_lon: float = -80.2,
) -> pd.DataFrame:
    """Fairway east of seed, berth near seed, anchorage further offshore."""
    rows = []
    t0 = 1_700_000_000.0
    # Moving fairway traffic at ~east of seed.
    fw_lat, fw_lon = offset_nm(seed_lat, seed_lon, 90.0, 5.0)
    for i in range(200):
        rows.append({
            "trip_id": 1000 + (i % 40),
            "t": t0 + i * 60,
            "lat": fw_lat + np.random.default_rng(i).normal(0, 0.01),
            "lon": fw_lon + np.random.default_rng(i + 1).normal(0, 0.01),
            "sog": 8.0,
        })
    # Long-dwell berth near seed (harbour).
    berth_lat, berth_lon = offset_nm(seed_lat, seed_lon, 90.0, 1.0)
    for tid in range(10):
        for k in range(400):  # ~6.6 h at 60 s
            rows.append({
                "trip_id": 2000 + tid,
                "t": t0 + k * 60,
                "lat": berth_lat + 0.001 * (tid % 3),
                "lon": berth_lon + 0.001 * (tid % 3),
                "sog": 0.1,
            })
    # Long-dwell anchorage seaward of fairway.
    anc_lat, anc_lon = offset_nm(fw_lat, fw_lon, 90.0, 2.0)
    for tid in range(8):
        for k in range(400):
            rows.append({
                "trip_id": 3000 + tid,
                "t": t0 + k * 60,
                "lat": anc_lat + 0.002 * (tid % 2),
                "lon": anc_lon + 0.002 * (tid % 2),
                "sog": 0.1,
            })
    # Inbound approaches toward berth for validity.
    for tid in range(60):
        for k, r_nm in enumerate([5.0, 4.0, 3.0, 2.0, 1.0, 0.8]):
            la, lo = offset_nm(berth_lat, berth_lon, 0.0, r_nm)
            rows.append({
                "trip_id": 4000 + tid,
                "t": t0 + tid * 1000 + k * 60,
                "lat": la,
                "lon": lo,
                "sog": 12.0,
            })
    return pd.DataFrame(rows)


def test_derive_seed_separates_berth_and_fairway():
    cfg = load_engagement_geometry()
    seed = PortRegion("miami_approach", 26.0, -80.2, "test")
    pts = _synthetic_port_cloud()
    cands = derive_seed_placements(pts, seed, cfg, search_nm=40)
    by_name = {c.placement_class: c for c in cands}
    assert set(by_name) == {
        "berth_approach", "anchorage", "offshore_terminal", "fairway_stress",
    }
    # Fairway should be east of berth (larger lon in this geometry).
    assert by_name["fairway_stress"].lon > by_name["berth_approach"].lon - 0.05
    # Anchorage separated from berth.
    from counterusv.kinematics.features import haversine_km
    d = haversine_km(
        by_name["berth_approach"].lat, by_name["berth_approach"].lon,
        by_name["anchorage"].lat, by_name["anchorage"].lon,
    ) / 1.852
    assert d >= 0.5


def test_materialize_table_schema():
    cfg = load_engagement_geometry()
    seed = PortRegion("miami_approach", 26.0, -80.2, "test")
    df = materialize_placements(
        _synthetic_port_cloud(), cfg, seeds=[seed], search_nm=40,
    )
    assert len(df) == 4
    for col in (
        "asset_id", "port_region", "placement_class", "role", "lat", "lon",
        "n_inbound_legs_in_annulus", "min_observed_range_nm", "water_occupied",
        "valid", "reject_reason",
    ):
        assert col in df.columns
    assert set(df.loc[df.role == "fit", "placement_class"]) == {
        "berth_approach", "anchorage",
    }


def test_pairing_radius_in_config():
    cfg = load_engagement_geometry()
    assert cfg.placement_policy.get("pairing_radius_nm") == 100
