"""Unit tests for geometry-arm placement-swept FAR helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "behavior"))

from counterusv.kinematics.behavior_model import (  # noqa: E402
    EnvelopeModel,
    FittedSubspace,
    MultiHorizonEnvelope,
)
from validate_geometry_far import (  # noqa: E402
    assign_multihorizon_encounters,
    contact_key,
    stratum_far,
    verify_placements_digest,
)

CORE = [
    "sog_med", "sog_p95", "sog_std",
    "loiter_frac", "straightness", "accel_mean_abs",
]
GEOM = [
    "range_min_nm", "closing_rate_med_kn", "closing_rate_p90_kn",
    "bearing_rate_std_dps", "dcpa_nm", "tcpa_s",
    "closing_frac", "inbound_leg_persistence_s",
]


def _toy_envelope(window_s: int) -> EnvelopeModel:
    rng = np.random.default_rng(0)
    cols = CORE + GEOM
    X = rng.normal(0, 1, size=(60, len(cols)))
    scaler = StandardScaler().fit(X)
    gmm = GaussianMixture(n_components=1, random_state=0).fit(scaler.transform(X))
    scores = -gmm.score_samples(scaler.transform(X))
    thr = float(np.percentile(scores, 95))
    fs = FittedSubspace(
        subspace="core",
        feature_names=cols,
        scaler=scaler,
        model_name="gmm",
        model=gmm,
        n_train=len(X),
        n_val=20,
        thresholds={"far_0.05": thr},
    )
    return EnvelopeModel(
        name="toy",
        members={"canonical_class": ["recreational"]},
        subspaces={"core": {"gmm": fs}},
        core_features=CORE,
        course_features=[],
        window_s=float(window_s),
    )


def test_contact_key_distinguishes_assets():
    df = pd.DataFrame({
        "trip_id": ["t1", "t1", "t2"],
        "asset_id": ["a", "b", "a"],
    })
    keys = contact_key(df).tolist()
    assert keys[0] != keys[1]
    assert keys[0] == "t1||a"


def test_assign_multihorizon_prefers_longest_complete_per_asset():
    """Same trip, two assets — each keeps its own longest complete window."""
    env300 = _toy_envelope(300)
    env600 = _toy_envelope(600)
    bundle = MultiHorizonEnvelope(
        name="toy",
        members={"canonical_class": ["recreational"]},
        primary_window_s=600,
        core_features=CORE,
        course_features=[],
        horizons={300: env300, 600: env600},
    )
    rows = []
    for w, complete_a, complete_b in (
        (300, True, True),
        (600, True, False),  # asset b incomplete at 600
    ):
        for asset, complete in (("asset_a", complete_a), ("asset_b", complete_b)):
            row = {
                "trip_id": "trip1",
                "asset_id": asset,
                "window_s": w,
                "window_complete": complete,
                "split": "test",
                "canonical_class": "recreational",
                "placement_class": "anchorage",
                "port_region": "miami_approach",
            }
            for c in CORE + GEOM:
                row[c] = 0.0
            rows.append(row)
    tables = {
        300: pd.DataFrame([r for r in rows if r["window_s"] == 300]),
        600: pd.DataFrame([r for r in rows if r["window_s"] == 600]),
    }
    assigned = assign_multihorizon_encounters(
        bundle, tables, {"canonical_class": ["recreational"]}, "test",
    )
    assert len(assigned) == 2
    by_asset = assigned.set_index("asset_id")["assigned_window_s"].to_dict()
    assert by_asset["asset_a"] == 600
    assert by_asset["asset_b"] == 300


def test_stratum_far_groups_by_placement():
    detail = pd.DataFrame({
        "placement_class": ["anchorage", "anchorage", "fairway_stress"],
        "flag_far_0.05": [True, False, True],
    })
    rows = stratum_far(detail, group_cols=["placement_class"])
    by = {r["placement_class"]: r for r in rows}
    assert by["anchorage"]["n_scored"] == 2
    assert by["anchorage"]["far"] == pytest.approx(0.5)
    assert by["fairway_stress"]["far"] == pytest.approx(1.0)


def test_placements_digest_matches_freeze():
    freeze_path = REPO_ROOT / "results" / "behavior_model_geometry" / "FROZEN.json"
    if not freeze_path.is_file():
        pytest.skip("geometry freeze not present")
    freeze = json.loads(freeze_path.read_text())
    pin = verify_placements_digest(freeze)
    assert pin["ok"] is True
