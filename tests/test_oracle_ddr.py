"""Unit tests for oracle DDR helpers (gap, NC gate, anti-DDR freeze)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.defense.presence import (  # noqa: E402
    PresenceOnlyDefense,
    presence_for_disguise,
)
from counterusv.eval.oracle_ddr import (  # noqa: E402
    assert_freeze_has_no_ddr,
    cell_kind,
    ddr_table,
    defensibility_gap,
    extract_windows,
    gap_table,
    has_ddr_payload,
    load_oracle_ddr_config,
    nc_sanity,
    rate_from_actions,
)
from counterusv.attacks.oracle import OracleAssertion  # noqa: E402


def test_load_oracle_ddr_config():
    cfg = load_oracle_ddr_config()
    assert cfg["condition"] == "perfect_disguise_oracle"
    assert "kinematics_only" in cfg["arms"]
    assert "kinematics_geometry" in cfg["arms"]
    assert cfg["arms"]["kinematics_only"]["windows_s"] == [120, 180, 300]
    assert cfg.get("envelope_routing", "conditional") == "conditional"
    assert cfg["arms"]["kinematics_geometry"]["windows_s"] == [180, 300, 600]


def test_rate_and_gap():
    stats = rate_from_actions(["flag", "flag", "pass", "abstain"])
    assert stats["n"] == 3
    assert stats["n_flag"] == 2
    assert stats["ddr"] == pytest.approx(2 / 3)
    assert defensibility_gap(0.8, 0.0) == pytest.approx(0.8)
    assert defensibility_gap(None, 0.0) is None


def test_ddr_table_excludes_nc():
    rows = [
        {"mimicked_class": "fishing", "arm": "kinematics_only",
         "defense_kind": "consistency", "action": "flag",
         "negative_control": False},
        {"mimicked_class": "fishing", "arm": "kinematics_only",
         "defense_kind": "consistency", "action": "pass",
         "negative_control": False},
        {"mimicked_class": "fishing", "arm": "kinematics_only",
         "defense_kind": "consistency", "action": "pass",
         "negative_control": True},
    ]
    tab = ddr_table(rows, thin_n_threshold=20)
    assert len(tab) == 1
    assert tab.iloc[0]["n"] == 2
    assert tab.iloc[0]["ddr"] == pytest.approx(0.5)
    assert tab.iloc[0]["thin_n"] == "thin-n"


def test_gap_table_broadcasts_presence():
    cons = pd.DataFrame([
        {"mimicked_class": "fishing", "arm": "kinematics_only",
         "defense_kind": "consistency", "ddr": 0.8, "n": 10, "thin_n": None},
        {"mimicked_class": "fishing", "arm": "kinematics_geometry",
         "defense_kind": "consistency", "ddr": 0.9, "n": 10, "thin_n": None},
    ])
    pres = pd.DataFrame([
        {"mimicked_class": "fishing", "defense_kind": "presence_only",
         "ddr": 0.0, "n": 10, "thin_n": None},
    ])
    g = gap_table(cons, pres)
    assert len(g) == 2
    assert all(g["presence_ddr"] == 0.0)
    assert list(g["gap"]) == pytest.approx([0.8, 0.9])


def test_nc_sanity_within_tol():
    rows = [
        {"arm": "kinematics_only", "defense_kind": "consistency",
         "action": "pass", "negative_control": True},
        {"arm": "kinematics_only", "defense_kind": "consistency",
         "action": "pass", "negative_control": True},
        {"arm": "kinematics_only", "defense_kind": "consistency",
         "action": "flag", "negative_control": True},
    ]
    # 1/3 ≈ 33% vs FAR 3.5% with factor 2 + 0.05 → ceiling 12% → FAIL
    bad = nc_sanity(rows, far=0.035, arm="kinematics_only", factor=2.0, slack=0.05)
    assert bad["within_tol"] is False
    # All pass → within tol
    good_rows = [{**r, "action": "pass"} for r in rows]
    good = nc_sanity(good_rows, far=0.035, arm="kinematics_only")
    assert good["within_tol"] is True
    assert good["ddr"] == 0.0


def test_has_ddr_payload_and_assert_freeze(tmp_path: Path):
    clean = {"version": "1", "n_cells": 10, "note": "no DDR here"}
    dirty = {"version": "1", "ddr": 0.5, "nested": {"detection_rate": 0.1}}
    assert has_ddr_payload(clean) is False
    assert has_ddr_payload(dirty) is True

    p = tmp_path / "FROZEN_SWEEP.json"
    p.write_text(json.dumps(clean))
    loaded = assert_freeze_has_no_ddr(p)
    assert loaded["n_cells"] == 10

    p.write_text(json.dumps(dirty))
    with pytest.raises(ValueError, match="anti-circularity"):
        assert_freeze_has_no_ddr(p)


def test_presence_always_passes_disguise():
    defense = PresenceOnlyDefense.from_config()
    oracle = OracleAssertion(
        contact_id="t1",
        true_class="usv",
        asserted_class="fishing",
        condition="perfect_disguise_oracle",
        score=1.0,
        role="benign",
        box_xyxy=(0.0, 0.0, 5.0, 5.0),
        class_id=1,
    )
    d = defense.evaluate(
        oracle, presence=presence_for_disguise(oracle), purpose="eval"
    )
    assert d.action == "pass"
    assert d.defense_kind == "presence_only"


def test_cell_kind():
    assert cell_kind({"negative_control": True, "unconstrained": False}) == "nc"
    assert cell_kind({"negative_control": False, "unconstrained": True}) == "unc"
    assert cell_kind({"negative_control": False, "unconstrained": False}) == "swp"


def test_extract_windows_roundtrip():
    # Straight inbound track at ~20 kn for 700 s @ 60 s cadence.
    n = 15
    t = np.arange(n, dtype=float) * 60.0
    # ~0.00028 deg lat ≈ 0.017 nm per step → ~1 kn; use larger steps for 20 kn.
    # 20 kn ≈ 20/60 nm per minute ≈ 0.333 nm/min ≈ 0.00556 deg lat/min.
    lat = 25.7 + np.arange(n) * 0.00556
    lon = np.full(n, -80.1)
    sog = np.full(n, 20.0)
    cog = np.full(n, 0.0)
    pts = pd.DataFrame({
        "trip_id": ["t0"] * n,
        "t": t, "lat": lat, "lon": lon, "sog": sog, "cog": cog,
    })
    by_w, complete = extract_windows(pts, [120, 180, 300], join_geometry=False)
    assert 300 in complete
    assert "sog_p95" in by_w[300]
    assert by_w[300]["sog_p95"] == pytest.approx(20.0, abs=0.5)
