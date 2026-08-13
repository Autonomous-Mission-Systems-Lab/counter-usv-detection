"""Unit tests for the engagement-geometry spec."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.defense.engagement import (  # noqa: E402
    load_engagement_geometry,
)


def test_load_engagement_geometry():
    cfg = load_engagement_geometry()
    assert cfg.version == 1
    assert cfg.max_range_nm == pytest.approx(6.0)
    assert cfg.min_range_nm == pytest.approx(0.25)
    assert cfg.expected_cadence_s == pytest.approx(60.0)
    assert cfg.placement_policy.get("report_as") == "distribution"
    assert cfg.placement_policy.get("pairing_radius_nm") == 100


def test_port_regions_and_placement_classes():
    cfg = load_engagement_geometry()
    ports = cfg.port_regions()
    assert len(ports) == 5
    assert cfg.placement_policy.get("n_ports") == 5
    ids = {p.id for p in ports}
    assert ids == {
        "miami_approach",
        "mississippi_delta",
        "puget_sound",
        "ny_harbor",
        "la_san_pedro",
    }
    classes = {c.name for c in cfg.placement_classes()}
    assert classes == {
        "berth_approach",
        "anchorage",
        "offshore_terminal",
        "fairway_stress",
    }


def test_fit_population_is_realistic_assets_only():
    cfg = load_engagement_geometry()
    assert set(cfg.fit_population()) == {"berth_approach", "anchorage"}
    by_name = {c.name: c for c in cfg.placement_classes()}
    # The fairway is a deliberate stress placement, not a real asset: it must
    # never leak into the training population.
    assert by_name["fairway_stress"].is_fit is False
    assert by_name["offshore_terminal"].is_fit is False
    assert all(by_name[n].represents for n in by_name)


def test_annulus_helpers():
    cfg = load_engagement_geometry()
    assert cfg.geometry_scoreable(5.0) is True
    assert cfg.geometry_scoreable(6.0) is True
    assert cfg.geometry_scoreable(6.1) is False
    assert cfg.in_annulus(1.0) is True
    assert cfg.in_annulus(0.1) is False  # inside min_range terminal band edge
