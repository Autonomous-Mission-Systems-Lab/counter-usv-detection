"""Unit tests for the perfect-disguise oracle (no-patch class assertion)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.oracle import (  # noqa: E402
    PerfectDisguiseOracle,
    assertions_to_records,
    build_assertions_for_contacts,
    load_oracle_config,
)
from counterusv.models.detector import Detection  # noqa: E402


def test_load_oracle_config():
    cfg = load_oracle_config()
    assert cfg.true_class == "usv"
    assert "fishing" in cfg.target_benign_classes
    assert "recreational" in cfg.target_benign_classes
    assert cfg.condition == "perfect_disguise_oracle"
    assert cfg.assertion_score == pytest.approx(1.0)


def test_assert_class_basic():
    oracle = PerfectDisguiseOracle.from_config()
    a = oracle.assert_class("fishing", contact_id=42)
    assert a.true_class == "usv"
    assert a.asserted_class == "fishing"
    assert a.role == "benign"
    assert a.condition == "perfect_disguise_oracle"
    assert a.score == pytest.approx(1.0)
    assert a.contact_id == 42
    det = a.as_detection()
    assert det.class_name == "fishing"
    assert det.role == "benign"
    assert det.score == pytest.approx(1.0)


def test_reject_unknown_benign():
    oracle = PerfectDisguiseOracle.from_config()
    with pytest.raises(ValueError, match="not in"):
        oracle.assert_class("cargo_merchant")


def test_apply_to_detection_preserves_box():
    oracle = PerfectDisguiseOracle.from_config()
    src = Detection(
        box_xyxy=(10.0, 20.0, 50.0, 80.0),
        score=0.77,
        class_name="usv",
        class_id=0,
        role="hostile",
    )
    a = oracle.apply_to_detection(src, "recreational", contact_id="c1")
    assert a.asserted_class == "recreational"
    assert a.box_xyxy == (10.0, 20.0, 50.0, 80.0)
    assert a.source_detection is not None
    assert a.source_detection["class_name"] == "usv"
    # Oracle overrides score to assertion_score (perfect label).
    assert a.as_detection().score == pytest.approx(1.0)


def test_build_assertions_and_records():
    oracle = PerfectDisguiseOracle.from_config()
    contacts = [
        {"image_id": 1, "box_xyxy": [0, 0, 10, 10]},
        {"contact_id": "t2", "target_xyxy": (1.0, 2.0, 3.0, 4.0)},
    ]
    assertions = build_assertions_for_contacts(oracle, contacts, "fishing")
    assert len(assertions) == 2
    assert assertions[0].contact_id == 1
    assert assertions[1].contact_id == "t2"
    assert assertions[1].box_xyxy == (1.0, 2.0, 3.0, 4.0)
    records = assertions_to_records(assertions)
    assert records[0]["asserted_class"] == "fishing"
    assert records[0]["true_class"] == "usv"
    assert records[0]["condition"] == "perfect_disguise_oracle"


def test_asserted_class_name():
    oracle = PerfectDisguiseOracle.from_config()
    assert oracle.asserted_class_name("recreational") == "recreational"
