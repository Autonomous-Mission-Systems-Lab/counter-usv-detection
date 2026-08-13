"""Unit tests for access-level transfer helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.patch import Placement  # noqa: E402
from counterusv.attacks.transfer import (  # noqa: E402
    access_level,
    base_rate_from_agg,
    default_transfer_targets,
    load_access_levels_config,
    load_patch_bank,
    save_patch_bank_entry,
    transfer_gap,
    write_patch_bank_manifest,
)


def test_access_level_map():
    cfg = load_access_levels_config()
    assert access_level("yolo11s", "yolo11s", cfg) == "white"
    assert access_level("yolo11s", "yolo11l", cfg) == "grey"
    assert access_level("yolo11l", "yolo11s", cfg) == "grey"
    assert access_level("yolo11s", "rtdetr_l", cfg) == "black"
    assert access_level("yolo11l", "rtdetr_l", cfg) == "black"


def test_default_transfer_targets():
    cfg = load_access_levels_config()
    assert default_transfer_targets("yolo11s", cfg) == ["yolo11l", "rtdetr_l"]
    assert default_transfer_targets("yolo11s", cfg, include_white=True) == [
        "yolo11s", "yolo11l", "rtdetr_l",
    ]


def test_transfer_gap_and_base_rate():
    assert transfer_gap(0.18, 0.05) == pytest.approx(-0.13)
    assert transfer_gap(None, 0.1) is None
    agg = {
        "scale:L0": {"esr": 0.18, "n_success": 6, "n_attackable": 33},
        "scale:L1": {"esr": 0.45, "n_success": 15, "n_attackable": 33},
    }
    base = base_rate_from_agg(agg, rate_key="esr", axes=["scale"])
    assert base is not None
    assert base["esr"] == pytest.approx(0.18)


def test_patch_bank_roundtrip(tmp_path: Path):
    bank = tmp_path / "patch_bank"
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[10:20, 10:20] = 200
    patch = np.random.rand(3, 16, 16).astype(np.float32)
    placement = Placement(x0=10, y0=10, side=16, size_frac=0.35)
    row = save_patch_bank_entry(
        bank,
        image_id=25029,
        target_xyxy=(5.0, 5.0, 40.0, 40.0),
        placement=placement,
        attacked_rgb=rgb,
        patch_chw=patch,
    )
    write_patch_bank_manifest(
        bank,
        attack="evasion",
        surrogate="yolo11s",
        instances=[row],
        extra={"steps": 150},
    )
    man, entries = load_patch_bank(bank)
    assert man["surrogate"] == "yolo11s"
    assert man["n_instances"] == 1
    assert len(entries) == 1
    e = entries[0]
    assert e.image_id == 25029
    assert e.placement.side == 16
    assert e.attacked_rgb.shape == (64, 64, 3)
    assert e.patch_chw.shape == (3, 16, 16)
    assert np.allclose(e.patch_chw, patch, atol=1e-5)


def test_load_patch_bank_missing_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_patch_bank(tmp_path / "empty")
