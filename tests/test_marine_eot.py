"""Unit tests for the marine-EOT transform library."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from counterusv.attacks.marine_eot import (
    AXES,
    SEVERITY_LEVELS,
    MarineEOT,
    apply_numpy,
    apply_torch,
    load_config,
    sample_params,
    severity_params,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "attacks" / "marine_eot.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture
def rgb():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (64, 80, 3), dtype=np.uint8)


def test_config_axes_and_levels(cfg):
    assert cfg.severity_levels == SEVERITY_LEVELS
    for axis in AXES:
        assert axis in cfg.axes
        for level in SEVERITY_LEVELS:
            assert level in cfg.axes[axis]["severity"]


def test_l0_is_near_identity(cfg, rgb):
    for axis in AXES:
        out = apply_numpy(rgb, severity_params(cfg, axis, "L0"), border_value=cfg.border_value)
        # L0 may still re-encode through float; allow tiny rounding.
        assert out.shape == rgb.shape
        assert out.dtype == np.uint8
        # Spray/sea/glare at zero coverage should match exactly; geometric L0 too.
        np.testing.assert_allclose(out.astype(float), rgb.astype(float), atol=1.0)


def test_severity_changes_image(cfg, rgb):
    for axis in AXES:
        a = apply_numpy(rgb, severity_params(cfg, axis, "L0"), border_value=cfg.border_value)
        b = apply_numpy(rgb, severity_params(cfg, axis, "L4"), border_value=cfg.border_value)
        assert a.shape == b.shape
        assert not np.array_equal(a, b), f"{axis}: L0 and L4 should differ"


def test_sample_reproducible(cfg, rgb):
    rng1 = np.random.default_rng(99)
    rng2 = np.random.default_rng(99)
    p1 = sample_params(cfg, rng1)
    p2 = sample_params(cfg, rng2)
    assert p1.scale == p2.scale
    assert p1.rotation_deg == p2.rotation_deg
    assert p1.spray_seed == p2.spray_seed
    o1 = apply_numpy(rgb, p1, border_value=cfg.border_value)
    o2 = apply_numpy(rgb, p2, border_value=cfg.border_value)
    np.testing.assert_array_equal(o1, o2)


def test_wrapper_apply_axis(rgb):
    eot = MarineEOT.from_config(CONFIG)
    out = eot.apply_axis(rgb, "motion_blur", "L3")
    assert out.shape == rgb.shape


def test_torch_matches_numpy_shape_and_grad(cfg, rgb):
    eot = MarineEOT.from_config(CONFIG)
    params = eot.severity_params("glare", "L3")
    t = torch.tensor(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t = t.requires_grad_(True)
    y = apply_torch(t, params, border_value=cfg.border_value)
    assert y.shape == t.shape
    y.mean().backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_torch_geometric_grad(cfg, rgb):
    params = severity_params(cfg, "rotation", "L2")
    t = torch.tensor(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    t.requires_grad_(True)
    y = apply_torch(t, params, border_value=cfg.border_value)
    y.sum().backward()
    assert t.grad is not None
    # Rotation should route gradient to most pixels.
    assert (t.grad.abs() > 0).float().mean() > 0.5
