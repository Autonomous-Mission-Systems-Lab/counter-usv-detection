"""Unit tests for the physically-realizable patch core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from counterusv.attacks.patch import (
    PatchCore,
    apply_patch_torch,
    filter_patch_eligible,
    is_patch_eligible,
    load_patch_config,
    non_printability_score,
    place_on_bbox,
    total_variation,
    xywh_to_xyxy,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "attacks" / "patch.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_patch_config(CONFIG)


@pytest.fixture
def core(cfg):
    return PatchCore(cfg)


def test_eligibility_floor(cfg):
    assert is_patch_eligible([0, 0, 40, 40], min_side=cfg.patch_min_side)
    assert not is_patch_eligible([0, 0, 20, 40], min_side=cfg.patch_min_side)
    kept = filter_patch_eligible(
        [[0, 0, 20, 20], [10, 10, 48, 48]], min_side=cfg.patch_min_side
    )
    assert len(kept) == 1
    assert kept[0, 2] == 48


def test_placement_inside_image(cfg):
    box = (100.0, 80.0, 220.0, 180.0)  # 120×100
    p = place_on_bbox(box, (256, 256), cfg, rng=None)
    assert p.side >= int(cfg.geometry["side_px_min"])
    assert p.x0 >= 0 and p.y0 >= 0
    assert p.x0 + p.side <= 256
    assert p.y0 + p.side <= 256
    # Hull anchor is below vertical mid of the box.
    cy = p.y0 + p.side / 2.0
    assert cy > (80 + 180) / 2.0 - 5


def test_apply_changes_region(cfg):
    img = torch.zeros(3, 128, 128)
    patch = torch.ones(3, 64, 64)
    place = place_on_bbox((32, 32, 96, 96), (128, 128), cfg)
    out = apply_patch_torch(img, patch, place, feather_frac=0.0)
    # Center of placement should be near-white.
    cy = place.y0 + place.side // 2
    cx = place.x0 + place.side // 2
    assert float(out[:, cy, cx].mean()) > 0.9
    # Far corner untouched.
    assert float(out[:, 0, 0].mean()) == 0.0


def test_tv_and_nps_finite(core):
    patch = core.init_patch(device="cpu")
    regs = core.regularizer_loss(patch)
    assert torch.isfinite(regs["tv"])
    assert torch.isfinite(regs["nps"])
    assert regs["reg"].ndim == 0


def test_nps_zero_on_palette_color(cfg):
    colors = cfg.regularization["printable_colors_rgb"]
    # Pure black is in the palette → NPS ≈ 0.
    patch = torch.zeros(3, 16, 16)
    printable = torch.tensor(
        [[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] for c in colors],
        dtype=torch.float32,
    )
    assert float(non_printability_score(patch, printable)) < 1e-6


def test_optimize_dummy_objective_reduces_loss(core):
    """Brighten the frame under the patch (no detector) — core wiring check."""
    torch.manual_seed(0)
    img = torch.rand(3, 160, 160) * 0.3
    box = (40.0, 40.0, 120.0, 120.0)
    patch = core.init_patch(device="cpu")
    # Placement jitter only (no marine-EOT) for a cheap deterministic smoke.
    core.eot = None

    def attack_loss(patched: torch.Tensor) -> torch.Tensor:
        return -patched.mean()

    losses = []
    for m in core.optimize(img, box, patch, attack_loss, steps=8, lr=0.1):
        losses.append(m["attack"])
    assert len(losses) == 8
    assert losses[-1] < losses[0]
    assert float(patch.detach().min()) >= 0.0 - 1e-5
    assert float(patch.detach().max()) <= 1.0 + 1e-5


def test_eot_expectation_has_grad(core):
    img = torch.rand(3, 96, 96)
    box = xywh_to_xyxy([20, 20, 50, 50])
    patch = core.init_patch(device="cpu")
    assert is_patch_eligible([20, 20, 50, 50], min_side=32)

    def attack_loss(patched: torch.Tensor) -> torch.Tensor:
        return patched.pow(2).mean()

    loss, _ = core.eot_expectation(img, patch, box, attack_loss, n_samples=2)
    loss.backward()
    assert patch.grad is not None
    assert torch.isfinite(patch.grad).all()
    assert float(total_variation(patch).detach()) >= 0.0
