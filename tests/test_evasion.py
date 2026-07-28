"""Unit tests for the evasion attack (ESR).

The differentiable surrogate is faked so these run without detector weights or
ultralytics; a full-model wiring check lives in the run script's smoke path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.evasion import (  # noqa: E402
    EvasionInstanceResult,
    aggregate_esr,
    find_target_detection,
    iou_xyxy,
    load_evasion_config,
    make_evasion_loss,
    score_instance_esr,
    target_confidence,
)
from counterusv.models.detector import Detection  # noqa: E402


class FakeSurrogate:
    """Minimal differentiable stand-in: scores depend on the input mean."""

    device = torch.device("cpu")

    def __init__(self, centers, name_to_idx=None):
        self.centers = centers  # list of (cx, cy) in canvas pixels
        self.name_to_idx = name_to_idx or {"usv": 0}

    def class_index(self, name):
        return self.name_to_idx[name]

    def forward_scores(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b = x.shape[0]
        a = len(self.centers)
        boxes = torch.zeros(b, 4, a)
        for i, (cx, cy) in enumerate(self.centers):
            boxes[:, 0, i] = cx
            boxes[:, 1, i] = cy
        # Confidence rises with brightness so a grad exists w.r.t. the patch.
        conf = x.mean().clamp(0, 1)
        scores = conf.expand(b, 1, a).clone()
        return boxes, scores


def _usv(box_xyxy, score):
    return Detection(box_xyxy=tuple(box_xyxy), score=score, class_name="usv",
                     class_id=8, role="hostile")


# --------------------------------------------------------------------------- #
# geometry / matching
# --------------------------------------------------------------------------- #


def test_iou_basic():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_find_target_detection_filters_class_conf_iou():
    tgt = (100, 100, 200, 200)
    dets = [
        _usv((100, 100, 200, 200), 0.9),          # good match
        _usv((100, 100, 200, 200), 0.10),         # below conf
        Detection((100, 100, 200, 200), 0.9, "fishing", 0, "benign"),  # wrong class
        _usv((300, 300, 400, 400), 0.9),          # no overlap
    ]
    hit = find_target_detection(dets, tgt, "usv", conf=0.25, iou_match=0.5)
    assert hit is not None and hit.score == pytest.approx(0.9)
    assert find_target_detection([], tgt, "usv", conf=0.25, iou_match=0.5) is None


# --------------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------------- #


def test_evasion_loss_has_grad_to_patch():
    surrogate = FakeSurrogate(centers=[(150.0, 150.0)])
    loss_fn = make_evasion_loss(surrogate, (100, 100, 200, 200), 0, temperature=20.0)
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    loss = loss_fn(x)
    loss.backward()
    assert x.grad is not None and torch.isfinite(loss)
    assert float(x.grad.abs().sum()) > 0


def test_evasion_loss_tracks_inbox_confidence():
    surrogate = FakeSurrogate(centers=[(150.0, 150.0)])
    loss_fn = make_evasion_loss(surrogate, (100, 100, 200, 200), 0, temperature=30.0)
    dim = torch.full((1, 3, 32, 32), 0.1)
    bright = torch.full((1, 3, 32, 32), 0.9)
    # Higher in-box confidence (brighter) → higher loss (we minimize it).
    assert float(loss_fn(bright)) > float(loss_fn(dim))


def test_evasion_loss_fallback_when_no_anchor_inside():
    # Anchor center is OUTSIDE the target; fallback uses the global max.
    surrogate = FakeSurrogate(centers=[(500.0, 500.0)])
    x = torch.full((1, 3, 32, 32), 0.7)
    with_fb = make_evasion_loss(surrogate, (0, 0, 50, 50), 0,
                                temperature=30.0, fallback_global_max=True)
    no_fb = make_evasion_loss(surrogate, (0, 0, 50, 50), 0,
                              temperature=30.0, fallback_global_max=False)
    # With fallback the loss reflects the (0.7) global conf; without, it is the
    # masked -30 floor → far smaller.
    assert float(with_fb(x)) > float(no_fb(x))
    assert float(with_fb(x)) == pytest.approx(0.7, abs=1e-2)


def test_target_confidence_reads_inbox_max():
    surrogate = FakeSurrogate(centers=[(150.0, 150.0)])
    x = torch.full((1, 3, 16, 16), 0.42)
    c = target_confidence(surrogate, x, (100, 100, 200, 200), 0)
    assert c == pytest.approx(0.42, abs=1e-3)


# --------------------------------------------------------------------------- #
# ESR scoring / aggregation
# --------------------------------------------------------------------------- #


def test_score_instance_esr_clean_hit_attacked_suppressed():
    tgt = (100, 100, 200, 200)
    clean = np.zeros((64, 64, 3), np.uint8)
    attacked = np.ones((64, 64, 3), np.uint8)

    def predict_fn(arr):
        # Detect the target on the clean canvas only.
        return [_usv(tgt, 0.9)] if arr is clean else []

    res = score_instance_esr(
        predict_fn, clean, attacked, tgt,
        class_name="usv", conf=0.25, iou_match=0.5,
        marine_eot=None, severity_axes=["scale"], severity_levels=["L0", "L4"],
        image_id=7,
    )
    assert res.clean_detected is True
    flags = res.esr_flags()
    assert flags["scale:L0"] is True and flags["scale:L4"] is True


def test_score_instance_esr_not_attackable_when_clean_missed():
    tgt = (100, 100, 200, 200)
    clean = np.zeros((64, 64, 3), np.uint8)

    res = score_instance_esr(
        lambda arr: [], clean, clean, tgt,
        class_name="usv", conf=0.25, iou_match=0.5,
        marine_eot=None, severity_axes=["scale"], severity_levels=["L0"],
        image_id=1,
    )
    assert res.clean_detected is False
    assert res.esr_flags()["scale:L0"] is False


def test_aggregate_esr_uses_attackable_denominator():
    def mk(clean, l0_detected):
        return EvasionInstanceResult(
            image_id=0, target_xyxy=(0, 0, 1, 1), clean_detected=clean,
            clean_score=0.9 if clean else None,
            attacked={"scale:L0": {"detected": l0_detected, "score": None}},
            steps=1, attack_loss_init=None, attack_loss_final=None,
        )
    results = [
        mk(True, False),   # attackable, suppressed → success
        mk(True, True),    # attackable, still detected → fail
        mk(False, True),   # not attackable → excluded from denominator
    ]
    agg = aggregate_esr(results)
    assert agg["scale:L0"]["n_attackable"] == 2
    assert agg["scale:L0"]["n_success"] == 1
    assert agg["scale:L0"]["esr"] == pytest.approx(0.5)


def test_aggregate_esr_patch_attributable_excludes_transform_suppressed():
    def mk(attacked_detected, clean_under_transform):
        return EvasionInstanceResult(
            image_id=0, target_xyxy=(0, 0, 1, 1), clean_detected=True,
            clean_score=0.9,
            attacked={"scale:L4": {
                "detected": attacked_detected, "score": None,
                "clean_detected_under_transform": clean_under_transform,
            }},
            steps=1, attack_loss_init=None, attack_loss_final=None,
        )
    results = [
        mk(False, True),    # patch suppressed a target the transform left visible
        mk(False, False),   # suppressed, but transform alone already hid it → excluded
        mk(True, True),     # transform-visible but patch failed → counts as fail
    ]
    agg = aggregate_esr(results)["scale:L4"]
    # Raw ESR: 2 of 3 suppressed.
    assert agg["esr"] == pytest.approx(2 / 3)
    # Patch-attributable: denom excludes the transform-suppressed one (2 left),
    # 1 of those 2 suppressed by the patch.
    assert agg["n_patch_attributable_denom"] == 2
    assert agg["esr_patch_attributable"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_load_evasion_config_reads_repo_yaml():
    cfg = load_evasion_config()
    assert cfg.target_class == "usv"
    assert 0.0 < cfg.conf_threshold < 1.0
    assert "scale" in cfg.severity_axes
    assert cfg.severity_levels[0] == "L0"


def test_to_torch_device_maps_ultralytics_gpu_index():
    from counterusv.attacks.evasion import to_torch_device, to_ultralytics_device
    # Without CUDA these still map the *string* forms deterministically when
    # resolve_device returns them as-is (explicit request, not auto).
    assert to_torch_device("cpu").type == "cpu"
    assert to_ultralytics_device(torch.device("cpu")) == "cpu"
    # Explicit cuda index string → cuda:N (even if unavailable — torch.device accepts it).
    d = to_torch_device("0")
    assert d.type == "cuda" and (d.index == 0 or d.index is None)
    assert to_ultralytics_device(d) == "0"
