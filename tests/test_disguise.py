"""Unit tests for the disguise attack (TMSR)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.disguise import (  # noqa: E402
    DisguiseInstanceResult,
    aggregate_tmsr,
    load_disguise_config,
    make_disguise_loss,
    score_instance_tmsr,
)
from counterusv.models.detector import Detection  # noqa: E402


class FakeSurrogate:
    device = torch.device("cpu")

    def __init__(self, centers):
        self.centers = centers
        self.name_to_idx = {"usv": 0, "fishing": 1, "recreational": 2}

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
        # Brightness drives "usv"; darkness drives "fishing" (toy).
        mean = x.mean().clamp(0, 1)
        scores = torch.zeros(b, 3, a)
        scores[:, 0, :] = mean
        scores[:, 1, :] = 1.0 - mean
        scores[:, 2, :] = 0.1
        return boxes, scores


def _det(name, box, score, role="benign"):
    return Detection(
        box_xyxy=tuple(box), score=score, class_name=name,
        class_id=0, role=role,
    )


def test_disguise_loss_has_grad():
    sur = FakeSurrogate([(150.0, 150.0)])
    loss_fn = make_disguise_loss(sur, (100, 100, 200, 200), 0, 1)
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss = loss_fn(x)
    loss.backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_disguise_loss_prefers_raising_benign():
    sur = FakeSurrogate([(150.0, 150.0)])
    loss_fn = make_disguise_loss(sur, (100, 100, 200, 200), 0, 1)
    # Bright → high usv / low fishing → higher loss; dark → opposite.
    bright = torch.full((1, 3, 16, 16), 0.9)
    dark = torch.full((1, 3, 16, 16), 0.1)
    assert float(loss_fn(bright)) > float(loss_fn(dark))


def test_score_tmsr_success_and_eligible():
    tgt = (100, 100, 200, 200)
    clean = np.zeros((64, 64, 3), np.uint8)
    attacked = np.ones((64, 64, 3), np.uint8)

    def predict_fn(arr):
        if arr is clean:
            return [_det("usv", tgt, 0.9, "hostile")]
        return [_det("fishing", tgt, 0.8, "benign")]

    res = score_instance_tmsr(
        predict_fn, clean, attacked, tgt,
        true_class="usv", benign_class="fishing",
        conf=0.25, iou_match=0.5,
        marine_eot=None, severity_axes=["scale"], severity_levels=["L0"],
        image_id=1,
    )
    assert res.clean_hostile is True
    assert res.tmsr_flags()["scale:L0"] is True


def test_score_tmsr_not_eligible_without_clean_hostile():
    tgt = (100, 100, 200, 200)
    clean = np.zeros((64, 64, 3), np.uint8)
    res = score_instance_tmsr(
        lambda arr: [], clean, clean, tgt,
        true_class="usv", benign_class="fishing",
        conf=0.25, iou_match=0.5,
        marine_eot=None, severity_axes=["scale"], severity_levels=["L0"],
    )
    assert res.clean_hostile is False
    assert res.tmsr_flags()["scale:L0"] is False


def test_aggregate_tmsr_patch_attributable():
    def mk(tmsr, clean_benign_t):
        return DisguiseInstanceResult(
            image_id=0, benign_class="fishing", target_xyxy=(0, 0, 1, 1),
            clean_hostile=True, clean_hostile_score=0.9,
            attacked={"scale:L0": {
                "tmsr": tmsr, "benign_score": None, "still_true": False,
                "clean_benign_under_transform": clean_benign_t,
            }},
            steps=1, attack_loss_init=None, attack_loss_final=None,
        )
    results = [
        mk(True, False),   # patch success, transform alone did not flip
        mk(True, True),    # success but transform already benign → exclude from PA
        mk(False, False),  # fail
    ]
    agg = aggregate_tmsr(results)["scale:L0"]
    assert agg["n_eligible"] == 3
    assert agg["n_success"] == 2
    assert agg["tmsr"] == pytest.approx(2 / 3)
    assert agg["n_patch_attributable_denom"] == 2
    assert agg["tmsr_patch_attributable"] == pytest.approx(0.5)


def test_load_disguise_config():
    cfg = load_disguise_config()
    assert cfg.true_class == "usv"
    assert "fishing" in cfg.target_benign_classes
    assert "recreational" in cfg.target_benign_classes
    assert cfg.severity_levels[0] == "L0"
