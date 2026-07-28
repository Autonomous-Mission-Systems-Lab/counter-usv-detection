"""Disguise attack (TMSR) — targeted hostile→benign misclassification.

Layers a disguise objective on the shared patch core
(:mod:`counterusv.attacks.patch`) and marine-EOT expectation. The patch is
optimized white-box so predictions overlapping the target box raise a chosen
**benign** class (fishing / recreational) while suppressing the true hostile
class (``usv``), keeping the contact detected under the benign label.

Success is **TMSR** (``docs/METRICS.md``): clean prediction was the correct
hostile class; attacked canvas still has a box IoU ≥ 0.5 with GT whose predicted
class is the target benign class. Reported per benign class and across the
marine-EOT severity ladder (raw + patch-attributable).

Config: ``configs/attacks/disguise.yaml``. Reuses
:class:`~counterusv.attacks.evasion.DifferentiableSurrogate` and hard-detector
helpers from the evasion module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from counterusv.attacks.evasion import (
    DifferentiableSurrogate,
    PredictFn,
    find_target_detection,
    make_predict_fn,
    to_torch_device,
    to_ultralytics_device,
)
from counterusv.attacks.patch import AttackLossFn, PatchCore, load_patch_config

DEFAULT_CONFIG = Path("configs/attacks/disguise.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisguiseConfig:
    version: int
    frozen: str
    surrogate: dict[str, Any]
    target_benign_classes: tuple[str, ...]
    objective: dict[str, Any]
    optimization: dict[str, Any]
    evaluation: dict[str, Any]
    seed: int
    path: Path | None = None

    @property
    def surrogate_family(self) -> str:
        return str(self.surrogate.get("family", "yolo11s"))

    @property
    def true_class(self) -> str:
        return str(self.surrogate.get("true_class", "usv"))

    @property
    def device(self) -> str:
        return str(self.surrogate.get("device", "auto"))

    @property
    def target_pad_frac(self) -> float:
        return float(self.objective.get("target_pad_frac", 0.15))

    @property
    def smoothmax_temperature(self) -> float:
        return float(self.objective.get("smoothmax_temperature", 20.0))

    @property
    def fallback_global_max(self) -> bool:
        return bool(self.objective.get("fallback_global_max", True))

    @property
    def true_weight(self) -> float:
        return float(self.objective.get("true_weight", 1.0))

    @property
    def benign_weight(self) -> float:
        return float(self.objective.get("benign_weight", 1.0))

    @property
    def steps(self) -> int | None:
        v = self.optimization.get("steps")
        return int(v) if v is not None else None

    @property
    def lr(self) -> float | None:
        v = self.optimization.get("lr")
        return float(v) if v is not None else None

    @property
    def conf_threshold(self) -> float:
        return float(self.evaluation.get("conf_threshold", 0.25))

    @property
    def iou_match(self) -> float:
        return float(self.evaluation.get("iou_match", 0.5))

    @property
    def nms_iou(self) -> float:
        return float(self.evaluation.get("nms_iou", 0.7))

    @property
    def severity_axes(self) -> list[str]:
        return list(self.evaluation.get("severity_axes") or [])

    @property
    def severity_levels(self) -> list[str]:
        return list(self.evaluation.get("severity_levels") or [])


def load_disguise_config(path: str | Path | None = None) -> DisguiseConfig:
    p = Path(path) if path is not None else _REPO_ROOT / DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path(DEFAULT_CONFIG)
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"disguise config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    benign = tuple(raw.get("target_benign_classes") or ["fishing", "recreational"])
    return DisguiseConfig(
        version=int(raw["version"]),
        frozen=str(raw["frozen"]),
        surrogate=dict(raw.get("surrogate") or {}),
        target_benign_classes=benign,
        objective=dict(raw.get("objective") or {}),
        optimization=dict(raw.get("optimization") or {}),
        evaluation=dict(raw.get("evaluation") or {}),
        seed=int(raw.get("seed", 1337)),
        path=p.resolve(),
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _smoothmax_inbox(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_idx: int,
    *,
    ex1: float,
    ey1: float,
    ex2: float,
    ey2: float,
    temperature: float,
    fallback_global_max: bool,
) -> torch.Tensor:
    """Smooth-max of ``class_idx`` conf over anchors whose center is in-box."""
    cx, cy = boxes[:, 0, :], boxes[:, 1, :]
    inside = (cx >= ex1) & (cx <= ex2) & (cy >= ey1) & (cy <= ey2)
    conf = scores[:, class_idx, :]
    masked = torch.where(inside, conf, torch.full_like(conf, -30.0))
    if fallback_global_max:
        has = inside.any(dim=1, keepdim=True)
        masked = torch.where(has.expand_as(conf), masked, conf)
    T = float(temperature)
    return torch.logsumexp(T * masked, dim=1) / T


def make_disguise_loss(
    surrogate: DifferentiableSurrogate,
    target_box_xyxy: Sequence[float],
    true_class_idx: int,
    benign_class_idx: int,
    *,
    temperature: float = 20.0,
    target_pad_frac: float = 0.15,
    fallback_global_max: bool = True,
    true_weight: float = 1.0,
    benign_weight: float = 1.0,
) -> AttackLossFn:
    """Build an :data:`AttackLossFn` for hostile→benign flip.

    ``loss = true_weight · smoothmax(true) − benign_weight · smoothmax(benign)``.
    Minimizing raises the target benign class on the box while suppressing the
    true hostile class — keeping a detection under the benign label.
    """
    x1, y1, x2, y2 = [float(v) for v in target_box_xyxy]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = target_pad_frac * bw, target_pad_frac * bh
    ex1, ey1, ex2, ey2 = x1 - px, y1 - py, x2 + px, y2 + py

    def loss_fn(patched: torch.Tensor) -> torch.Tensor:
        boxes, scores = surrogate.forward_scores(patched)
        true_sm = _smoothmax_inbox(
            boxes, scores, true_class_idx,
            ex1=ex1, ey1=ey1, ex2=ex2, ey2=ey2,
            temperature=temperature, fallback_global_max=fallback_global_max,
        )
        benign_sm = _smoothmax_inbox(
            boxes, scores, benign_class_idx,
            ex1=ex1, ey1=ey1, ex2=ex2, ey2=ey2,
            temperature=temperature, fallback_global_max=fallback_global_max,
        )
        return (true_weight * true_sm - benign_weight * benign_sm).mean()

    return loss_fn


# ---------------------------------------------------------------------------
# Attacker
# ---------------------------------------------------------------------------


class DisguiseAttacker:
    """Craft a disguise patch for one target toward a benign class."""

    def __init__(
        self,
        config: DisguiseConfig,
        surrogate: DifferentiableSurrogate,
        patch_core: PatchCore,
        *,
        benign_class: str,
    ):
        if benign_class not in config.target_benign_classes:
            raise ValueError(
                f"benign_class {benign_class!r} not in "
                f"{config.target_benign_classes}"
            )
        self.config = config
        self.surrogate = surrogate
        self.core = patch_core
        self.benign_class = benign_class
        self.true_idx = surrogate.class_index(config.true_class)
        self.benign_idx = surrogate.class_index(benign_class)

    @classmethod
    def from_configs(
        cls,
        *,
        benign_class: str,
        disguise_config: str | Path | None = None,
        patch_config: str | Path | None = None,
        surrogate: DifferentiableSurrogate | None = None,
        device: str | torch.device | None = None,
    ) -> "DisguiseAttacker":
        cfg = load_disguise_config(disguise_config)
        dev = device if device is not None else cfg.device
        if surrogate is None:
            surrogate = DifferentiableSurrogate.from_family(
                cfg.surrogate_family, device=dev
            )
        core = PatchCore(load_patch_config(patch_config))
        return cls(cfg, surrogate, core, benign_class=benign_class)

    def craft(
        self,
        canvas: torch.Tensor,
        target_box_xyxy: Sequence[float],
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.nn.Parameter, list[dict[str, float]]]:
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        loss_fn = make_disguise_loss(
            self.surrogate,
            target_box_xyxy,
            self.true_idx,
            self.benign_idx,
            temperature=self.config.smoothmax_temperature,
            target_pad_frac=self.config.target_pad_frac,
            fallback_global_max=self.config.fallback_global_max,
            true_weight=self.config.true_weight,
            benign_weight=self.config.benign_weight,
        )
        canvas = canvas.to(self.surrogate.device)
        patch = self.core.init_patch(device=self.surrogate.device, rng=rng)
        history: list[dict[str, float]] = []
        for m in self.core.optimize(
            canvas,
            target_box_xyxy,
            patch,
            loss_fn,
            steps=self.config.steps,
            lr=self.config.lr,
            rng=rng,
        ):
            history.append(m)
        return patch, history


# ---------------------------------------------------------------------------
# TMSR scoring
# ---------------------------------------------------------------------------


@dataclass
class DisguiseInstanceResult:
    """Clean / attacked disguise status for one crafted patch."""

    image_id: int
    benign_class: str
    target_xyxy: tuple[float, float, float, float]
    clean_hostile: bool
    clean_hostile_score: float | None
    attacked: dict[str, dict[str, Any]]  # "axis:level" → status
    steps: int
    attack_loss_init: float | None
    attack_loss_final: float | None

    def tmsr_flags(self) -> dict[str, bool]:
        """Per (axis:level) TMSR success (clean hostile + attacked benign match)."""
        if not self.clean_hostile:
            return {k: False for k in self.attacked}
        return {k: bool(v["tmsr"]) for k, v in self.attacked.items()}


def score_instance_tmsr(
    predict_fn: PredictFn,
    clean_canvas_rgb: np.ndarray,
    attacked_canvas_rgb: np.ndarray,
    target_xyxy: Sequence[float],
    *,
    true_class: str,
    benign_class: str,
    conf: float,
    iou_match: float,
    marine_eot=None,
    severity_axes: Sequence[str] = (),
    severity_levels: Sequence[str] = (),
    image_id: int = -1,
    steps: int = 0,
    attack_loss_init: float | None = None,
    attack_loss_final: float | None = None,
) -> DisguiseInstanceResult:
    """Score clean hostile vs. attacked benign across the marine-EOT sweep.

    Patch-attributable bookkeeping: ``clean_benign_under_transform`` is True when
    the *un-patched* canvas under the same transform already shows the target
    benign class — those cases are excluded from the patch-attributable denom.
    """
    tgt = tuple(float(v) for v in target_xyxy)
    clean_hit = find_target_detection(
        predict_fn(clean_canvas_rgb), tgt, true_class, conf=conf, iou_match=iou_match
    )
    attacked: dict[str, dict[str, Any]] = {}
    for axis in severity_axes:
        for level in severity_levels:
            key = f"{axis}:{level}"
            if marine_eot is not None:
                atk_canvas = marine_eot.apply_axis(attacked_canvas_rgb, axis, level)
                cln_canvas = marine_eot.apply_axis(clean_canvas_rgb, axis, level)
            else:
                atk_canvas = attacked_canvas_rgb
                cln_canvas = clean_canvas_rgb
            atk_hit = find_target_detection(
                predict_fn(atk_canvas), tgt, benign_class, conf=conf, iou_match=iou_match
            )
            # Still-detected under true class (failed disguise / residual usv).
            still_true = find_target_detection(
                predict_fn(atk_canvas), tgt, true_class, conf=conf, iou_match=iou_match
            )
            clean_benign_t = find_target_detection(
                predict_fn(cln_canvas), tgt, benign_class, conf=conf, iou_match=iou_match
            )
            attacked[key] = {
                "tmsr": atk_hit is not None,
                "benign_score": float(atk_hit.score) if atk_hit is not None else None,
                "still_true": still_true is not None,
                "clean_benign_under_transform": clean_benign_t is not None,
            }
    return DisguiseInstanceResult(
        image_id=int(image_id),
        benign_class=benign_class,
        target_xyxy=tgt,
        clean_hostile=clean_hit is not None,
        clean_hostile_score=float(clean_hit.score) if clean_hit is not None else None,
        attacked=attacked,
        steps=int(steps),
        attack_loss_init=attack_loss_init,
        attack_loss_final=attack_loss_final,
    )


def aggregate_tmsr(
    results: Sequence[DisguiseInstanceResult],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per (axis:level) TMSR over clean-hostile (eligible) instances.

    * ``tmsr`` — fraction of eligible instances with a successful benign flip.
    * ``tmsr_patch_attributable`` — among instances where the same transform did
      **not** already show the target benign class on the clean canvas.
    """
    eligible = [r for r in results if r.clean_hostile]
    n = len(eligible)
    out: dict[str, dict[str, float | int]] = {}
    keys = eligible[0].attacked.keys() if eligible else []
    for key in keys:
        succ = sum(1 for r in eligible if r.attacked[key]["tmsr"])
        pa_denom = [
            r for r in eligible
            if not r.attacked[key].get("clean_benign_under_transform", False)
        ]
        pa_succ = sum(1 for r in pa_denom if r.attacked[key]["tmsr"])
        out[key] = {
            "tmsr": (succ / n) if n else 0.0,
            "n_success": succ,
            "n_eligible": n,
            "tmsr_patch_attributable": (pa_succ / len(pa_denom)) if pa_denom else 0.0,
            "n_patch_attributable_denom": len(pa_denom),
        }
    return out


__all__ = [
    "DisguiseAttacker",
    "DisguiseConfig",
    "DisguiseInstanceResult",
    "aggregate_tmsr",
    "load_disguise_config",
    "make_disguise_loss",
    "make_predict_fn",
    "score_instance_tmsr",
    "to_torch_device",
    "to_ultralytics_device",
    "DifferentiableSurrogate",
]
