"""Evasion attack (ESR) — suppress the true-class ``usv`` detection.

Layers the evasion objective on the shared, objective-agnostic patch core
(:mod:`counterusv.attacks.patch`) and the marine-EOT expectation
(:mod:`counterusv.attacks.marine_eot`). The patch is optimized white-box against
a **surrogate** detector so that every prediction overlapping the target box has
its target-class confidence driven below the detection threshold (full
non-detection). Success is scored as **ESR** (``docs/METRICS.md``): a contact
detected on the clean canvas that is no longer detected on the attacked canvas,
reported across the marine-EOT severity ladder.

Config: ``configs/attacks/evasion.yaml`` (objective + eval thresholds), on top of
``configs/attacks/patch.yaml`` (geometry / printability / EOT).

The differentiable forward uses the surrogate's raw detection-head output
(``(B, 4+nc, A)``: 4 box + ``nc`` post-sigmoid class rows); the sequestered
held-out transfer family is refused for crafting via
:meth:`DetectorBaseline.assert_attack_crafting_allowed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import yaml

from counterusv.attacks.patch import AttackLossFn, PatchCore, load_patch_config
from counterusv.models.detector import Detection, DetectorBaseline, resolve_device

DEFAULT_CONFIG = Path("configs/attacks/evasion.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def to_torch_device(requested: str | torch.device = "auto") -> torch.device:
    """Map ``auto`` / Ultralytics ``\"0\"`` / ``cuda`` / ``cpu`` / ``mps`` → torch device."""
    if isinstance(requested, torch.device):
        return requested
    d = resolve_device(str(requested))
    if d.isdigit():
        return torch.device(f"cuda:{d}")
    if d == "cuda":
        return torch.device("cuda")
    return torch.device(d)


def to_ultralytics_device(requested: str | torch.device = "auto") -> str:
    """Map a torch / auto device string to Ultralytics predict device (``\"0\"`` / ``cpu`` / ``mps``)."""
    if isinstance(requested, torch.device):
        if requested.type == "cuda":
            idx = requested.index if requested.index is not None else 0
            return str(idx)
        return requested.type  # cpu / mps
    d = resolve_device(str(requested))
    if d.startswith("cuda"):
        return d.split(":")[-1] if ":" in d else "0"
    return d


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvasionConfig:
    version: int
    frozen: str
    surrogate: dict[str, Any]
    objective: dict[str, Any]
    optimization: dict[str, Any]
    evaluation: dict[str, Any]
    seed: int
    path: Path | None = None

    @property
    def surrogate_family(self) -> str:
        return str(self.surrogate.get("family", "yolo11s"))

    @property
    def target_class(self) -> str:
        return str(self.surrogate.get("target_class", "usv"))

    @property
    def device(self) -> str:
        return str(self.surrogate.get("device", "cpu"))

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


def load_evasion_config(path: str | Path | None = None) -> EvasionConfig:
    p = Path(path) if path is not None else _REPO_ROOT / DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path(DEFAULT_CONFIG)
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"evasion config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return EvasionConfig(
        version=int(raw["version"]),
        frozen=str(raw["frozen"]),
        surrogate=dict(raw.get("surrogate") or {}),
        objective=dict(raw.get("objective") or {}),
        optimization=dict(raw.get("optimization") or {}),
        evaluation=dict(raw.get("evaluation") or {}),
        seed=int(raw.get("seed", 1337)),
        path=p.resolve(),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``(x1,y1,x2,y2)`` boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def find_target_detection(
    dets: Sequence[Detection],
    target_xyxy: Sequence[float],
    class_name: str,
    *,
    conf: float,
    iou_match: float,
) -> Detection | None:
    """Highest-scoring ``class_name`` detection matching the target box, or None.

    A match requires ``score >= conf`` and ``IoU(pred, target) >= iou_match``.
    """
    best: Detection | None = None
    for d in dets:
        if d.class_name != class_name or d.score < conf:
            continue
        if iou_xyxy(d.box_xyxy, target_xyxy) < iou_match:
            continue
        if best is None or d.score > best.score:
            best = d
    return best


# ---------------------------------------------------------------------------
# Differentiable surrogate (white-box hook)
# ---------------------------------------------------------------------------


class DifferentiableSurrogate:
    """Raw-head wrapper over a :class:`DetectorBaseline` for gradient crafting.

    ``forward_scores`` returns ``(boxes_xywh, class_scores)`` where boxes are
    center ``xywh`` in **input-canvas pixels** (the 640 letterbox) and
    ``class_scores`` are post-sigmoid ``(B, nc, A)`` confidences — differentiable
    w.r.t. the input (and thus an upstream patch composite).
    """

    def __init__(self, baseline: DetectorBaseline, *, device: str | torch.device = "auto"):
        baseline.assert_attack_crafting_allowed()
        self.baseline = baseline
        self.device = to_torch_device(device)
        # Keep hard-detector predict on the same accelerator.
        baseline.device = to_ultralytics_device(self.device)
        module = baseline.model.model  # ultralytics DetectionModel (nn.Module)
        self.module = module.to(self.device).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.names: dict[int, str] = dict(baseline.yolo_names)
        self.name_to_idx: dict[str, int] = {v: k for k, v in self.names.items()}

    @classmethod
    def from_family(
        cls,
        family: str,
        *,
        device: str | torch.device = "auto",
        **baseline_kwargs: Any,
    ) -> "DifferentiableSurrogate":
        ultra = to_ultralytics_device(device)
        baseline_kwargs = {**baseline_kwargs, "device": ultra}
        try:
            baseline = DetectorBaseline.from_freeze(family, **baseline_kwargs)
        except FileNotFoundError:
            baseline = DetectorBaseline.from_roster(family, **baseline_kwargs)
        return cls(baseline, device=device)

    def class_index(self, name: str) -> int:
        if name not in self.name_to_idx:
            raise KeyError(
                f"class {name!r} not in surrogate head {sorted(self.name_to_idx)}"
            )
        return self.name_to_idx[name]

    def forward_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        out = self.module(x)
        preds = out[0] if isinstance(out, (list, tuple)) else out
        boxes = preds[:, :4, :]
        scores = preds[:, 4:, :]
        return boxes, scores


# ---------------------------------------------------------------------------
# Evasion objective
# ---------------------------------------------------------------------------


def make_evasion_loss(
    surrogate: DifferentiableSurrogate,
    target_box_xyxy: Sequence[float],
    class_idx: int,
    *,
    temperature: float = 20.0,
    target_pad_frac: float = 0.15,
    fallback_global_max: bool = True,
) -> AttackLossFn:
    """Build an :data:`AttackLossFn` that suppresses ``class_idx`` on the target.

    Loss = smooth-max of target-class confidence over predictions whose center
    lands in the (dilated) target box. Minimizing it pushes every overlapping
    prediction below the detection threshold (full evasion). Lower is better.
    """
    x1, y1, x2, y2 = [float(v) for v in target_box_xyxy]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = target_pad_frac * bw, target_pad_frac * bh
    ex1, ey1, ex2, ey2 = x1 - px, y1 - py, x2 + px, y2 + py
    T = float(temperature)

    def loss_fn(patched: torch.Tensor) -> torch.Tensor:
        boxes, scores = surrogate.forward_scores(patched)
        cx, cy = boxes[:, 0, :], boxes[:, 1, :]  # (B,A)
        inside = (cx >= ex1) & (cx <= ex2) & (cy >= ey1) & (cy <= ey2)
        conf = scores[:, class_idx, :]  # (B,A) post-sigmoid
        masked = torch.where(inside, conf, torch.full_like(conf, -30.0))
        if fallback_global_max:
            has = inside.any(dim=1, keepdim=True)  # (B,1)
            masked = torch.where(has.expand_as(conf), masked, conf)
        # Smooth-max over anchors: (1/T)·logsumexp(T·conf).
        return (torch.logsumexp(T * masked, dim=1) / T).mean()

    return loss_fn


@torch.no_grad()
def target_confidence(
    surrogate: DifferentiableSurrogate,
    canvas: torch.Tensor,
    target_box_xyxy: Sequence[float],
    class_idx: int,
    *,
    target_pad_frac: float = 0.15,
) -> float:
    """Max target-class confidence over anchors inside the (dilated) box."""
    x1, y1, x2, y2 = [float(v) for v in target_box_xyxy]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = target_pad_frac * bw, target_pad_frac * bh
    boxes, scores = surrogate.forward_scores(canvas)
    cx, cy = boxes[:, 0, :], boxes[:, 1, :]
    inside = (cx >= x1 - px) & (cx <= x2 + px) & (cy >= y1 - py) & (cy <= y2 + py)
    conf = scores[:, class_idx, :]
    masked = conf[inside]
    if masked.numel() == 0:
        return float(conf.max())
    return float(masked.max())


# ---------------------------------------------------------------------------
# Attacker
# ---------------------------------------------------------------------------


class EvasionAttacker:
    """Craft an evasion patch for one target on the letterboxed canvas."""

    def __init__(
        self,
        config: EvasionConfig,
        surrogate: DifferentiableSurrogate,
        patch_core: PatchCore,
    ):
        self.config = config
        self.surrogate = surrogate
        self.core = patch_core
        self.class_idx = surrogate.class_index(config.target_class)

    @classmethod
    def from_configs(
        cls,
        *,
        evasion_config: str | Path | None = None,
        patch_config: str | Path | None = None,
        surrogate: DifferentiableSurrogate | None = None,
        device: str | torch.device | None = None,
    ) -> "EvasionAttacker":
        cfg = load_evasion_config(evasion_config)
        dev = device if device is not None else cfg.device
        if surrogate is None:
            surrogate = DifferentiableSurrogate.from_family(
                cfg.surrogate_family, device=dev
            )
        core = PatchCore(load_patch_config(patch_config))
        return cls(cfg, surrogate, core)

    def craft(
        self,
        canvas: torch.Tensor,
        target_box_xyxy: Sequence[float],
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.nn.Parameter, list[dict[str, float]]]:
        """Optimize a patch to suppress the target; return ``(patch, history)``."""
        rng = rng if rng is not None else np.random.default_rng(self.config.seed)
        loss_fn = make_evasion_loss(
            self.surrogate,
            target_box_xyxy,
            self.class_idx,
            temperature=self.config.smoothmax_temperature,
            target_pad_frac=self.config.target_pad_frac,
            fallback_global_max=self.config.fallback_global_max,
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
# Hard-detector ESR scoring
# ---------------------------------------------------------------------------

PredictFn = Callable[[np.ndarray], list[Detection]]
"""Maps a HxWx3 **RGB uint8** canvas → hard detections (post-NMS)."""


def make_predict_fn(
    baseline: DetectorBaseline,
    *,
    conf: float,
    iou: float,
    imgsz: int = 640,
) -> PredictFn:
    """Hard-detector predict on an in-memory canvas (RGB→BGR for Ultralytics)."""

    def predict(canvas_rgb: np.ndarray) -> list[Detection]:
        # Ultralytics treats bare numpy arrays as BGR (cv2 convention) and flips
        # to RGB internally; feed BGR so the model sees the intended RGB.
        bgr = np.ascontiguousarray(canvas_rgb[..., ::-1])
        return baseline.predict_one(bgr, conf=conf, iou=iou, imgsz=imgsz)

    return predict


@dataclass
class EvasionInstanceResult:
    """Clean / attacked target-detection status for one crafted patch."""

    image_id: int
    target_xyxy: tuple[float, float, float, float]
    clean_detected: bool
    clean_score: float | None
    attacked: dict[str, dict[str, bool | float | None]]  # "axis:level" → status
    steps: int
    attack_loss_init: float | None
    attack_loss_final: float | None

    def esr_flags(self) -> dict[str, bool]:
        """Per (axis:level) evasion success (clean-detected and now suppressed)."""
        if not self.clean_detected:
            return {k: False for k in self.attacked}
        return {
            k: (not bool(v["detected"])) for k, v in self.attacked.items()
        }


def score_instance_esr(
    predict_fn: PredictFn,
    clean_canvas_rgb: np.ndarray,
    attacked_canvas_rgb: np.ndarray,
    target_xyxy: Sequence[float],
    *,
    class_name: str,
    conf: float,
    iou_match: float,
    marine_eot=None,
    severity_axes: Sequence[str] = (),
    severity_levels: Sequence[str] = (),
    image_id: int = -1,
    steps: int = 0,
    attack_loss_init: float | None = None,
    attack_loss_final: float | None = None,
) -> EvasionInstanceResult:
    """Score clean vs. attacked target detection across the marine-EOT sweep.

    ``attacked_canvas_rgb`` is the composited (patched) canvas; each (axis,level)
    applies the corresponding single-axis severity to it before the hard detector
    runs. L0 (identity) equals the base ESR.
    """
    tgt = tuple(float(v) for v in target_xyxy)
    clean_hit = find_target_detection(
        predict_fn(clean_canvas_rgb), tgt, class_name, conf=conf, iou_match=iou_match
    )
    attacked: dict[str, dict[str, bool | float | None]] = {}
    for axis in severity_axes:
        for level in severity_levels:
            key = f"{axis}:{level}"
            if marine_eot is not None:
                atk_canvas = marine_eot.apply_axis(attacked_canvas_rgb, axis, level)
                cln_canvas = marine_eot.apply_axis(clean_canvas_rgb, axis, level)
            else:
                atk_canvas = attacked_canvas_rgb
                cln_canvas = clean_canvas_rgb
            hit = find_target_detection(
                predict_fn(atk_canvas), tgt, class_name, conf=conf, iou_match=iou_match
            )
            # Whether the SAME transform suppresses the clean (un-patched) target:
            # lets the full run isolate the patch's contribution from transform-
            # induced non-detection (transform-adjusted ESR).
            clean_hit_t = find_target_detection(
                predict_fn(cln_canvas), tgt, class_name, conf=conf, iou_match=iou_match
            )
            attacked[key] = {
                "detected": hit is not None,
                "score": float(hit.score) if hit is not None else None,
                "clean_detected_under_transform": clean_hit_t is not None,
            }
    return EvasionInstanceResult(
        image_id=int(image_id),
        target_xyxy=tgt,
        clean_detected=clean_hit is not None,
        clean_score=float(clean_hit.score) if clean_hit is not None else None,
        attacked=attacked,
        steps=int(steps),
        attack_loss_init=attack_loss_init,
        attack_loss_final=attack_loss_final,
    )


def aggregate_esr(
    results: Sequence[EvasionInstanceResult],
) -> dict[str, dict[str, float | int]]:
    """Aggregate per (axis:level) ESR over clean-detected (attackable) instances.

    * ``esr`` — fraction of attackable instances suppressed when attacked
      (matches ``docs/METRICS.md``; at high severity this includes suppression
      the transform would cause on its own).
    * ``esr_patch_attributable`` — fraction suppressed among instances the same
      transform did **not** suppress un-patched, isolating the patch's effect.
    """
    attackable = [r for r in results if r.clean_detected]
    n = len(attackable)
    out: dict[str, dict[str, float | int]] = {}
    keys = attackable[0].attacked.keys() if attackable else []
    for key in keys:
        succ = sum(1 for r in attackable if not r.attacked[key]["detected"])
        # Patch-attributable: transform alone left the clean target detected.
        pa_denom = [
            r for r in attackable
            if r.attacked[key].get("clean_detected_under_transform", True)
        ]
        pa_succ = sum(1 for r in pa_denom if not r.attacked[key]["detected"])
        out[key] = {
            "esr": (succ / n) if n else 0.0,
            "n_success": succ,
            "n_attackable": n,
            "esr_patch_attributable": (pa_succ / len(pa_denom)) if pa_denom else 0.0,
            "n_patch_attributable_denom": len(pa_denom),
        }
    return out
