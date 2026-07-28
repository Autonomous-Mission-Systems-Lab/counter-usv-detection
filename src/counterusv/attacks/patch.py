"""Physically-realizable adversarial patch core (objective-agnostic).

Shared by evasion (ESR) and disguise (TMSR): bounded hull/superstructure
placement, printability regularizers (TV + NPS), and marine-EOT expectation
inside optimization. Attack losses are injected by the caller.

Config: ``configs/attacks/patch.yaml``. Eligibility follows
``data/HARMONIZATION.md`` (patch floor ≥32px; letterbox-only apply/eval).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from counterusv.attacks.marine_eot import MarineEOT

DEFAULT_CONFIG = Path("configs/attacks/patch.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchConfig:
    version: int
    frozen: str
    eligibility: dict[str, Any]
    geometry: dict[str, Any]
    regularization: dict[str, Any]
    optimization: dict[str, Any]
    seed: int
    path: Path | None = None

    @property
    def patch_min_side(self) -> float:
        return float(self.eligibility.get("patch_min_pixels_on_target", 32))

    @property
    def canonical_side(self) -> int:
        return int(self.geometry.get("canonical_side", 64))

    @property
    def tv_weight(self) -> float:
        return float(self.regularization.get("tv_weight", 0.05))

    @property
    def nps_weight(self) -> float:
        return float(self.regularization.get("nps_weight", 0.01))

    @property
    def eot_samples(self) -> int:
        return int(self.optimization.get("eot_samples", 4))

    @property
    def eot_enabled(self) -> bool:
        return bool(self.optimization.get("eot_enabled", True))

    @property
    def lr(self) -> float:
        return float(self.optimization.get("lr", 0.05))

    @property
    def steps(self) -> int:
        return int(self.optimization.get("steps", 200))


def load_patch_config(path: str | Path | None = None) -> PatchConfig:
    p = Path(path) if path is not None else _REPO_ROOT / DEFAULT_CONFIG
    if not p.is_file():
        alt = Path(path) if path is not None else Path(DEFAULT_CONFIG)
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"patch config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return PatchConfig(
        version=int(raw["version"]),
        frozen=str(raw["frozen"]),
        eligibility=dict(raw.get("eligibility") or {}),
        geometry=dict(raw.get("geometry") or {}),
        regularization=dict(raw.get("regularization") or {}),
        optimization=dict(raw.get("optimization") or {}),
        seed=int(raw.get("seed", 1337)),
        path=p.resolve(),
    )


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def is_patch_eligible(
    box_xywh: Sequence[float],
    *,
    min_side: float = 32.0,
) -> bool:
    """True iff the box short side meets the patch floor (HARMONIZATION)."""
    _, _, w, h = [float(v) for v in box_xywh]
    return min(w, h) >= float(min_side)


def filter_patch_eligible(
    boxes_xywh: np.ndarray | Sequence[Sequence[float]],
    *,
    min_side: float = 32.0,
) -> np.ndarray:
    """Return ``(N,4)`` xywh boxes that pass the patch floor (possibly empty)."""
    if boxes_xywh is None or len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    keep = np.minimum(b[:, 2], b[:, 3]) >= float(min_side)
    return b[keep]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Pixel placement of a square patch on one image."""

    x0: int
    y0: int
    side: int
    size_frac: float


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def place_on_bbox(
    box_xyxy: Sequence[float],
    image_hw: tuple[int, int],
    cfg: PatchConfig,
    rng: np.random.Generator | None = None,
) -> Placement:
    """Compute hull/superstructure placement for one target box.

    ``box_xyxy`` and ``image_hw`` are in the **same** frame (typically the
    letterboxed input canvas). Placement is deterministic when ``rng`` is None
    (eval); pass an RNG for EOT jitter during optimization.
    """
    g = cfg.geometry
    h_img, w_img = image_hw
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    short = min(bw, bh)

    size_frac = float(g.get("size_frac", 0.35))
    if rng is not None:
        jit = float(g.get("size_frac_jitter", 0.0))
        if jit > 0:
            size_frac = float(size_frac + rng.uniform(-jit, jit))
    size_frac = float(
        np.clip(
            size_frac,
            float(g.get("size_frac_min", 0.2)),
            float(g.get("size_frac_max", 0.45)),
        )
    )
    side = int(round(size_frac * short))
    side = int(
        np.clip(
            side,
            int(g.get("side_px_min", 16)),
            int(g.get("side_px_max", 96)),
        )
    )
    side = max(1, min(side, h_img, w_img))

    ax = float(g.get("anchor_x_frac", 0.5))
    ay = float(g.get("anchor_y_frac", 0.62))
    if rng is not None:
        pj = float(g.get("place_jitter_frac", 0.0))
        if pj > 0:
            ax = float(np.clip(ax + rng.uniform(-pj, pj), 0.05, 0.95))
            ay = float(np.clip(ay + rng.uniform(-pj, pj), 0.05, 0.95))

    cx = x1 + ax * bw
    cy = y1 + ay * bh
    x0 = _clamp_int(cx - side / 2.0, 0, max(w_img - side, 0))
    y0 = _clamp_int(cy - side / 2.0, 0, max(h_img - side, 0))
    return Placement(x0=x0, y0=y0, side=side, size_frac=size_frac)


def xywh_to_xyxy(box_xywh: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return (x, y, x + w, y + h)


# ---------------------------------------------------------------------------
# Regularizers
# ---------------------------------------------------------------------------


def total_variation(patch: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV on a ``(3,H,W)`` or ``(B,3,H,W)`` patch in ``[0,1]``."""
    if patch.ndim == 3:
        patch = patch.unsqueeze(0)
    dh = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().mean()
    dw = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().mean()
    return dh + dw


def printable_color_tensor(
    colors_rgb: Sequence[Sequence[float]],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``(K,3)`` printable colors in ``[0,1]`` from 0–255 RGB rows."""
    arr = np.asarray(colors_rgb, dtype=np.float32).reshape(-1, 3) / 255.0
    return torch.as_tensor(arr, device=device, dtype=dtype)


def non_printability_score(
    patch: torch.Tensor,
    printable: torch.Tensor,
) -> torch.Tensor:
    """Mean per-pixel distance² to the nearest printable color.

    ``patch``: ``(3,H,W)`` or ``(B,3,H,W)``; ``printable``: ``(K,3)`` in ``[0,1]``.
    """
    if patch.ndim == 3:
        patch = patch.unsqueeze(0)
    # (B,H,W,3)
    pix = patch.permute(0, 2, 3, 1)
    # (B,H,W,1,3) - (1,1,1,K,3) → (B,H,W,K)
    d2 = ((pix.unsqueeze(-2) - printable.view(1, 1, 1, -1, 3)) ** 2).sum(-1)
    return d2.min(dim=-1).values.mean()


# ---------------------------------------------------------------------------
# Differentiable composite
# ---------------------------------------------------------------------------


def _feather_mask(side: int, feather_frac: float, device, dtype) -> torch.Tensor:
    """``(1,1,side,side)`` alpha mask with a soft border."""
    if feather_frac <= 0 or side < 3:
        return torch.ones(1, 1, side, side, device=device, dtype=dtype)
    feather = max(int(round(feather_frac * side)), 1)
    yy = torch.arange(side, device=device, dtype=dtype)
    xx = torch.arange(side, device=device, dtype=dtype)
    # Distance to nearest edge.
    dist = torch.minimum(
        torch.minimum(yy, side - 1 - yy).unsqueeze(1),
        torch.minimum(xx, side - 1 - xx).unsqueeze(0),
    )
    alpha = (dist / float(feather)).clamp(0.0, 1.0)
    return alpha.view(1, 1, side, side)


def apply_patch_torch(
    image: torch.Tensor,
    patch: torch.Tensor,
    placement: Placement,
    *,
    feather_frac: float = 0.08,
) -> torch.Tensor:
    """Paste ``patch`` onto ``image`` at ``placement`` (differentiable in patch).

    Parameters
    ----------
    image
        ``(3,H,W)`` or ``(B,3,H,W)`` float in ``[0,1]``.
    patch
        ``(3,Ph,Pw)`` float in ``[0,1]`` (canonical resolution OK — resized).
    placement
        Pixel location / side on the image canvas.
    """
    squeeze = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze = True
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise ValueError(f"patch must be (3,H,W), got {tuple(patch.shape)}")
    if patch.device != image.device:
        patch = patch.to(image.device)

    b, _, h, w = image.shape
    side = int(placement.side)
    x0, y0 = int(placement.x0), int(placement.y0)
    # Clip paste window to image bounds.
    x1 = min(x0 + side, w)
    y1 = min(y0 + side, h)
    pw, ph = x1 - x0, y1 - y0
    if pw <= 0 or ph <= 0:
        return image.squeeze(0) if squeeze else image

    resized = F.interpolate(
        patch.unsqueeze(0),
        size=(side, side),
        mode="bilinear",
        align_corners=False,
    )  # (1,3,side,side)
    if (pw, ph) != (side, side):
        resized = resized[:, :, :ph, :pw]
        mask = _feather_mask(side, feather_frac, image.device, image.dtype)[:, :, :ph, :pw]
    else:
        mask = _feather_mask(side, feather_frac, image.device, image.dtype)

    out = image.clone()
    region = out[:, :, y0:y1, x0:x1]
    blended = region * (1.0 - mask) + resized.expand(b, -1, -1, -1) * mask
    out[:, :, y0:y1, x0:x1] = blended
    return out.squeeze(0) if squeeze else out


def apply_patch_numpy(
    image: np.ndarray,
    patch: np.ndarray,
    placement: Placement,
    *,
    feather_frac: float = 0.08,
) -> np.ndarray:
    """NumPy/uint8 or float01 wrapper around ``apply_patch_torch``."""
    was_uint8 = image.dtype == np.uint8
    img_f = image.astype(np.float32) / 255.0 if was_uint8 else image.astype(np.float32)
    if img_f.max() > 1.5:
        img_f = img_f / 255.0
    p_f = patch.astype(np.float32)
    if p_f.max() > 1.5:
        p_f = p_f / 255.0
    t_img = torch.from_numpy(img_f).permute(2, 0, 1)
    t_pat = torch.from_numpy(np.clip(p_f, 0.0, 1.0)).permute(2, 0, 1)
    out = apply_patch_torch(t_img, t_pat, placement, feather_frac=feather_frac)
    arr = out.permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    if was_uint8:
        return (arr * 255.0 + 0.5).astype(np.uint8)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------


AttackLossFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps a patched ``(B,3,H,W)`` batch (after optional EOT) → scalar attack loss."""


class PatchCore:
    """Objective-agnostic physically-realizable patch optimizer.

    Typical use (evasion / disguise supply ``attack_loss_fn``)::

        core = PatchCore.from_config()
        patch = core.init_patch(device=\"cpu\")
        for step in core.optimize(images, boxes_xyxy, patch, attack_loss_fn):
            ...
    """

    def __init__(
        self,
        config: PatchConfig,
        *,
        marine_eot: MarineEOT | None = None,
    ):
        self.config = config
        eot_path = config.optimization.get("marine_eot_config")
        if marine_eot is not None:
            self.eot = marine_eot
        elif config.eot_enabled:
            self.eot = MarineEOT.from_config(
                _REPO_ROOT / eot_path if eot_path else None
            )
        else:
            self.eot = None
        colors = config.regularization.get("printable_colors_rgb") or []
        self._printable_rgb = list(colors)
        self._rng = np.random.default_rng(config.seed)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "PatchCore":
        return cls(load_patch_config(path))

    def init_patch(
        self,
        *,
        device: str | torch.device = "cpu",
        rng: np.random.Generator | None = None,
    ) -> torch.nn.Parameter:
        """Create a learnable ``(3,S,S)`` patch Parameter in ``[0,1]``."""
        rng = rng if rng is not None else self._rng
        opt = self.config.optimization
        lo = float(opt.get("init_lo", 0.25))
        hi = float(opt.get("init_hi", 0.75))
        s = self.config.canonical_side
        arr = rng.uniform(lo, hi, size=(3, s, s)).astype(np.float32)
        return torch.nn.Parameter(torch.as_tensor(arr, device=device))

    def printable_tensor(self, patch: torch.Tensor) -> torch.Tensor:
        return printable_color_tensor(
            self._printable_rgb, device=patch.device, dtype=patch.dtype
        )

    def regularizer_loss(self, patch: torch.Tensor) -> dict[str, torch.Tensor]:
        tv = total_variation(patch)
        nps = non_printability_score(patch, self.printable_tensor(patch))
        total = self.config.tv_weight * tv + self.config.nps_weight * nps
        return {"tv": tv, "nps": nps, "reg": total}

    def composite(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        box_xyxy: Sequence[float],
        *,
        rng: np.random.Generator | None = None,
        apply_eot: bool = False,
    ) -> tuple[torch.Tensor, Placement]:
        """Place ``patch`` on one image; optionally apply one marine-EOT draw."""
        if image.ndim == 3:
            h, w = int(image.shape[1]), int(image.shape[2])
        else:
            h, w = int(image.shape[2]), int(image.shape[3])
        placement = place_on_bbox(
            box_xyxy, (h, w), self.config, rng=rng
        )
        feather = float(self.config.geometry.get("feather_frac", 0.08))
        out = apply_patch_torch(image, patch, placement, feather_frac=feather)
        if apply_eot and self.eot is not None:
            if out.ndim == 3:
                out = self.eot.apply_torch(out.unsqueeze(0), rng=rng).squeeze(0)
            else:
                out = self.eot.apply_torch(out, rng=rng)
        return out, placement

    def eot_expectation(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        box_xyxy: Sequence[float],
        attack_loss_fn: AttackLossFn,
        *,
        n_samples: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Monte-Carlo EOT: mean attack loss over composite(+transform) draws.

        Gradients flow through each draw into ``patch``. Regularizers are NOT
        included here — add ``regularizer_loss`` in the outer step.
        """
        rng = rng if rng is not None else self._rng
        n = int(n_samples if n_samples is not None else self.config.eot_samples)
        n = max(n, 1)
        losses = []
        for _ in range(n):
            patched, _ = self.composite(
                image,
                patch,
                box_xyxy,
                rng=rng,
                apply_eot=bool(self.eot is not None and self.config.eot_enabled),
            )
            if patched.ndim == 3:
                patched = patched.unsqueeze(0)
            losses.append(attack_loss_fn(patched))
        attack = torch.stack(losses).mean()
        return attack, {"eot_samples": float(n), "attack": float(attack.detach())}

    def step(
        self,
        image: torch.Tensor,
        box_xyxy: Sequence[float],
        patch: torch.nn.Parameter,
        attack_loss_fn: AttackLossFn,
        optimizer: torch.optim.Optimizer,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, float]:
        """One optimize step: EOT attack loss + TV/NPS, then clamp patch."""
        optimizer.zero_grad(set_to_none=True)
        attack, extras = self.eot_expectation(
            image, patch, box_xyxy, attack_loss_fn, rng=rng
        )
        regs = self.regularizer_loss(patch)
        loss = attack + regs["reg"]
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            lo = float(self.config.optimization.get("clamp_min", 0.0))
            hi = float(self.config.optimization.get("clamp_max", 1.0))
            patch.clamp_(lo, hi)
        return {
            "loss": float(loss.detach()),
            "attack": extras["attack"],
            "tv": float(regs["tv"].detach()),
            "nps": float(regs["nps"].detach()),
            "eot_samples": extras["eot_samples"],
        }

    def optimize(
        self,
        image: torch.Tensor,
        box_xyxy: Sequence[float],
        patch: torch.nn.Parameter,
        attack_loss_fn: AttackLossFn,
        *,
        steps: int | None = None,
        lr: float | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Yield per-step metric dicts while optimizing ``patch`` in place."""
        lr = float(lr if lr is not None else self.config.lr)
        n_steps = int(steps if steps is not None else self.config.steps)
        opt_name = str(self.config.optimization.get("optimizer", "adam")).lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD([patch], lr=lr)
        else:
            optimizer = torch.optim.Adam([patch], lr=lr)
        rng = rng if rng is not None else self._rng
        for i in range(n_steps):
            metrics = self.step(
                image, box_xyxy, patch, attack_loss_fn, optimizer, rng=rng
            )
            metrics["step"] = float(i)
            yield metrics

    # --- NumPy helpers for QA / smoke -------------------------------------

    def patch_to_uint8(self, patch: torch.Tensor) -> np.ndarray:
        arr = patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (arr * 255.0 + 0.5).astype(np.uint8)
