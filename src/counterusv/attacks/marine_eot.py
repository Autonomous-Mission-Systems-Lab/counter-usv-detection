"""Marine expectation-over-transformation (marine-EOT) library.

Implements the marine transform distribution used both as a Monte-Carlo
expectation inside patch optimization and as an eval-time per-axis severity
sweep. Parameter ranges and discrete severity levels are frozen in
``configs/attacks/marine_eot.yaml`` so both consumers share one definition.

Axes (see ``docs/THREAT_MODEL.md`` / ``docs/METRICS.md``):
  scale, rotation, motion_blur, glare, spray, grazing_angle, sea_state.

Two backends share the same sampled parameters:
  * NumPy / OpenCV — eval sweeps, sample-grid figures, CPU tooling.
  * PyTorch (``grid_sample`` / conv) — differentiable path for EOT expectation
    inside patch optimization (gradients flow to the patched image).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

AXES: tuple[str, ...] = (
    "scale",
    "rotation",
    "motion_blur",
    "glare",
    "spray",
    "grazing_angle",
    "sea_state",
)
SEVERITY_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")

DEFAULT_CONFIG = Path("configs/attacks/marine_eot.yaml")
_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Config / parameter containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarineEOTConfig:
    """Frozen marine-EOT recipe loaded from YAML."""

    version: int
    frozen: str
    severity_levels: tuple[str, ...]
    axes: dict[str, Any]
    expectation: dict[str, Any]
    seed: int
    path: Path | None = None

    @property
    def border_value(self) -> int:
        return int(self.expectation.get("border_value", 114))

    @property
    def n_samples(self) -> int:
        return int(self.expectation.get("n_samples", 10))


@dataclass
class TransformParams:
    """One joint draw (or single-axis severity) of marine-EOT parameters."""

    scale: float = 1.0
    rotation_deg: float = 0.0
    blur_length: int = 1
    blur_angle_deg: float = 0.0
    glare_intensity: float = 0.0
    glare_radius_frac: float = 0.25
    glare_cx_frac: float = 0.5
    glare_cy_frac: float = 0.4
    spray_coverage: float = 0.0
    spray_n_blobs: int = 0
    spray_seed: int = 0
    grazing_strength: float = 0.0
    sea_mix: float = 0.0
    sea_band_top_frac: float = 0.45
    sea_ripple_amp: float = 0.0
    sea_wave_freq: float = 1.0
    sea_seed: int = 0
    # Which axes were intentionally set (for apply_axis bookkeeping).
    active_axes: frozenset[str] = field(default_factory=lambda: frozenset(AXES))


def load_config(path: str | Path | None = None) -> MarineEOTConfig:
    """Load and validate the frozen marine-EOT config."""
    p = Path(path) if path is not None else _REPO_ROOT / DEFAULT_CONFIG
    if not p.is_file():
        # Fall back to CWD-relative (scripts often run from repo root).
        alt = Path(path) if path is not None else Path(DEFAULT_CONFIG)
        if alt.is_file():
            p = alt
        else:
            raise FileNotFoundError(f"marine-EOT config not found: {p}")
    with p.open() as f:
        raw = yaml.safe_load(f)
    levels = tuple(raw["severity_levels"])
    if levels != SEVERITY_LEVELS:
        raise ValueError(
            f"severity_levels must be {SEVERITY_LEVELS}, got {levels}"
        )
    axes = raw["axes"]
    missing = [a for a in AXES if a not in axes]
    if missing:
        raise ValueError(f"config missing axes: {missing}")
    return MarineEOTConfig(
        version=int(raw["version"]),
        frozen=str(raw["frozen"]),
        severity_levels=levels,
        axes=axes,
        expectation=dict(raw.get("expectation") or {}),
        seed=int(raw.get("seed", 1337)),
        path=p.resolve(),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def sample_params(
    cfg: MarineEOTConfig,
    rng: np.random.Generator | None = None,
    *,
    axes: Sequence[str] | None = None,
) -> TransformParams:
    """Draw one joint sample from the continuous EOT distribution."""
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    active = tuple(axes) if axes is not None else AXES
    for a in active:
        if a not in AXES:
            raise ValueError(f"unknown axis {a!r}; expected one of {AXES}")
    p = TransformParams(active_axes=frozenset(active))
    ax = cfg.axes

    if "scale" in active:
        s = ax["scale"]["sample"]
        p.scale = _uniform(rng, s["min"], s["max"])
    if "rotation" in active:
        s = ax["rotation"]["sample"]
        p.rotation_deg = _uniform(rng, s["min"], s["max"])
    if "motion_blur" in active:
        s = ax["motion_blur"]["sample"]
        p.blur_length = int(rng.integers(s["length_min"], s["length_max"] + 1))
        if p.blur_length % 2 == 0:
            p.blur_length = min(p.blur_length + 1, int(s["length_max"]))
            if p.blur_length % 2 == 0:
                p.blur_length = max(p.blur_length - 1, 1)
        p.blur_angle_deg = _uniform(rng, s["angle_deg_min"], s["angle_deg_max"])
    if "glare" in active:
        s = ax["glare"]["sample"]
        p.glare_intensity = _uniform(rng, s["intensity_min"], s["intensity_max"])
        p.glare_radius_frac = _uniform(rng, s["radius_frac_min"], s["radius_frac_max"])
        j = float(s["center_jitter_frac"])
        p.glare_cx_frac = float(np.clip(0.5 + rng.uniform(-j, j), 0.05, 0.95))
        p.glare_cy_frac = float(np.clip(0.5 + rng.uniform(-j, j), 0.05, 0.95))
    if "spray" in active:
        s = ax["spray"]["sample"]
        p.spray_coverage = _uniform(rng, s["coverage_min"], s["coverage_max"])
        p.spray_n_blobs = int(rng.integers(s["n_blobs_min"], s["n_blobs_max"] + 1))
        p.spray_seed = int(rng.integers(0, 2**31 - 1))
    if "grazing_angle" in active:
        s = ax["grazing_angle"]["sample"]
        p.grazing_strength = _uniform(rng, s["strength_min"], s["strength_max"])
    if "sea_state" in active:
        s = ax["sea_state"]["sample"]
        p.sea_mix = _uniform(rng, s["mix_min"], s["mix_max"])
        p.sea_band_top_frac = _uniform(rng, s["band_top_frac_min"], s["band_top_frac_max"])
        # Ripple scales with mix so calm samples stay calm.
        p.sea_ripple_amp = 0.5 * p.sea_mix
        p.sea_wave_freq = 1.0 + 4.0 * p.sea_mix
        p.sea_seed = int(rng.integers(0, 2**31 - 1))
    return p


def severity_params(cfg: MarineEOTConfig, axis: str, level: str) -> TransformParams:
    """Build params that apply a single axis at a frozen severity level."""
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}")
    if level not in cfg.severity_levels:
        raise ValueError(f"unknown severity {level!r}; expected {cfg.severity_levels}")
    sev = cfg.axes[axis]["severity"][level]
    p = TransformParams(active_axes=frozenset({axis}))

    if axis == "scale":
        p.scale = float(sev)
    elif axis == "rotation":
        p.rotation_deg = float(sev)
    elif axis == "motion_blur":
        p.blur_length = int(sev["length"])
        p.blur_angle_deg = float(sev["angle_deg"])
    elif axis == "glare":
        p.glare_intensity = float(sev["intensity"])
        p.glare_radius_frac = float(sev["radius_frac"])
        p.glare_cx_frac = float(sev["cx_frac"])
        p.glare_cy_frac = float(sev["cy_frac"])
    elif axis == "spray":
        p.spray_coverage = float(sev["coverage"])
        p.spray_n_blobs = int(sev["n_blobs"])
        # Deterministic seed per (axis, level) for reproducible sweeps.
        p.spray_seed = 10_000 + SEVERITY_LEVELS.index(level)
    elif axis == "grazing_angle":
        p.grazing_strength = float(sev)
    elif axis == "sea_state":
        p.sea_mix = float(sev["mix"])
        p.sea_band_top_frac = float(sev["band_top_frac"])
        p.sea_ripple_amp = float(sev["ripple_amp"])
        p.sea_wave_freq = float(sev["wave_freq"])
        p.sea_seed = 20_000 + SEVERITY_LEVELS.index(level)
    return p


# ---------------------------------------------------------------------------
# NumPy / OpenCV transforms
# ---------------------------------------------------------------------------


def _as_float01(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return float32 [0,1] image and whether the input was uint8."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3, got {image.shape}")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0, True
    out = image.astype(np.float32)
    if out.max() > 1.5:  # likely 0..255 float
        out = out / 255.0
    return np.clip(out, 0.0, 1.0), False


def _from_float01(image: np.ndarray, was_uint8: bool) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    if was_uint8:
        return (clipped * 255.0 + 0.5).astype(np.uint8)
    return clipped.astype(np.float32)


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    # Odd length keeps SAME spatial size under conv2d with pad=length//2.
    length = max(int(length), 1)
    if length % 2 == 0:
        length += 1
    if length == 1:
        return np.array([[1.0]], dtype=np.float32)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    import cv2

    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, m, (length, length))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    else:
        kernel[length // 2, length // 2] = 1.0
    return kernel


def _glare_map(
    h: int,
    w: int,
    cx_frac: float,
    cy_frac: float,
    radius_frac: float,
) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = cx_frac * (w - 1), cy_frac * (h - 1)
    r = max(radius_frac * min(h, w), 1.0)
    # Anisotropic: wider horizontally (glint on water).
    rx, ry = r * 1.4, r * 0.7
    val = np.exp(-(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2))
    return val.astype(np.float32)


def _spray_mask(
    h: int,
    w: int,
    coverage: float,
    n_blobs: int,
    seed: int,
) -> np.ndarray:
    if coverage <= 0.0 or n_blobs <= 0:
        return np.zeros((h, w), dtype=np.float32)
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=np.float32)
    # Target approximate coverage via blob radii.
    area = h * w
    target = coverage * area
    per = target / max(n_blobs, 1)
    for _ in range(n_blobs):
        cx = float(rng.uniform(0, w))
        cy = float(rng.uniform(h * 0.45, h))  # lower-half bias
        radius = max(float(np.sqrt(per / np.pi) * rng.uniform(0.6, 1.4)), 2.0)
        ys, xs = np.ogrid[0:h, 0:w]
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * radius**2)))
        mask = np.maximum(mask, blob.astype(np.float32) * float(rng.uniform(0.5, 1.0)))
    return np.clip(mask, 0.0, 1.0)


def _sea_texture(
    h: int,
    w: int,
    band_top_frac: float,
    ripple_amp: float,
    wave_freq: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Procedural blue-grey sea field for the lower band (HxWx3 float01)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = ys / max(h - 1, 1)
    xn = xs / max(w - 1, 1)
    # Base water colour (slightly varied).
    base = np.array([0.12, 0.28, 0.38], dtype=np.float32) + rng.uniform(-0.02, 0.02, 3).astype(
        np.float32
    )
    field = np.zeros((h, w, 3), dtype=np.float32)
    field[:] = base
    if ripple_amp > 0:
        phase = float(rng.uniform(0, 2 * np.pi))
        waves = (
            np.sin(2 * np.pi * wave_freq * yn * 3.0 + phase)
            + 0.55 * np.sin(2 * np.pi * wave_freq * xn + 1.3 * yn + phase)
            + 0.35 * np.sin(2 * np.pi * wave_freq * 2.1 * xn - yn + phase * 0.7)
        )
        foam = 0.5 + 0.5 * waves
        field = field + ripple_amp * foam[..., None]
        # Sparse whitecaps.
        noise = rng.random((h, w)).astype(np.float32)
        caps = (noise > 0.97).astype(np.float32) * ripple_amp
        field = field + caps[..., None]
    field = np.clip(field, 0.0, 1.0)
    # Soft vertical band mask.
    top = float(np.clip(band_top_frac, 0.0, 1.0))
    band = np.clip((yn - top) / max(1.0 - top, 1e-6), 0.0, 1.0)
    band = (band * band)[..., None]  # ease-in
    return field, band.astype(np.float32)


def apply_numpy(
    image: np.ndarray,
    params: TransformParams,
    *,
    border_value: int = 114,
) -> np.ndarray:
    """Apply marine-EOT params to an HxWx3 image (uint8 or float01)."""
    import cv2

    img, was_uint8 = _as_float01(image)
    h, w = img.shape[:2]
    # Work in uint8 space for geometric cv2 warps, then continue in float.
    work_u8 = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bv = int(border_value)

    # --- geometric: scale ---
    if "scale" in params.active_axes and abs(params.scale - 1.0) > 1e-6:
        nh = max(int(round(h * params.scale)), 1)
        nw = max(int(round(w * params.scale)), 1)
        resized = cv2.resize(work_u8, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((h, w, 3), bv, dtype=np.uint8)
        if params.scale >= 1.0:
            # Center-crop back.
            y0 = max((nh - h) // 2, 0)
            x0 = max((nw - w) // 2, 0)
            canvas[:] = resized[y0 : y0 + h, x0 : x0 + w]
        else:
            y0 = (h - nh) // 2
            x0 = (w - nw) // 2
            canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
        work_u8 = canvas

    # --- geometric: rotation ---
    if "rotation" in params.active_axes and abs(params.rotation_deg) > 1e-6:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), params.rotation_deg, 1.0)
        work_u8 = cv2.warpAffine(
            work_u8, m, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(bv, bv, bv),
        )

    # --- geometric: grazing-angle perspective ---
    if "grazing_angle" in params.active_axes and params.grazing_strength > 1e-6:
        s = float(np.clip(params.grazing_strength, 0.0, 1.0))
        # Pull top corners inward; keep bottom edge fixed (shore camera).
        inset = s * 0.35 * w
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([
            [inset, s * 0.15 * h],
            [w - 1 - inset, s * 0.15 * h],
            [w - 1, h - 1],
            [0, h - 1],
        ])
        m = cv2.getPerspectiveTransform(src, dst)
        work_u8 = cv2.warpPerspective(
            work_u8, m, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(bv, bv, bv),
        )

    img = work_u8.astype(np.float32) / 255.0

    # --- motion blur ---
    if "motion_blur" in params.active_axes and params.blur_length > 1:
        k = _motion_kernel(params.blur_length, params.blur_angle_deg)
        img = cv2.filter2D(img, -1, k)

    # --- glare ---
    if "glare" in params.active_axes and params.glare_intensity > 1e-6:
        g = _glare_map(
            h, w, params.glare_cx_frac, params.glare_cy_frac, params.glare_radius_frac
        )
        # Warm-white glint.
        tint = np.array([1.0, 0.98, 0.90], dtype=np.float32)
        img = img + (params.glare_intensity * g[..., None] * tint)
        img = np.clip(img, 0.0, 1.0)

    # --- spray ---
    if "spray" in params.active_axes and params.spray_coverage > 1e-6:
        m = _spray_mask(h, w, params.spray_coverage, params.spray_n_blobs, params.spray_seed)
        spray_color = np.array([0.92, 0.95, 0.98], dtype=np.float32)
        alpha = (0.65 * m)[..., None]
        img = img * (1.0 - alpha) + spray_color * alpha

    # --- sea state ---
    if "sea_state" in params.active_axes and params.sea_mix > 1e-6:
        tex, band = _sea_texture(
            h, w,
            params.sea_band_top_frac,
            params.sea_ripple_amp,
            params.sea_wave_freq,
            params.sea_seed,
        )
        mix = float(np.clip(params.sea_mix, 0.0, 1.0)) * band
        img = img * (1.0 - mix) + tex * mix

    return _from_float01(img, was_uint8)


# ---------------------------------------------------------------------------
# PyTorch differentiable path
# ---------------------------------------------------------------------------


def _torch_grid_affine(b: int, h: int, w: int, angle_deg: float, scale: float, device, dtype):
    import torch

    ang = angle_deg * np.pi / 180.0
    c, s = np.cos(ang), np.sin(ang)
    # Map output coords -> input coords (inverse). grid_sample expects theta
    # for affine_grid: [[a00, a01, tx], [a10, a11, ty]] in normalized coords.
    a00 = c / scale
    a01 = s / scale
    a10 = -s / scale
    a11 = c / scale
    theta = torch.tensor(
        [[[a00, a01, 0.0], [a10, a11, 0.0]]],
        device=device, dtype=dtype,
    ).expand(b, -1, -1)
    return torch.nn.functional.affine_grid(theta, size=(b, 3, h, w), align_corners=False)


def _torch_perspective_grid(b: int, h: int, w: int, strength: float, device, dtype):
    """Build a sampling grid that approximates the NumPy grazing warp."""
    import torch

    s = float(np.clip(strength, 0.0, 1.0))
    # Normalized coords in [-1, 1].
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # Inverse of the forward warp used in NumPy (approx): at the top, output
    # x is compressed, so sampling x must expand.
    # Forward: top y' ≈ -1 + s*0.15*2 stretch... use a simple model:
    #   x_src = x / (1 - s * 0.35 * (1 - y_n)/2 ) roughly
    y_n = (yy + 1.0) * 0.5  # 0 at top, 1 at bottom
    top_weight = 1.0 - y_n
    shrink = 1.0 - s * 0.35 * top_weight
    shrink = shrink.clamp(min=0.3)
    xx_src = xx / shrink
    yy_src = yy + s * 0.15 * top_weight * 2.0 * 0.0  # keep y mostly; slight
    # Match NumPy: top edge drops down by s*0.15*h → in norm: + s*0.15*2 from top.
    # Forward maps src top to y = s*0.15*h. Inverse: when sampling at yy near top,
    # pull from slightly above... For simplicity use vertical squash from top:
    yy_src = -1.0 + (yy + 1.0) / max(1.0 - s * 0.15, 0.5)
    grid = torch.stack([xx_src, yy_src], dim=-1)  # H W 2
    return grid.unsqueeze(0).expand(b, -1, -1, -1)


def apply_torch(
    image,
    params: TransformParams,
    *,
    border_value: int = 114,
):
    """Differentiable marine-EOT on a ``(B,3,H,W)`` float tensor in ``[0, 1]``.

    Gradients flow through geometric resampling and additive overlays into
    ``image`` (and thus into an upstream patch composite).
    """
    import torch
    import torch.nn.functional as F

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected Bx3xHxW, got {tuple(image.shape)}")
    b, _, h, w = image.shape
    device, dtype = image.device, image.dtype
    border = border_value / 255.0
    out = image

    need_geom = (
        ("scale" in params.active_axes and abs(params.scale - 1.0) > 1e-6)
        or ("rotation" in params.active_axes and abs(params.rotation_deg) > 1e-6)
    )
    if need_geom:
        scale = params.scale if "scale" in params.active_axes else 1.0
        angle = params.rotation_deg if "rotation" in params.active_axes else 0.0
        grid = _torch_grid_affine(b, h, w, angle, scale, device, dtype)
        out = F.grid_sample(
            out, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # Fill zeros (out-of-bounds) with border grey.
        # grid_sample with zeros pads 0; replace near-zero holes when scale<1.
        if scale < 1.0 - 1e-6:
            # Mask of pixels that sampled outside: approximate via inverse map.
            ones = torch.ones(b, 1, h, w, device=device, dtype=dtype)
            cover = F.grid_sample(
                ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            out = out * cover + border * (1.0 - cover)

    if "grazing_angle" in params.active_axes and params.grazing_strength > 1e-6:
        grid = _torch_perspective_grid(b, h, w, params.grazing_strength, device, dtype)
        ones = torch.ones(b, 1, h, w, device=device, dtype=dtype)
        cover = F.grid_sample(
            ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        out = F.grid_sample(
            out, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        out = out * cover + border * (1.0 - cover)

    if "motion_blur" in params.active_axes and params.blur_length > 1:
        k_np = _motion_kernel(params.blur_length, params.blur_angle_deg)
        k = torch.as_tensor(k_np, device=device, dtype=dtype)
        k = k.view(1, 1, k.shape[0], k.shape[1]).repeat(3, 1, 1, 1)
        pad = k_np.shape[0] // 2
        out = F.conv2d(out, k, padding=pad, groups=3)
        # Belt-and-suspenders: keep the letterboxed canvas size.
        if out.shape[-2] != h or out.shape[-1] != w:
            out = out[:, :, :h, :w]

    if "glare" in params.active_axes and params.glare_intensity > 1e-6:
        g = _glare_map(
            h, w, params.glare_cx_frac, params.glare_cy_frac, params.glare_radius_frac
        )
        g_t = torch.as_tensor(g, device=device, dtype=dtype).view(1, 1, h, w)
        tint = torch.tensor([1.0, 0.98, 0.90], device=device, dtype=dtype).view(1, 3, 1, 1)
        out = (out + params.glare_intensity * g_t * tint).clamp(0.0, 1.0)

    if "spray" in params.active_axes and params.spray_coverage > 1e-6:
        m = _spray_mask(
            h, w, params.spray_coverage, params.spray_n_blobs, params.spray_seed
        )
        m_t = torch.as_tensor(m, device=device, dtype=dtype).view(1, 1, h, w)
        spray = torch.tensor([0.92, 0.95, 0.98], device=device, dtype=dtype).view(1, 3, 1, 1)
        alpha = 0.65 * m_t
        out = out * (1.0 - alpha) + spray * alpha

    if "sea_state" in params.active_axes and params.sea_mix > 1e-6:
        tex, band = _sea_texture(
            h, w,
            params.sea_band_top_frac,
            params.sea_ripple_amp,
            params.sea_wave_freq,
            params.sea_seed,
        )
        tex_t = torch.as_tensor(tex, device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        band_t = torch.as_tensor(band, device=device, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        mix = float(np.clip(params.sea_mix, 0.0, 1.0)) * band_t
        out = out * (1.0 - mix) + tex_t * mix

    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class MarineEOT:
    """Marine-EOT transform distribution (sample + per-axis severity)."""

    def __init__(self, config: MarineEOTConfig | None = None):
        self.config = config if config is not None else load_config()
        self._rng = np.random.default_rng(self.config.seed)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> MarineEOT:
        return cls(load_config(path))

    @property
    def axes(self) -> tuple[str, ...]:
        return AXES

    @property
    def severity_levels(self) -> tuple[str, ...]:
        return self.config.severity_levels

    def sample_params(
        self,
        rng: np.random.Generator | None = None,
        *,
        axes: Sequence[str] | None = None,
    ) -> TransformParams:
        return sample_params(self.config, rng if rng is not None else self._rng, axes=axes)

    def severity_params(self, axis: str, level: str) -> TransformParams:
        return severity_params(self.config, axis, level)

    def apply(
        self,
        image: np.ndarray,
        params: TransformParams | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply a given (or freshly sampled) param draw to a NumPy image."""
        if params is None:
            params = self.sample_params(rng)
        return apply_numpy(image, params, border_value=self.config.border_value)

    def apply_axis(self, image: np.ndarray, axis: str, level: str) -> np.ndarray:
        """Eval-time single-axis severity application."""
        return self.apply(image, self.severity_params(axis, level))

    def sample(self, image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """One Monte-Carlo EOT draw (joint over all axes)."""
        return self.apply(image, self.sample_params(rng), rng=rng)

    def apply_torch(self, image, params: TransformParams | None = None, *, rng=None):
        """Differentiable apply on ``Bx3xHxW`` float ``[0,1]``."""
        if params is None:
            params = self.sample_params(rng)
        return apply_torch(image, params, border_value=self.config.border_value)

    def sample_torch(self, image, rng=None):
        return self.apply_torch(image, self.sample_params(rng), rng=rng)

    def expect_torch(self, image, n_samples: int | None = None, *, rng=None):
        """Average ``n_samples`` EOT draws (Monte-Carlo expectation).

        Useful as a differentiable surrogate for ``E_t[f(t(x))]`` when the
        downstream loss is linear; for nonlinear losses prefer sampling inside
        the training loop and averaging the loss, not the image.
        """
        import torch

        n = int(n_samples if n_samples is not None else self.config.n_samples)
        rng = rng if rng is not None else self._rng
        acc = None
        for _ in range(n):
            y = self.sample_torch(image, rng=rng)
            acc = y if acc is None else acc + y
        return acc / float(n)
