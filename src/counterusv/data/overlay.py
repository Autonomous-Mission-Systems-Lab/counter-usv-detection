"""SeaShips burned-in overlay mask (see ``data/HARMONIZATION.md``).

SeaShips 1920×1080 frames carry a fixed surveillance overlay:
  * timestamp band along the top
  * camera-ID band along the bottom

Measured on a 30-image aggregate (high text-edge / extreme-luminance score) the
text lives roughly in rows ``[~65, ~105]`` (top) and ``[~990, ~1045]`` (bottom).
We mask slightly larger fixed bands so every glyph is covered with margin, and
fill with the same neutral grey as the letterbox pad (114) — a non-destructive,
fixed-region load-time op. Image geometry and all boxes are unchanged.

Applied to ``source == "seaships"`` only, in both train and eval.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Locked for the native SeaShips release (1920 × 1080). Absolute pixel rows
# (half-open intervals). If a SeaShips frame ever arrives at a different size,
# bands are scaled proportionally from these reference coords.
SEASHIPS_NATIVE_W = 1920
SEASHIPS_NATIVE_H = 1080
SEASHIPS_TOP_BAND = (0, 110)       # [y0, y1) — timestamp
SEASHIPS_BOTTOM_BAND = (980, 1080) # [y0, y1) — camera ID
SEASHIPS_BANDS: tuple[tuple[int, int], ...] = (SEASHIPS_TOP_BAND, SEASHIPS_BOTTOM_BAND)

DEFAULT_MASK_VALUE = 114  # matches configs/base.yaml letterbox_pad


def seaships_bands_for_size(height: int, width: int) -> list[tuple[int, int]]:
    """Scale the locked bands to an actual image size (identity at 1920×1080)."""
    if height == SEASHIPS_NATIVE_H and width == SEASHIPS_NATIVE_W:
        return list(SEASHIPS_BANDS)
    sy = height / SEASHIPS_NATIVE_H
    out = []
    for y0, y1 in SEASHIPS_BANDS:
        out.append((int(round(y0 * sy)), int(round(y1 * sy))))
    return out


def mask_seaships_overlay(
    image: np.ndarray,
    *,
    value: int = DEFAULT_MASK_VALUE,
    bands: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Overwrite the SeaShips text bands with ``value`` (in-place safe copy).

    Parameters
    ----------
    image : HxWx3 uint8 RGB (native resolution, pre-letterbox)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {image.shape}")
    out = np.array(image, copy=True)
    h, w = out.shape[:2]
    for y0, y1 in (bands if bands is not None else seaships_bands_for_size(h, w)):
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if y1 > y0:
            out[y0:y1, :, :] = value
    return out
