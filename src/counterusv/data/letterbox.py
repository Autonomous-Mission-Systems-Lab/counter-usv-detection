"""Letterbox transform + box remap (see ``data/HARMONIZATION.md``).

Native resolution stays on disk; loaders call these at train/eval time.

For an image of native size ``(W, H)`` to square input ``S ∈ {640, 1280}``::

    s = S / max(W, H)
    new_w, new_h = round(W * s), round(H * s)
    pad_x = (S - new_w) / 2   # centered
    pad_y = (S - new_h) / 2
    pad value = 114 per RGB channel

Boxes (COCO ``[x, y, w, h]`` in native pixels) remap as::

    [x, y, w, h]_input = [x*s + pad_x, y*s + pad_y, w*s, h*s]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_PAD_VALUE = 114
DEFAULT_INPUT_SIZE = 640


@dataclass(frozen=True)
class LetterboxMeta:
    """Scale / pad applied to one image; enough to remap boxes and undo later."""

    input_size: int
    scale: float
    pad_x: float
    pad_y: float
    new_w: int
    new_h: int
    orig_w: int
    orig_h: int


def letterbox_params(orig_w: int, orig_h: int, input_size: int = DEFAULT_INPUT_SIZE) -> LetterboxMeta:
    """Compute centered letterbox parameters (no pixels touched)."""
    s = input_size / max(orig_w, orig_h)
    new_w = int(round(orig_w * s))
    new_h = int(round(orig_h * s))
    pad_x = (input_size - new_w) / 2.0
    pad_y = (input_size - new_h) / 2.0
    return LetterboxMeta(
        input_size=input_size, scale=s, pad_x=pad_x, pad_y=pad_y,
        new_w=new_w, new_h=new_h, orig_w=orig_w, orig_h=orig_h,
    )


def letterbox_image(
    image: np.ndarray,
    input_size: int = DEFAULT_INPUT_SIZE,
    pad_value: int = DEFAULT_PAD_VALUE,
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize (bilinear) + centered pad to ``input_size × input_size``.

    Parameters
    ----------
    image : HxWxC uint8 RGB
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got shape {image.shape}")
    h, w = image.shape[:2]
    meta = letterbox_params(w, h, input_size)

    # cv2 is optional at import; prefer it for bilinear quality, fall back to PIL.
    try:
        import cv2
        resized = cv2.resize(image, (meta.new_w, meta.new_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        from PIL import Image
        resized = np.asarray(
            Image.fromarray(image).resize((meta.new_w, meta.new_h), Image.BILINEAR))

    out = np.full((input_size, input_size, 3), pad_value, dtype=np.uint8)
    x0 = int(round(meta.pad_x))
    y0 = int(round(meta.pad_y))
    out[y0:y0 + meta.new_h, x0:x0 + meta.new_w] = resized
    return out, meta


def remap_boxes(
    boxes: Sequence[Sequence[float]] | np.ndarray,
    meta: LetterboxMeta,
    *,
    clip: bool = True,
    min_side: float = 0.0,
) -> np.ndarray:
    """Remap native COCO ``[x,y,w,h]`` boxes to letterboxed input coords.

    Drops boxes whose remapped shortest side is ``< min_side``
    (see ``data/HARMONIZATION.md`` letterbox / eligibility floors).
    Returns an ``(N, 4)`` float64 array (possibly empty).
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] * meta.scale + meta.pad_x
    out[:, 1] = b[:, 1] * meta.scale + meta.pad_y
    out[:, 2] = b[:, 2] * meta.scale
    out[:, 3] = b[:, 3] * meta.scale
    if clip:
        x2 = np.minimum(out[:, 0] + out[:, 2], meta.input_size)
        y2 = np.minimum(out[:, 1] + out[:, 3], meta.input_size)
        out[:, 0] = np.maximum(out[:, 0], 0.0)
        out[:, 1] = np.maximum(out[:, 1], 0.0)
        out[:, 2] = np.maximum(x2 - out[:, 0], 0.0)
        out[:, 3] = np.maximum(y2 - out[:, 1], 0.0)
    if min_side > 0:
        keep = np.minimum(out[:, 2], out[:, 3]) >= min_side
        out = out[keep]
    return out


def coco_to_yolo_norm(boxes_xywh: np.ndarray, input_size: int) -> np.ndarray:
    """COCO ``[x,y,w,h]`` (input coords) → YOLO ``[cx,cy,w,h]`` normalized to [0,1]."""
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    b = np.asarray(boxes_xywh, dtype=np.float64)
    cx = (b[:, 0] + b[:, 2] / 2.0) / input_size
    cy = (b[:, 1] + b[:, 3] / 2.0) / input_size
    w = b[:, 2] / input_size
    h = b[:, 3] / input_size
    return np.stack([cx, cy, w, h], axis=1)
