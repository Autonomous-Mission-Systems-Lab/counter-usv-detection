"""Unit tests for the harmonized letterbox + SeaShips overlay contract."""

from __future__ import annotations

import numpy as np

from counterusv.data.letterbox import letterbox_image, letterbox_params, remap_boxes
from counterusv.data.overlay import SEASHIPS_BANDS, mask_seaships_overlay


def test_letterbox_params_1920x1080_to_640():
    meta = letterbox_params(1920, 1080, 640)
    assert meta.scale == 640 / 1920
    assert meta.new_w == 640
    assert meta.new_h == 360
    assert abs(meta.pad_x - 0.0) < 1e-9
    assert abs(meta.pad_y - 140.0) < 1e-9  # (640-360)/2


def test_letterbox_image_shape_and_pad():
    img = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out, meta = letterbox_image(img, input_size=640, pad_value=114)
    assert out.shape == (640, 640, 3)
    # Top pad rows should be pure 114.
    assert np.all(out[0:int(meta.pad_y)] == 114)
    # Content region should be the resized value (~200).
    y0 = int(round(meta.pad_y))
    assert out[y0 + 10, 10, 0] == 200


def test_remap_boxes_and_min_side_drop():
    meta = letterbox_params(1920, 1080, 640)
    # A 20×20 native box → 20*(640/1920) ≈ 6.67 after letterbox → drops at floor 8.
    boxes = remap_boxes([[100, 100, 20, 20]], meta, min_side=8.0)
    assert len(boxes) == 0
    # A 40×40 native box → ≈13.3 → kept.
    boxes = remap_boxes([[100, 100, 40, 40]], meta, min_side=8.0)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert abs(w - 40 * meta.scale) < 1e-6
    assert abs(x - (100 * meta.scale + meta.pad_x)) < 1e-6


def test_seaships_overlay_mask_bands():
    img = np.full((1080, 1920, 3), 50, dtype=np.uint8)
    out = mask_seaships_overlay(img, value=114)
    for y0, y1 in SEASHIPS_BANDS:
        assert np.all(out[y0:y1] == 114)
    # Mid-frame untouched.
    assert np.all(out[500:600] == 50)
