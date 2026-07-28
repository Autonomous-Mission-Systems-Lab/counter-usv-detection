#!/usr/bin/env python3
"""Render a marine-EOT sample grid (per-axis × severity).

Builds a contact sheet showing each frozen severity level (L0–L4) for every
marine-EOT axis on a single seed image, plus a small joint-sample strip.
Writes under ``results/attacks/marine_eot/``.

Usage
-----
    python scripts/attacks/render_marine_eot_grid.py
    python scripts/attacks/render_marine_eot_grid.py --image path/to.jpg
    python scripts/attacks/render_marine_eot_grid.py --synthetic   # no EO imagery needed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.marine_eot import AXES, MarineEOT  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "attacks" / "marine_eot"


def _font(size: int = 14):
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def synthetic_maritime(size: int = 320, seed: int = 1337) -> np.ndarray:
    """Procedural shore-viewpoint stand-in (sky / hull / water) for offline QA."""
    rng = np.random.default_rng(seed)
    h = w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky
    for y in range(h // 2):
        t = y / max(h // 2 - 1, 1)
        img[y, :] = (int(135 + 40 * t), int(180 + 30 * t), int(220 - 20 * t))
    # Water
    for y in range(h // 2, h):
        t = (y - h // 2) / max(h // 2 - 1, 1)
        base = np.array([30, 70, 100], dtype=np.float32)
        ripple = 12 * np.sin(np.linspace(0, 8 * np.pi, w) + t * 4)
        row = np.clip(base + ripple[:, None] * np.array([0.4, 0.6, 0.5]), 0, 255)
        img[y, :] = row.astype(np.uint8)
    # Hull (dark rectangle + cabin)
    x0, x1 = int(0.28 * w), int(0.72 * w)
    y0, y1 = int(0.42 * h), int(0.58 * h)
    img[y0:y1, x0:x1] = (40, 45, 50)
    img[y0 - int(0.08 * h) : y0, x0 + int(0.15 * w) : x1 - int(0.15 * w)] = (55, 60, 70)
    # Soft wake
    wake = rng.normal(0, 8, (y1 - y0, x1 - x0, 3))
    patch = np.clip(img[y0:y1, x0:x1].astype(np.float32) + wake, 0, 255)
    img[y0:y1, x0:x1] = patch.astype(np.uint8)
    return img


def load_seed_image(path: Path | None, size: int) -> tuple[np.ndarray, str]:
    if path is not None:
        im = Image.open(path).convert("RGB")
        im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8), str(path)
    # Try a real EO sample if the curated usv scrape exists; else synthetic.
    candidates = [
        REPO_ROOT / "data" / "raw" / "usv",
        REPO_ROOT / "data" / "eo_views" / "yolo" / "images" / "test",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            hits = sorted(root.rglob(ext))
            if hits:
                im = Image.open(hits[0]).convert("RGB").resize((size, size), Image.BILINEAR)
                return np.asarray(im, dtype=np.uint8), str(hits[0].relative_to(REPO_ROOT))
    return synthetic_maritime(size), "synthetic"


def label_cell(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _font(12)
    draw.rectangle([0, 0, img.width, 18], fill=(20, 20, 20))
    draw.text((4, 2), text[:48], fill=(240, 240, 240), font=font)
    return img


def contact_sheet(
    cells: list[Image.Image],
    cols: int,
    pad: int = 6,
    bg: tuple[int, int, int] = (30, 30, 30),
) -> Image.Image:
    if not cells:
        return Image.new("RGB", (64, 64), bg)
    w = max(c.width for c in cells)
    h = max(c.height for c in cells)
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet.paste(cell, (pad + c * (w + pad), pad + r * (h + pad)))
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/attacks/marine_eot.yaml")
    ap.add_argument("--image", type=Path, default=None, help="Seed RGB image (optional)")
    ap.add_argument("--synthetic", action="store_true", help="Force procedural seed image")
    ap.add_argument("--size", type=int, default=256, help="Cell edge length in pixels")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    eot = MarineEOT.from_config(args.config)
    if args.synthetic:
        seed_img, seed_src = synthetic_maritime(args.size, args.seed), "synthetic"
    else:
        seed_img, seed_src = load_seed_image(args.image, args.size)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Per-axis severity grid: rows = axes, cols = L0..L4
    cells: list[Image.Image] = []
    for axis in AXES:
        for level in eot.severity_levels:
            out = eot.apply_axis(seed_img, axis, level)
            cell = label_cell(Image.fromarray(out), f"{axis} {level}")
            cells.append(cell)
            Image.fromarray(out).save(args.out_dir / f"{axis}_{level}.png")

    grid = contact_sheet(cells, cols=len(eot.severity_levels), pad=6)
    # Row labels strip on the left.
    font = _font(13)
    label_w = 130
    labeled = Image.new("RGB", (grid.width + label_w, grid.height + 28), (25, 25, 25))
    labeled.paste(grid, (label_w, 28))
    draw = ImageDraw.Draw(labeled)
    draw.text((label_w + 8, 6), "marine-EOT severity grid (L0=identity → L4=extreme)", fill=(230, 230, 230), font=font)
    cell_h = cells[0].height + 6
    for i, axis in enumerate(AXES):
        y = 28 + 6 + i * cell_h + cell_h // 2 - 6
        draw.text((8, y), axis, fill=(210, 210, 210), font=font)
    grid_path = args.out_dir / "severity_grid.png"
    labeled.save(grid_path)

    # Joint-sample strip
    joint_cells = [label_cell(Image.fromarray(seed_img), "clean")]
    for i in range(5):
        samp = eot.sample(seed_img, rng=rng)
        joint_cells.append(label_cell(Image.fromarray(samp), f"sample {i + 1}"))
        Image.fromarray(samp).save(args.out_dir / f"joint_sample_{i + 1}.png")
    joint = contact_sheet(joint_cells, cols=6, pad=6)
    joint_path = args.out_dir / "joint_samples.png"
    joint.save(joint_path)

    meta = {
        "config": str(args.config),
        "config_frozen": eot.config.frozen,
        "config_version": eot.config.version,
        "seed_image": seed_src,
        "seed": args.seed,
        "axes": list(AXES),
        "severity_levels": list(eot.severity_levels),
        "outputs": {
            "severity_grid": str(grid_path.relative_to(REPO_ROOT)),
            "joint_samples": str(joint_path.relative_to(REPO_ROOT)),
        },
    }
    meta_path = args.out_dir / "grid_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    report = args.out_dir / "report.md"
    report.write_text(
        "\n".join([
            "# Marine-EOT sample grid",
            "",
            f"- Config: `{args.config.relative_to(REPO_ROOT)}` (frozen {eot.config.frozen})",
            f"- Seed image: `{seed_src}`",
            f"- Axes: {', '.join(AXES)}",
            f"- Severity levels: {', '.join(eot.severity_levels)} (L0 = identity)",
            "",
            f"![severity grid]({grid_path.name})",
            "",
            f"![joint samples]({joint_path.name})",
            "",
        ])
    )

    print(f"wrote {grid_path.relative_to(REPO_ROOT)}")
    print(f"wrote {joint_path.relative_to(REPO_ROOT)}")
    print(f"wrote {report.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
