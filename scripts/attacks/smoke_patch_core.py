#!/usr/bin/env python3
"""Smoke / visual QA for the physically-realizable patch core.

Composites a learnable patch onto a seed image at the hull/superstructure
anchor, runs a few objective-agnostic optimize steps (dummy brightness loss —
no detector), and writes before/after + patch PNGs under
``results/attacks/patch_core/``.

Usage
-----
    python scripts/attacks/smoke_patch_core.py
    python scripts/attacks/smoke_patch_core.py --synthetic --steps 20
    python scripts/attacks/smoke_patch_core.py --image path/to.jpg --box 80,60,200,140
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.patch import PatchCore, xywh_to_xyxy  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "attacks" / "patch_core"


def synthetic_scene(size: int = 320, seed: int = 1337) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size // 2):
        t = y / max(size // 2 - 1, 1)
        img[y, :] = (int(130 + 40 * t), int(175 + 30 * t), int(215 - 15 * t))
    for y in range(size // 2, size):
        t = (y - size // 2) / max(size // 2 - 1, 1)
        base = np.array([28, 68, 98], dtype=np.float32)
        ripple = 10 * np.sin(np.linspace(0, 6 * np.pi, size) + t * 3)
        img[y] = np.clip(base + ripple[:, None] * 0.5, 0, 255).astype(np.uint8)
    x0, x1 = int(0.30 * size), int(0.70 * size)
    y0, y1 = int(0.42 * size), int(0.58 * size)
    img[y0:y1, x0:x1] = (45, 50, 55)
    img[y0 - int(0.07 * size) : y0, x0 + int(0.12 * size) : x1 - int(0.12 * size)] = (60, 65, 72)
    # Guaranteed patch-eligible box (≥32px short side).
    box = (float(x0), float(y0 - int(0.07 * size)), float(x1), float(y1))
    _ = rng  # seed reserved for future jittered demos
    return img, box


def load_image(path: Path, size: int) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def annotate(img: np.ndarray, box_xyxy, placement, title: str) -> Image.Image:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=2)
    draw.rectangle(
        [placement.x0, placement.y0, placement.x0 + placement.side, placement.y0 + placement.side],
        outline=(64, 220, 120), width=2,
    )
    draw.rectangle([0, 0, im.width, 18], fill=(20, 20, 20))
    draw.text((4, 2), title[:60], fill=(240, 240, 240))
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/attacks/patch.yaml")
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--box", type=str, default=None, help="x1,y1,x2,y2 on the resized canvas")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--no-eot", action="store_true", help="Disable marine-EOT during smoke")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    core = PatchCore.from_config(args.config)
    if args.no_eot:
        core.eot = None
        # Keep expectation loop at 1 sample of plain composite.
        opt = {**core.config.optimization, "eot_enabled": False, "eot_samples": 1}
        from counterusv.attacks.patch import PatchConfig
        core.config = PatchConfig(**{**core.config.__dict__, "optimization": opt})

    if args.synthetic or args.image is None:
        rgb, box = synthetic_scene(args.size, args.seed)
        src = "synthetic"
    else:
        rgb = load_image(args.image, args.size)
        if args.box:
            box = tuple(float(x) for x in args.box.split(","))
            if len(box) != 4:
                raise SystemExit("--box must be x1,y1,x2,y2")
        else:
            # Center box covering ~40% of the frame (patch-eligible).
            m = 0.3 * args.size
            box = (m, m, args.size - m, args.size - m)
        src = str(args.image)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    patch = core.init_patch(device="cpu", rng=rng)

    # Before
    before, place0 = core.composite(img_t, patch, box, rng=None, apply_eot=False)
    before_u8 = (before.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    annotate(rgb, box, place0, "clean + box (red) / patch slot (green)").save(
        args.out_dir / "00_clean_annotated.png"
    )
    annotate(before_u8, box, place0, "init patch composite").save(
        args.out_dir / "01_init_composite.png"
    )
    Image.fromarray(core.patch_to_uint8(patch)).save(args.out_dir / "patch_init.png")

    def attack_loss(patched: torch.Tensor) -> torch.Tensor:
        return -patched.mean()

    history = []
    for m in core.optimize(
        img_t, box, patch, attack_loss, steps=args.steps, lr=args.lr, rng=rng
    ):
        history.append(m)

    after, place1 = core.composite(img_t, patch, box, rng=None, apply_eot=False)
    after_u8 = (after.detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    annotate(after_u8, box, place1, f"after {args.steps} steps (dummy brightness loss)").save(
        args.out_dir / "02_optimized_composite.png"
    )
    Image.fromarray(core.patch_to_uint8(patch)).save(args.out_dir / "patch_optimized.png")

    meta = {
        "config": str(args.config.relative_to(REPO_ROOT)),
        "config_frozen": core.config.frozen,
        "seed_image": src,
        "box_xyxy": list(box),
        "steps": args.steps,
        "lr": args.lr,
        "eot_enabled": bool(core.config.eot_enabled and core.eot is not None),
        "history_tail": history[-3:] if history else [],
        "attack_loss_init": history[0]["attack"] if history else None,
        "attack_loss_final": history[-1]["attack"] if history else None,
        "note": (
            "Dummy objective (−mean brightness) only verifies the core loop; "
            "evasion/disguise losses land in later attack steps."
        ),
    }
    (args.out_dir / "smoke_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out_dir / "report.md").write_text(
        "\n".join([
            "# Patch core smoke",
            "",
            f"- Config: `{meta['config']}` (frozen {meta['config_frozen']})",
            f"- Seed: `{src}`",
            f"- Steps: {args.steps} · EOT: {meta['eot_enabled']}",
            f"- Attack loss (dummy): {meta['attack_loss_init']:.4f} → {meta['attack_loss_final']:.4f}",
            "",
            "![clean](00_clean_annotated.png)",
            "",
            "![init](01_init_composite.png)",
            "",
            "![optimized](02_optimized_composite.png)",
            "",
        ])
    )
    print(f"wrote {args.out_dir.relative_to(REPO_ROOT)}/")
    print(f"attack {meta['attack_loss_init']:.4f} → {meta['attack_loss_final']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
