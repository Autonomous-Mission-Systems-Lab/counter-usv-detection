#!/usr/bin/env python3
"""Evasion attack (ESR) on the ``usv`` test slice.

For each patch-eligible ``usv`` target: letterbox the image to the 640 canvas,
optimize a physically-realizable evasion patch white-box against a surrogate
detector (suppress the target-class confidence via the shared patch core +
marine-EOT expectation), then score **ESR** with the hard detector — clean vs.
attacked, swept over the marine-EOT severity ladder (per axis, L0=identity).

Outputs (under ``results/attacks/evasion/<family>/``):
  * ``esr_by_severity.json`` — ESR per (axis, level) + base (L0) ESR
  * ``instances.json``       — per-target clean/attacked status + loss history tail
  * ``report.md``            — headline ESR table + marine-EOT survival curves
  * ``gallery/``             — clean / patched / worst-case PNGs for a few targets

Compute note: optimization is the heavy part (steps × EOT samples forwards per
target). Full runs belong on a GPU host (docs/RUNPOD.md); use ``--max-images`` /
``--steps`` / ``--dry-run`` for laptop smoke wiring.

Usage
-----
    python scripts/attacks/run_evasion.py --dry-run
    python scripts/attacks/run_evasion.py --max-images 3 --steps 30
    python scripts/attacks/run_evasion.py --family yolo11s        # full slice
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplcache"))

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from counterusv.attacks.evasion import (  # noqa: E402
    EvasionAttacker,
    aggregate_esr,
    make_predict_fn,
    score_instance_esr,
    target_confidence,
    to_torch_device,
)
from counterusv.attacks.marine_eot import MarineEOT  # noqa: E402
from counterusv.attacks.patch import load_patch_config  # noqa: E402
from counterusv.attacks.transfer import (  # noqa: E402
    patch_bank_dir,
    save_patch_bank_entry,
    write_patch_bank_manifest,
)
from counterusv.data.letterbox import letterbox_image, remap_boxes  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "attacks" / "evasion"


# ---------------------------------------------------------------------------
# Slice loading
# ---------------------------------------------------------------------------


def load_usv_targets(
    data_dir: Path,
    *,
    target_class: str,
    split: str,
    source: str,
    patch_min_native_side: float,
    input_size: int,
) -> list[dict[str, Any]]:
    """One target per eligible image: largest patch-eligible ``target_class`` box.

    Returns dicts with the native path, letterbox meta, and the target box in
    both native COCO xywh and 640-canvas xyxy.
    """
    master = json.loads((data_dir / "annotations" / "coco_master.json").read_text())
    cats = {c["name"]: int(c["id"]) for c in master.get("categories") or []}
    if target_class not in cats:
        raise KeyError(f"class {target_class!r} not in master categories {sorted(cats)}")
    target_cat = cats[target_class]
    id_to_img = {int(im["id"]): im for im in master["images"]}

    ann_by_img: dict[int, list[list[float]]] = defaultdict(list)
    for a in master.get("annotations") or []:
        if int(a["category_id"]) == target_cat:
            ann_by_img[int(a["image_id"])].append([float(v) for v in a["bbox"]])

    splits = pd.read_csv(data_dir / "splits" / "eo_image_splits.csv")
    sel = splits[(splits["split"] == split) & (splits["source"] == source)]
    ids = sorted(int(x) for x in sel["image_id"])

    targets: list[dict[str, Any]] = []
    for iid in ids:
        boxes = ann_by_img.get(iid) or []
        # Patch-eligible on the native shortest side (HARMONIZATION floor).
        eligible = [b for b in boxes if min(b[2], b[3]) >= patch_min_native_side]
        if not eligible:
            continue
        native_xywh = max(eligible, key=lambda b: b[2] * b[3])  # largest by area
        im = id_to_img.get(iid)
        if im is None:
            continue
        path = data_dir / im["file_name"]
        if not path.is_file():
            continue
        # Meta needs native size; use recorded size when present, else the box.
        w = int(im.get("width") or 0)
        h = int(im.get("height") or 0)
        targets.append({
            "image_id": iid,
            "path": path,
            "native_xywh": native_xywh,
            "native_wh": (w, h),
            "input_size": input_size,
        })
    return targets


def prepare_canvas(target: dict[str, Any], input_size: int):
    """Letterbox the native image to 640; return (canvas_rgb_u8, target_xyxy)."""
    rgb = np.asarray(Image.open(target["path"]).convert("RGB"), dtype=np.uint8)
    canvas, meta = letterbox_image(rgb, input_size)
    box_lb = remap_boxes([target["native_xywh"]], meta)  # (1,4) xywh in canvas
    x, y, w, h = box_lb[0]
    target_xyxy = (float(x), float(y), float(x + w), float(y + h))
    return canvas, target_xyxy


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


def save_gallery(
    out_dir: Path,
    image_id: int,
    clean_rgb: np.ndarray,
    patched_rgb: np.ndarray,
    target_xyxy,
    placement,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def annotate(arr, title):
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        x1, y1, x2, y2 = target_xyxy
        d.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=2)
        if placement is not None:
            d.rectangle(
                [placement.x0, placement.y0,
                 placement.x0 + placement.side, placement.y0 + placement.side],
                outline=(64, 220, 120), width=2,
            )
        d.rectangle([0, 0, im.width, 16], fill=(20, 20, 20))
        d.text((3, 2), title[:70], fill=(240, 240, 240))
        return im

    annotate(clean_rgb, f"id={image_id} clean (target red)").save(
        out_dir / f"{image_id}_clean.png")
    annotate(patched_rgb, f"id={image_id} evasion patch (slot green)").save(
        out_dir / f"{image_id}_patched.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    out_path: Path,
    *,
    family: str,
    cfg,
    n_targets: int,
    n_attackable: int,
    esr: dict[str, dict[str, float | int]],
    axes: list[str],
    levels: list[str],
    steps: int,
) -> None:
    def cell(axis, level, field="esr"):
        v = esr.get(f"{axis}:{level}")
        return f"{v[field]:.2f}" if v else "—"

    base_key = f"{axes[0]}:L0" if axes else None
    base = esr.get(base_key) if base_key else None
    lines = [
        f"# Evasion (ESR) — {family}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"White-box evasion patches vs. **{family}** on the `{cfg.target_class}` "
        f"test slice. Success = target detected clean, suppressed when attacked "
        f"(conf ≥ {cfg.conf_threshold}, IoU ≥ {cfg.iou_match}). "
        f"Optimized {steps} steps.",
        "",
        f"- Targets (patch-eligible): **{n_targets}** · clean-detected "
        f"(attackable): **{n_attackable}**",
        f"- Base ESR (L0, no transform): **{base['esr']:.2f}**"
        + (f" ({base['n_success']}/{base['n_attackable']})" if base else "")
        if base else "- Base ESR: —",
        "",
        "## Marine-EOT survival — ESR by axis × severity",
        "",
        "| axis | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for axis in axes:
        lines.append(
            f"| {axis} | " + " | ".join(cell(axis, lv) for lv in levels) + " |"
        )
    lines += [
        "",
        "### Patch-attributable ESR (transform-adjusted)",
        "",
        "Denominator excludes instances the same transform suppresses un-patched, "
        "isolating the patch's contribution from transform-induced non-detection.",
        "",
        "| axis | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for axis in axes:
        lines.append(
            f"| {axis} | "
            + " | ".join(cell(axis, lv, "esr_patch_attributable") for lv in levels)
            + " |"
        )
    lines += [
        "",
        "L0 is identical across axes (identity transform) = the base ESR. "
        "Falling ESR at higher severity is the honest robustness readout: the "
        "attack must survive the marine-EOT distribution to 'count' (RQ1 "
        "robustness condition). Compare the two tables: where raw ESR rises with "
        "severity but patch-attributable ESR does not, the transform (not the "
        "patch) is suppressing the detection.",
        "",
        "Notes:",
        "- Patches crafted white-box on the surrogate; grey-box / black-box "
        "transfer are the access-level step.",
        "- Confidence/IoU thresholds are reported above per METRICS.md; scope is "
        "full non-detection only.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="evasion.yaml override")
    ap.add_argument("--patch-config", type=Path, default=None)
    ap.add_argument("--family", type=str, default=None, help="surrogate override")
    ap.add_argument("--device", type=str, default=None,
                    help="auto|cpu|mps|0|cuda (default: evasion.yaml; use 0 on RunPod)")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--source", type=str, default="usv")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="override optimize steps")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-axes", type=str, nargs="+", default=None,
                    help="subset of marine-EOT axes for the ESR sweep")
    ap.add_argument("--gallery", type=int, default=4, help="save first N targets")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="load the slice + report wiring; no model / optimize")
    ap.add_argument(
        "--save-patches", action="store_true",
        help="write patch_bank/ (attacked PNGs + patch tensors) for transfer eval",
    )
    args = ap.parse_args()

    from counterusv.attacks.evasion import load_evasion_config
    cfg = load_evasion_config(args.config)
    patch_cfg = load_patch_config(args.patch_config)
    family = args.family or cfg.surrogate_family
    device = str(to_torch_device(args.device or cfg.device))
    seed = args.seed if args.seed is not None else cfg.seed
    steps = args.steps if args.steps is not None else (cfg.steps or patch_cfg.steps)
    lr = args.lr if args.lr is not None else (cfg.lr or patch_cfg.lr)
    axes = args.eval_axes or cfg.severity_axes
    levels = cfg.severity_levels
    input_size = int(patch_cfg.eligibility.get("input_size", 640))

    targets = load_usv_targets(
        args.data_dir,
        target_class=cfg.target_class,
        split=args.split,
        source=args.source,
        patch_min_native_side=patch_cfg.patch_min_side,
        input_size=input_size,
    )
    if args.max_images is not None:
        targets = targets[: args.max_images]

    print(f"[evasion] surrogate={family} device={device} seed={seed}")
    print(f"[evasion] {args.split}∩{args.source} patch-eligible targets: {len(targets)}")
    print(f"[evasion] steps={steps} lr={lr} eval axes={axes} levels={levels}")

    out_dir = args.out_dir / family
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        meta = {
            "dry_run": True,
            "family": family,
            "device": device,
            "n_targets": len(targets),
            "steps": steps,
            "lr": lr,
            "eval_axes": axes,
            "levels": levels,
            "image_ids": [t["image_id"] for t in targets],
        }
        (out_dir / "dry_run.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[evasion] dry-run wiring OK → {out_dir / 'dry_run.json'}")
        return 0

    import torch

    torch.manual_seed(seed)
    attacker = EvasionAttacker.from_configs(
        evasion_config=args.config,
        patch_config=args.patch_config,
        device=device,
    )
    # Reload with the resolved family if it differs from the config default.
    if family != cfg.surrogate_family:
        from counterusv.attacks.evasion import DifferentiableSurrogate
        surrogate = DifferentiableSurrogate.from_family(family, device=device)
        attacker.surrogate = surrogate
        attacker.class_idx = surrogate.class_index(cfg.target_class)

    eot = MarineEOT.from_config(
        REPO_ROOT / cfg.evaluation.get("marine_eot_config", "configs/attacks/marine_eot.yaml")
    )
    predict_fn = make_predict_fn(
        attacker.surrogate.baseline,
        conf=cfg.conf_threshold,
        iou=cfg.nms_iou,
        imgsz=input_size,
    )

    results = []
    bank_rows: list[dict] = []
    bank = patch_bank_dir(out_dir) if args.save_patches else None
    craft_device = attacker.surrogate.device
    for i, t in enumerate(targets):
        rng = np.random.default_rng(seed + i)
        canvas_rgb, target_xyxy = prepare_canvas(t, input_size)
        canvas_t = (
            torch.from_numpy(canvas_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(craft_device)
        )

        conf0 = target_confidence(
            attacker.surrogate, canvas_t, target_xyxy, attacker.class_idx,
            target_pad_frac=cfg.target_pad_frac,
        )
        patch, history = attacker.craft(canvas_t, target_xyxy, rng=rng)
        patched_t, placement = attacker.core.composite(
            canvas_t, patch, target_xyxy, rng=None, apply_eot=False)
        patched_rgb = (patched_t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                       * 255 + 0.5).astype(np.uint8)

        res = score_instance_esr(
            predict_fn, canvas_rgb, patched_rgb, target_xyxy,
            class_name=cfg.target_class,
            conf=cfg.conf_threshold, iou_match=cfg.iou_match,
            marine_eot=eot, severity_axes=axes, severity_levels=levels,
            image_id=t["image_id"], steps=steps,
            attack_loss_init=history[0]["attack"] if history else None,
            attack_loss_final=history[-1]["attack"] if history else None,
        )
        results.append(res)
        if bank is not None:
            bank_rows.append(save_patch_bank_entry(
                bank,
                image_id=t["image_id"],
                target_xyxy=target_xyxy,
                placement=placement,
                attacked_rgb=patched_rgb,
                patch_chw=patch.detach().clamp(0, 1).cpu().numpy(),
            ))
        if i < args.gallery:
            save_gallery(out_dir / "gallery", t["image_id"], canvas_rgb,
                         patched_rgb, target_xyxy, placement)
        print(f"  [{i+1}/{len(targets)}] id={t['image_id']} "
              f"conf {conf0:.3f} clean={res.clean_detected} "
              f"loss {res.attack_loss_init:.3f}→{res.attack_loss_final:.3f} "
              f"L0_suppressed={not res.attacked.get(f'{axes[0]}:L0',{}).get('detected', True)}",
              flush=True)

    if bank is not None:
        write_patch_bank_manifest(
            bank,
            attack="evasion",
            surrogate=family,
            instances=bank_rows,
            extra={
                "seed": seed,
                "steps": steps,
                "lr": lr,
                "target_class": cfg.target_class,
                "input_size": input_size,
            },
        )
        print(f"[evasion] wrote patch bank → {bank.relative_to(REPO_ROOT)}")

    esr = aggregate_esr(results)
    n_attackable = sum(1 for r in results if r.clean_detected)

    (out_dir / "instances.json").write_text(json.dumps([
        {
            "image_id": r.image_id,
            "target_xyxy": list(r.target_xyxy),
            "clean_detected": r.clean_detected,
            "clean_score": r.clean_score,
            "attack_loss_init": r.attack_loss_init,
            "attack_loss_final": r.attack_loss_final,
            "attacked": r.attacked,
        } for r in results
    ], indent=2) + "\n")
    (out_dir / "esr_by_severity.json").write_text(json.dumps({
        "family": family,
        "target_class": cfg.target_class,
        "conf_threshold": cfg.conf_threshold,
        "iou_match": cfg.iou_match,
        "steps": steps,
        "n_targets": len(results),
        "n_attackable": n_attackable,
        "esr": esr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    write_report(
        out_dir / "report.md", family=family, cfg=cfg,
        n_targets=len(results), n_attackable=n_attackable,
        esr=esr, axes=axes, levels=levels, steps=steps,
    )
    print(f"[evasion] wrote {out_dir}/ (report.md, esr_by_severity.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
