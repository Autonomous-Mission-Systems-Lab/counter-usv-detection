#!/usr/bin/env python3
"""Disguise attack (TMSR) on the ``usv`` test slice.

For each patch-eligible ``usv`` target and each target benign class (fishing /
recreational): letterbox → optimize a physically-realizable disguise patch
white-box against a surrogate (raise benign class, suppress ``usv``) → score
**TMSR** with the hard detector across the marine-EOT severity ladder.

Outputs (under ``results/attacks/disguise/<family>/<benign>/``):
  * ``tmsr_by_severity.json`` · ``instances.json`` · ``report.md`` · ``gallery/``

Also writes a family-level ``summary.md`` comparing benign classes.

Full runs belong on a GPU host (docs/RUNPOD.md). Use ``--dry-run`` /
``--max-images`` / ``--steps`` for laptop wiring.

Usage
-----
    python scripts/attacks/run_disguise.py --dry-run
    python scripts/attacks/run_disguise.py --max-images 2 --steps 20 --benign-class fishing
    python scripts/attacks/run_disguise.py --family yolo11s --device 0   # full (GPU)
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

from counterusv.attacks.disguise import (  # noqa: E402
    DisguiseAttacker,
    aggregate_tmsr,
    load_disguise_config,
    make_predict_fn,
    score_instance_tmsr,
    to_torch_device,
)
from counterusv.attacks.evasion import DifferentiableSurrogate, target_confidence  # noqa: E402
from counterusv.attacks.marine_eot import MarineEOT  # noqa: E402
from counterusv.attacks.patch import load_patch_config  # noqa: E402
from counterusv.data.letterbox import letterbox_image, remap_boxes  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "attacks" / "disguise"


def load_usv_targets(
    data_dir: Path,
    *,
    true_class: str,
    split: str,
    source: str,
    patch_min_native_side: float,
) -> list[dict[str, Any]]:
    master = json.loads((data_dir / "annotations" / "coco_master.json").read_text())
    cats = {c["name"]: int(c["id"]) for c in master.get("categories") or []}
    if true_class not in cats:
        raise KeyError(f"class {true_class!r} not in master categories {sorted(cats)}")
    target_cat = cats[true_class]
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
        eligible = [b for b in boxes if min(b[2], b[3]) >= patch_min_native_side]
        if not eligible:
            continue
        native_xywh = max(eligible, key=lambda b: b[2] * b[3])
        im = id_to_img.get(iid)
        if im is None:
            continue
        path = data_dir / im["file_name"]
        if not path.is_file():
            continue
        targets.append({
            "image_id": iid,
            "path": path,
            "native_xywh": native_xywh,
            "native_wh": (int(im.get("width") or 0), int(im.get("height") or 0)),
        })
    return targets


def prepare_canvas(target: dict[str, Any], input_size: int):
    rgb = np.asarray(Image.open(target["path"]).convert("RGB"), dtype=np.uint8)
    canvas, meta = letterbox_image(rgb, input_size)
    box_lb = remap_boxes([target["native_xywh"]], meta)
    x, y, w, h = box_lb[0]
    return canvas, (float(x), float(y), float(x + w), float(y + h))


def save_gallery(out_dir, image_id, clean_rgb, patched_rgb, target_xyxy, placement, title):
    out_dir.mkdir(parents=True, exist_ok=True)

    def annotate(arr, label):
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
        d.text((3, 2), label[:70], fill=(240, 240, 240))
        return im

    annotate(clean_rgb, f"id={image_id} clean").save(out_dir / f"{image_id}_clean.png")
    annotate(patched_rgb, f"id={image_id} {title}").save(out_dir / f"{image_id}_patched.png")


def write_report(out_path, *, family, benign, cfg, n_targets, n_eligible, tmsr, axes, levels, steps):
    def cell(axis, level, field="tmsr"):
        v = tmsr.get(f"{axis}:{level}")
        return f"{v[field]:.2f}" if v else "—"

    base_key = f"{axes[0]}:L0" if axes else None
    base = tmsr.get(base_key) if base_key else None
    lines = [
        f"# Disguise (TMSR) — {family} → `{benign}`",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"White-box disguise patches vs. **{family}**, flipping `{cfg.true_class}` → "
        f"`{benign}`. Success = clean was correct hostile, attacked has `{benign}` "
        f"with IoU ≥ {cfg.iou_match} and conf ≥ {cfg.conf_threshold}. "
        f"Optimized {steps} steps.",
        "",
        f"- Targets (patch-eligible): **{n_targets}** · clean-hostile (eligible): "
        f"**{n_eligible}**",
        (
            f"- Base TMSR (L0): **{base['tmsr']:.2f}** "
            f"({base['n_success']}/{base['n_eligible']})"
            if base else "- Base TMSR: —"
        ),
        "",
        "## Marine-EOT survival — TMSR by axis × severity",
        "",
        "| axis | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for axis in axes:
        lines.append(f"| {axis} | " + " | ".join(cell(axis, lv) for lv in levels) + " |")
    lines += [
        "",
        "### Patch-attributable TMSR (transform-adjusted)",
        "",
        "Denominator excludes instances where the same transform already yields "
        f"the `{benign}` label on the clean canvas.",
        "",
        "| axis | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for axis in axes:
        lines.append(
            f"| {axis} | "
            + " | ".join(cell(axis, lv, "tmsr_patch_attributable") for lv in levels)
            + " |"
        )
    lines += [
        "",
        "L0 is identical across axes (identity) = the base TMSR. Compare raw vs "
        "patch-attributable: where raw rises with severity but attributable does "
        "not, the transform (not the patch) is inducing the benign label.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def run_one_benign(
    *,
    benign: str,
    cfg,
    patch_cfg,
    family: str,
    device: str,
    targets: list,
    steps: int,
    lr: float,
    axes: list,
    levels: list,
    seed: int,
    gallery: int,
    out_root: Path,
    input_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir = out_root / family / benign
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[disguise] → {benign}  out={out_dir.relative_to(REPO_ROOT)}")

    if dry_run:
        meta = {
            "dry_run": True,
            "family": family,
            "benign_class": benign,
            "n_targets": len(targets),
            "steps": steps,
        }
        (out_dir / "dry_run.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta

    import torch

    attacker = DisguiseAttacker.from_configs(
        benign_class=benign, device=device
    )
    if family != cfg.surrogate_family:
        surrogate = DifferentiableSurrogate.from_family(family, device=device)
        attacker.surrogate = surrogate
        attacker.true_idx = surrogate.class_index(cfg.true_class)
        attacker.benign_idx = surrogate.class_index(benign)

    eot = MarineEOT.from_config(
        REPO_ROOT / cfg.evaluation.get(
            "marine_eot_config", "configs/attacks/marine_eot.yaml")
    )
    predict_fn = make_predict_fn(
        attacker.surrogate.baseline,
        conf=cfg.conf_threshold,
        iou=cfg.nms_iou,
        imgsz=input_size,
    )

    results = []
    craft_device = attacker.surrogate.device
    for i, t in enumerate(targets):
        rng = np.random.default_rng(seed + i)
        canvas_rgb, target_xyxy = prepare_canvas(t, input_size)
        canvas_t = (
            torch.from_numpy(canvas_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(craft_device)
        )
        usv_c = target_confidence(
            attacker.surrogate, canvas_t, target_xyxy, attacker.true_idx,
            target_pad_frac=cfg.target_pad_frac,
        )
        ben_c = target_confidence(
            attacker.surrogate, canvas_t, target_xyxy, attacker.benign_idx,
            target_pad_frac=cfg.target_pad_frac,
        )
        patch, history = attacker.craft(canvas_t, target_xyxy, rng=rng)
        patched_t, placement = attacker.core.composite(
            canvas_t, patch, target_xyxy, rng=None, apply_eot=False)
        patched_rgb = (
            patched_t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5
        ).astype(np.uint8)

        res = score_instance_tmsr(
            predict_fn, canvas_rgb, patched_rgb, target_xyxy,
            true_class=cfg.true_class, benign_class=benign,
            conf=cfg.conf_threshold, iou_match=cfg.iou_match,
            marine_eot=eot, severity_axes=axes, severity_levels=levels,
            image_id=t["image_id"], steps=steps,
            attack_loss_init=history[0]["attack"] if history else None,
            attack_loss_final=history[-1]["attack"] if history else None,
        )
        results.append(res)
        if i < gallery:
            save_gallery(
                out_dir / "gallery", t["image_id"], canvas_rgb, patched_rgb,
                target_xyxy, placement, f"disguise→{benign}",
            )
        l0 = res.attacked.get(f"{axes[0]}:L0", {})
        print(
            f"  [{i+1}/{len(targets)}] id={t['image_id']} "
            f"usv={usv_c:.3f} {benign}={ben_c:.3f} clean_h={res.clean_hostile} "
            f"loss {res.attack_loss_init:.3f}→{res.attack_loss_final:.3f} "
            f"L0_tmsr={bool(l0.get('tmsr'))}",
            flush=True,
        )

    tmsr = aggregate_tmsr(results)
    n_eligible = sum(1 for r in results if r.clean_hostile)
    (out_dir / "instances.json").write_text(json.dumps([
        {
            "image_id": r.image_id,
            "benign_class": r.benign_class,
            "target_xyxy": list(r.target_xyxy),
            "clean_hostile": r.clean_hostile,
            "clean_hostile_score": r.clean_hostile_score,
            "attack_loss_init": r.attack_loss_init,
            "attack_loss_final": r.attack_loss_final,
            "attacked": r.attacked,
        } for r in results
    ], indent=2) + "\n")
    payload = {
        "family": family,
        "true_class": cfg.true_class,
        "benign_class": benign,
        "conf_threshold": cfg.conf_threshold,
        "iou_match": cfg.iou_match,
        "steps": steps,
        "n_targets": len(results),
        "n_eligible": n_eligible,
        "tmsr": tmsr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "tmsr_by_severity.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(
        out_dir / "report.md", family=family, benign=benign, cfg=cfg,
        n_targets=len(results), n_eligible=n_eligible, tmsr=tmsr,
        axes=axes, levels=levels, steps=steps,
    )
    print(f"[disguise] wrote {out_dir.relative_to(REPO_ROOT)}/")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--patch-config", type=Path, default=None)
    ap.add_argument("--family", type=str, default=None)
    ap.add_argument("--device", type=str, default=None,
                    help="auto|cpu|mps|0|cuda (default: disguise.yaml; use 0 on RunPod)")
    ap.add_argument("--benign-class", type=str, default=None,
                    help="single benign class; default = all in config")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--source", type=str, default="usv")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-axes", type=str, nargs="+", default=None)
    ap.add_argument("--gallery", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_disguise_config(args.config)
    patch_cfg = load_patch_config(args.patch_config)
    family = args.family or cfg.surrogate_family
    device = str(to_torch_device(args.device or cfg.device))
    seed = args.seed if args.seed is not None else cfg.seed
    steps = args.steps if args.steps is not None else (cfg.steps or patch_cfg.steps)
    lr = args.lr if args.lr is not None else (cfg.lr or patch_cfg.lr)
    axes = args.eval_axes or cfg.severity_axes
    levels = cfg.severity_levels
    input_size = int(patch_cfg.eligibility.get("input_size", 640))
    benigns = (
        [args.benign_class] if args.benign_class
        else list(cfg.target_benign_classes)
    )
    for b in benigns:
        if b not in cfg.target_benign_classes:
            raise SystemExit(
                f"--benign-class {b!r} not in {cfg.target_benign_classes}"
            )

    targets = load_usv_targets(
        args.data_dir,
        true_class=cfg.true_class,
        split=args.split,
        source=args.source,
        patch_min_native_side=patch_cfg.patch_min_side,
    )
    if args.max_images is not None:
        targets = targets[: args.max_images]

    print(f"[disguise] surrogate={family} device={device} seed={seed}")
    print(f"[disguise] {args.split}∩{args.source} targets: {len(targets)}")
    print(f"[disguise] benign classes: {benigns}")
    print(f"[disguise] steps={steps} lr={lr} axes={axes}")

    import torch
    torch.manual_seed(seed)

    summaries = []
    for benign in benigns:
        summaries.append(run_one_benign(
            benign=benign, cfg=cfg, patch_cfg=patch_cfg, family=family,
            device=device, targets=targets, steps=steps, lr=lr,
            axes=axes, levels=levels, seed=seed, gallery=args.gallery,
            out_root=args.out_dir, input_size=input_size, dry_run=args.dry_run,
        ))

    # Family-level summary when more than one benign class.
    if not args.dry_run and len(summaries) >= 1:
        fam_dir = args.out_dir / family
        lines = [
            f"# Disguise (TMSR) summary — {family}",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "| benign class | eligible | base TMSR (L0) |",
            "|---|---:|---:|",
        ]
        for s in summaries:
            tmsr = s.get("tmsr") or {}
            base = next(iter(tmsr.values()), None) if tmsr else None
            # Prefer scale:L0
            base = tmsr.get("scale:L0") or base
            rate = f"{base['tmsr']:.2f} ({base['n_success']}/{base['n_eligible']})" if base else "—"
            lines.append(
                f"| {s.get('benign_class')} | {s.get('n_eligible', '—')} | {rate} |"
            )
        lines += [
            "",
            "Per-class detail: `results/attacks/disguise/"
            f"{family}/<benign>/report.md`.",
            "",
        ]
        (fam_dir / "summary.md").write_text("\n".join(lines) + "\n")
        print(f"[disguise] wrote {fam_dir.relative_to(REPO_ROOT)}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
