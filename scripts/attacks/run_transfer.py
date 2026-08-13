#!/usr/bin/env python3
"""Access-level transfer eval: grey-box + black-box on saved patches.

Prerequisite: craft with ``--save-patches`` so each white-box run writes a
``patch_bank/`` (attacked letterbox PNGs + patch tensors). This script hard-
evaluates those fixed attacks on other detectors — no re-optimization.

Access levels (``configs/attacks/access_levels.yaml`` / TRANSFER_PROTOCOL.md):

* **white** — surrogate == target (optional re-score via ``--include-white``)
* **grey**  — cross-scale YOLO (yolo11s ↔ yolo11l)
* **black** — craft on YOLO surrogate → eval on held-out ``rtdetr_l``

Usage
-----
    # After craft:
    python scripts/attacks/run_evasion.py --family yolo11s --device 0 --save-patches

    # Transfer (default targets = grey peer + rtdetr_l):
    python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --device 0
    python scripts/attacks/run_transfer.py --attack disguise --surrogate yolo11s \\
        --benign-class fishing --device 0

    python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplcache"))

from PIL import Image  # noqa: E402

from counterusv.attacks.disguise import load_disguise_config  # noqa: E402
from counterusv.attacks.evasion import (  # noqa: E402
    load_evasion_config,
    to_torch_device,
)
from counterusv.attacks.marine_eot import MarineEOT  # noqa: E402
from counterusv.attacks.patch import load_patch_config  # noqa: E402
from counterusv.attacks.transfer import (  # noqa: E402
    access_level,
    aggregate_esr,
    aggregate_tmsr,
    base_rate_from_agg,
    default_transfer_targets,
    load_access_levels_config,
    load_eval_baseline,
    load_patch_bank,
    make_predict_fn,
    patch_bank_dir,
    score_instance_esr,
    score_instance_tmsr,
    transfer_gap,
)
from counterusv.data.letterbox import letterbox_image, remap_boxes  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "attacks" / "transfer"


def _image_path_and_box(
    data_dir: Path,
    image_id: int,
    *,
    target_class: str,
    patch_min_native_side: float,
) -> tuple[Path, list[float]]:
    """Resolve native path + largest patch-eligible box for one image id."""
    master = json.loads((data_dir / "annotations" / "coco_master.json").read_text())
    cats = {c["name"]: int(c["id"]) for c in master.get("categories") or []}
    if target_class not in cats:
        raise KeyError(target_class)
    cat = cats[target_class]
    im = next(i for i in master["images"] if int(i["id"]) == int(image_id))
    boxes = [
        [float(v) for v in a["bbox"]]
        for a in master.get("annotations") or []
        if int(a["image_id"]) == int(image_id) and int(a["category_id"]) == cat
    ]
    eligible = [b for b in boxes if min(b[2], b[3]) >= patch_min_native_side]
    if not eligible:
        raise RuntimeError(f"no patch-eligible {target_class} box on image {image_id}")
    native_xywh = max(eligible, key=lambda b: b[2] * b[3])
    return data_dir / im["file_name"], native_xywh


def prepare_clean(
    data_dir: Path,
    image_id: int,
    *,
    target_class: str,
    patch_min_native_side: float,
    input_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    path, native_xywh = _image_path_and_box(
        data_dir, image_id,
        target_class=target_class,
        patch_min_native_side=patch_min_native_side,
    )
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    canvas, meta = letterbox_image(rgb, input_size)
    x, y, w, h = remap_boxes([native_xywh], meta)[0]
    return canvas, (float(x), float(y), float(x + w), float(y + h))


def write_transfer_report(
    out_path: Path,
    *,
    attack: str,
    surrogate: str,
    target: str,
    level: str,
    rate_name: str,
    base: dict[str, Any] | None,
    white_base: dict[str, Any] | None,
    axes: list[str],
    levels: list[str],
    agg: dict[str, dict[str, Any]],
    rate_key: str,
    attr_key: str,
    extra_lines: list[str] | None = None,
) -> None:
    def cell(axis, level_name, field):
        v = agg.get(f"{axis}:{level_name}")
        return f"{v[field]:.2f}" if v else "—"

    gap = None
    if base is not None and white_base is not None:
        gap = transfer_gap(float(white_base[rate_key]), float(base[rate_key]))

    lines = [
        f"# Transfer ({attack.upper()}) — {surrogate} → {target}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Access level: **{level}**",
        f"- Surrogate (craft): `{surrogate}`",
        f"- Target (eval): `{target}`",
    ]
    if base:
        lines.append(
            f"- Base {rate_name} (L0): **{base[rate_key]:.2f}** "
            f"({base.get('n_success', '?')}/{base.get('n_attackable', base.get('n_eligible', '?'))})"
        )
    if white_base is not None and gap is not None:
        lines.append(
            f"- White-box {rate_name} (same patches on `{surrogate}`): "
            f"**{white_base[rate_key]:.2f}**"
        )
        lines.append(
            f"- Transfer gap (target − white): **{gap:+.2f}**"
        )
    lines += [
        "",
        f"## Marine-EOT — raw {rate_name}",
        "",
        "| axis | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for axis in axes:
        lines.append(
            f"| {axis} | " + " | ".join(cell(axis, lv, rate_key) for lv in levels) + " |"
        )
    lines += [
        "",
        f"### Patch-attributable {rate_name}",
        "",
        "| axis | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for axis in axes:
        lines.append(
            f"| {axis} | "
            + " | ".join(cell(axis, lv, attr_key) for lv in levels)
            + " |"
        )
    if extra_lines:
        lines += ["", *extra_lines]
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", choices=["evasion", "disguise"], required=True)
    ap.add_argument("--surrogate", type=str, required=True,
                    help="family the patches were crafted on")
    ap.add_argument("--targets", type=str, nargs="+", default=None,
                    help="eval families (default: grey peer + rtdetr_l)")
    ap.add_argument("--include-white", action="store_true",
                    help="also re-score on the surrogate (white-box check)")
    ap.add_argument("--benign-class", type=str, default=None,
                    help="required for disguise (fishing|recreational)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--craft-root", type=Path, default=None,
                    help="root of craft outputs (default: results/attacks/<attack>)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--access-config", type=Path, default=None)
    ap.add_argument("--eval-axes", type=str, nargs="+", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    al_cfg = load_access_levels_config(args.access_config)
    patch_cfg = load_patch_config()
    input_size = int(patch_cfg.eligibility.get("input_size", 640))
    device = str(to_torch_device(args.device))

    if args.attack == "evasion":
        atk_cfg = load_evasion_config()
        true_class = atk_cfg.target_class
        conf = atk_cfg.conf_threshold
        iou_match = atk_cfg.iou_match
        nms_iou = atk_cfg.nms_iou
        axes = args.eval_axes or atk_cfg.severity_axes
        levels = atk_cfg.severity_levels
        eot_path = atk_cfg.evaluation.get("marine_eot_config")
        craft_root = args.craft_root or (REPO_ROOT / "results" / "attacks" / "evasion")
        bank_path = patch_bank_dir(craft_root / args.surrogate, al_cfg.patch_bank_dirname)
        benign = None
    else:
        if not args.benign_class:
            raise SystemExit("--benign-class is required for --attack disguise")
        atk_cfg = load_disguise_config()
        true_class = atk_cfg.true_class
        conf = atk_cfg.conf_threshold
        iou_match = atk_cfg.iou_match
        nms_iou = atk_cfg.nms_iou
        axes = args.eval_axes or atk_cfg.severity_axes
        levels = atk_cfg.severity_levels
        eot_path = atk_cfg.evaluation.get("marine_eot_config")
        craft_root = args.craft_root or (REPO_ROOT / "results" / "attacks" / "disguise")
        bank_path = patch_bank_dir(
            craft_root / args.surrogate / args.benign_class, al_cfg.patch_bank_dirname
        )
        benign = args.benign_class

    targets = args.targets or default_transfer_targets(
        args.surrogate, al_cfg, include_white=args.include_white
    )
    if args.include_white and args.surrogate not in targets:
        targets = [args.surrogate, *targets]

    print(f"[transfer] attack={args.attack} surrogate={args.surrogate} "
          f"targets={targets} device={device}")
    print(f"[transfer] patch bank: {bank_path}")

    if args.dry_run:
        ok = (bank_path / "manifest.json").is_file()
        meta = {
            "dry_run": True,
            "attack": args.attack,
            "surrogate": args.surrogate,
            "targets": targets,
            "benign_class": benign,
            "bank_path": str(bank_path),
            "bank_present": ok,
            "access_levels": {
                t: access_level(args.surrogate, t, al_cfg) for t in targets
            },
        }
        out = args.out_dir / args.attack / f"{args.surrogate}_dry_run.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[transfer] dry-run → {out.relative_to(REPO_ROOT)} bank_present={ok}")
        return 0 if ok else 1

    manifest, entries = load_patch_bank(bank_path)
    if args.max_images is not None:
        entries = entries[: args.max_images]
    print(f"[transfer] loaded {len(entries)} patched instances "
          f"(manifest n={manifest.get('n_instances')})")

    eot = MarineEOT.from_config(REPO_ROOT / (eot_path or "configs/attacks/marine_eot.yaml"))

    # White-box reference: prefer an in-run white re-score; else load craft JSON.
    white_base_cell: dict[str, Any] | None = None
    if args.attack == "evasion":
        white_json = craft_root / args.surrogate / "esr_by_severity.json"
        rate_key_w = "esr"
    else:
        white_json = (
            craft_root / args.surrogate / (benign or "") / "tmsr_by_severity.json"
        )
        rate_key_w = "tmsr"
    if white_json.is_file():
        wraw = json.loads(white_json.read_text())
        wagg = wraw.get(rate_key_w) or wraw.get("esr") or wraw.get("tmsr") or {}
        white_base_cell = base_rate_from_agg(wagg, rate_key=rate_key_w, axes=axes)

    summary_rows: list[dict[str, Any]] = []
    for target in targets:
        level = access_level(args.surrogate, target, al_cfg)
        print(f"[transfer] → {target} ({level})")
        baseline = load_eval_baseline(target, device=device)
        predict_fn = make_predict_fn(
            baseline, conf=conf, iou=nms_iou, imgsz=input_size
        )

        results = []
        for i, ent in enumerate(entries):
            clean_rgb, xyxy = prepare_clean(
                args.data_dir,
                ent.image_id,
                target_class=true_class,
                patch_min_native_side=patch_cfg.patch_min_side,
                input_size=input_size,
            )
            # Prefer bank target_xyxy (exact craft geometry).
            tgt = ent.target_xyxy
            if args.attack == "evasion":
                res = score_instance_esr(
                    predict_fn, clean_rgb, ent.attacked_rgb, tgt,
                    class_name=true_class, conf=conf, iou_match=iou_match,
                    marine_eot=eot, severity_axes=axes, severity_levels=levels,
                    image_id=ent.image_id,
                )
            else:
                res = score_instance_tmsr(
                    predict_fn, clean_rgb, ent.attacked_rgb, tgt,
                    true_class=true_class, benign_class=benign,
                    conf=conf, iou_match=iou_match,
                    marine_eot=eot, severity_axes=axes, severity_levels=levels,
                    image_id=ent.image_id,
                )
            results.append(res)
            if (i + 1) % 10 == 0 or i + 1 == len(entries):
                print(f"  [{i+1}/{len(entries)}] scored", flush=True)

        out_dir = args.out_dir / args.attack / f"{args.surrogate}_to_{target}"
        if benign:
            out_dir = out_dir / benign
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.attack == "evasion":
            agg = aggregate_esr(results)
            rate_key, attr_key, rate_name = "esr", "esr_patch_attributable", "ESR"
            n_ok = sum(1 for r in results if r.clean_detected)
            instances = [
                {
                    "image_id": r.image_id,
                    "target_xyxy": list(r.target_xyxy),
                    "clean_detected": r.clean_detected,
                    "clean_score": r.clean_score,
                    "attacked": r.attacked,
                }
                for r in results
            ]
        else:
            agg = aggregate_tmsr(results)
            rate_key, attr_key, rate_name = "tmsr", "tmsr_patch_attributable", "TMSR"
            n_ok = sum(1 for r in results if r.clean_hostile)
            instances = [
                {
                    "image_id": r.image_id,
                    "benign_class": r.benign_class,
                    "target_xyxy": list(r.target_xyxy),
                    "clean_hostile": r.clean_hostile,
                    "clean_hostile_score": r.clean_hostile_score,
                    "attacked": r.attacked,
                }
                for r in results
            ]

        base = base_rate_from_agg(agg, rate_key=rate_key, axes=axes)
        if level == "white":
            white_base_cell = base

        payload = {
            "attack": args.attack,
            "surrogate": args.surrogate,
            "target": target,
            "access_level": level,
            "benign_class": benign,
            "conf_threshold": conf,
            "iou_match": iou_match,
            "n_instances": len(results),
            "n_eligible": n_ok,
            rate_name.lower(): agg,
            "base_L0": base,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "instances.json").write_text(json.dumps(instances, indent=2) + "\n")
        (out_dir / f"{rate_name.lower()}_by_severity.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        write_transfer_report(
            out_dir / "report.md",
            attack=args.attack,
            surrogate=args.surrogate,
            target=target,
            level=level,
            rate_name=rate_name,
            base=base,
            white_base=white_base_cell if level != "white" else None,
            axes=axes,
            levels=levels,
            agg=agg,
            rate_key=rate_key,
            attr_key=attr_key,
        )
        summary_rows.append({
            "target": target,
            "access_level": level,
            "base": base,
            "rate_key": rate_key,
        })
        print(f"[transfer] wrote {out_dir.relative_to(REPO_ROOT)}/")

    # Family summary across targets for this surrogate.
    sum_dir = args.out_dir / args.attack
    if benign:
        sum_path = sum_dir / f"{args.surrogate}_{benign}_summary.md"
    else:
        sum_path = sum_dir / f"{args.surrogate}_summary.md"
    lines = [
        f"# Transfer summary — {args.attack} crafted on `{args.surrogate}`",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| target | access | base rate (L0) |",
        "|---|---|---:|",
    ]
    for row in summary_rows:
        b = row["base"]
        rk = row["rate_key"]
        if b:
            cell = f"{b[rk]:.2f} ({b.get('n_success', '?')}/{b.get('n_attackable', b.get('n_eligible', '?'))})"
        else:
            cell = "—"
        lines.append(f"| {row['target']} | {row['access_level']} | {cell} |")
    lines += [
        "",
        "White-box craft numbers (if not re-scored here) live under "
        f"`results/attacks/{args.attack}/{args.surrogate}/`.",
        "",
    ]
    sum_path.write_text("\n".join(lines) + "\n")
    print(f"[transfer] wrote {sum_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
