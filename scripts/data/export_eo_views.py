#!/usr/bin/env python3
"""Export per-split EO views for detector frameworks (no pixel copies).

Reads the COCO master + leakage-controlled split manifests and writes:

  * ``data/eo_views/coco/<split>.json`` — COCO JSON filtered to the split, with
    detector-ineligible boxes dropped; ``file_name`` still points at ``raw/...``
  * ``data/eo_views/yolo/`` — Ultralytics-ready layout using **symlinks** to the
    raw images (unique ``{source}__{stem}`` names to avoid basename collisions)
    plus YOLO-txt labels in native-normalized coords (cx,cy,w,h in [0,1] of the
    *native* image — Ultralytics letterboxes internally; we also write a
    ``letterbox_contract.json`` documenting our load-time contract for the
    custom Dataset path).

Does **not** re-split. Honors ``det_min_pixels_on_target`` (native shortest side).

Usage
-----
    python scripts/data/export_eo_views.py
    python scripts/data/export_eo_views.py --splits train val test --input-size 640
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.data.letterbox import letterbox_params  # noqa: E402


def unique_stem(source: str, file_name: str) -> str:
    stem = Path(file_name).stem
    return f"{source}__{stem}"


def load_master(path: Path) -> dict:
    return json.loads(path.read_text())


def export_coco_split(
    master: dict,
    image_ids: set[int],
    *,
    det_min_side: float,
    keep_cat_ids: set[int],
    out_path: Path,
) -> dict:
    id_to_img = {int(im["id"]): im for im in master["images"]}
    images = [id_to_img[i] for i in sorted(image_ids) if i in id_to_img]
    anns = []
    dropped = 0
    for a in master["annotations"]:
        iid = int(a["image_id"])
        if iid not in image_ids:
            continue
        if int(a["category_id"]) not in keep_cat_ids:
            dropped += 1
            continue
        x, y, w, h = a["bbox"]
        if min(w, h) < det_min_side:
            dropped += 1
            continue
        anns.append(a)
    coco = {
        "info": {**(master.get("info") or {}), "description": f"EO split export ({out_path.stem})"},
        "images": images,
        "annotations": anns,
        "categories": [c for c in master["categories"] if c["id"] in keep_cat_ids],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco))
    return {
        "images": len(images),
        "annotations": len(anns),
        "dropped_boxes": dropped,
        "path": str(out_path),
    }


def export_yolo_split(
    master: dict,
    image_ids: set[int],
    *,
    data_dir: Path,
    out_root: Path,
    split: str,
    det_min_side: float,
    cat_id_to_yolo: dict[int, int],
) -> dict:
    """Symlink images + write YOLO-txt labels (native-normalized)."""
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    id_to_img = {int(im["id"]): im for im in master["images"]}
    anns_by: dict[int, list] = defaultdict(list)
    dropped = 0
    for a in master["annotations"]:
        iid = int(a["image_id"])
        if iid not in image_ids:
            continue
        if int(a["category_id"]) not in cat_id_to_yolo:
            dropped += 1
            continue
        x, y, w, h = a["bbox"]
        if min(w, h) < det_min_side:
            dropped += 1
            continue
        anns_by[iid].append(a)

    n_imgs, n_lbls, missing = 0, 0, 0
    for iid in sorted(image_ids):
        im = id_to_img.get(iid)
        if im is None:
            continue
        src = data_dir / im["file_name"]
        if not src.is_file():
            missing += 1
            continue
        stem = unique_stem(im.get("source", "src"), im["file_name"])
        # Preserve extension for the symlink target name.
        ext = Path(im["file_name"]).suffix or ".jpg"
        link = img_dir / f"{stem}{ext}"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src.resolve())
        W, H = int(im["width"]), int(im["height"])
        lines = []
        for a in anns_by.get(iid, []):
            cid = int(a["category_id"])
            if cid not in cat_id_to_yolo:
                continue
            x, y, w, h = a["bbox"]
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            lines.append(f"{cat_id_to_yolo[cid]} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_imgs += 1
        n_lbls += len(lines)
    return {
        "images": n_imgs, "labels": n_lbls, "dropped_boxes": dropped,
        "missing_files": missing, "images_dir": str(img_dir), "labels_dir": str(lbl_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--input-size", type=int, default=640,
                    help="recorded in the letterbox contract (YOLO uses native labels)")
    ap.add_argument("--det-min-side", type=float, default=None,
                    help="override; default from configs/base.yaml")
    ap.add_argument("--drop-non-target", action="store_true", default=None,
                    help="omit static_aid/unknown_other (vessel-only ablation); "
                         "default from configs/base.yaml detector.non_target_policy")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <data-dir>/eo_views")
    args = ap.parse_args()

    data_dir = args.data_dir
    out = args.out or (data_dir / "eo_views")
    cfg_path = REPO_ROOT / "configs" / "base.yaml"
    base = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    det_min = args.det_min_side
    if det_min is None:
        det_min = float((base.get("data") or {}).get("det_min_pixels_on_target", 8))
    pad = int((base.get("data") or {}).get("letterbox_pad", 114))

    master = load_master(data_dir / "annotations" / "coco_master.json")
    splits = pd.read_csv(data_dir / "splits" / "eo_image_splits.csv")

    # --- Detector class-space & label policy (configs/base.yaml `detector`) ----
    # Trained classes = canonical taxonomy classes with >=1 EO instance:
    #   * `exclude_classes` (e.g. working_service, 0 EO boxes) never get a head slot.
    #   * non_target (static_aid/unknown_other) kept as explicit classes unless the
    #     policy/flag drops them for the vessel-only ablation.
    # Class ids map 1:1 to canonical names (data.yaml `names`) for the downstream
    # class-kinematics consistency check.
    det_cfg = base.get("detector") or {}
    exclude = set(det_cfg.get("exclude_classes") or [])
    policy = str(det_cfg.get("non_target_policy", "keep")).lower()
    drop_non_target = args.drop_non_target if args.drop_non_target is not None else (policy == "drop")
    non_names = {"static_aid", "unknown_other"}

    # Stable YOLO class order = master category order (0-indexed for YOLO).
    cats = sorted(master["categories"], key=lambda c: c["id"])
    cats = [c for c in cats if c["name"] not in exclude]
    if drop_non_target:
        cats = [c for c in cats if c["name"] not in non_names]
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(cats)}
    keep_cat_ids = set(cat_id_to_yolo)

    summary = {
        "det_min_side": det_min,
        "non_target_policy": "drop" if drop_non_target else "keep",
        "excluded_classes": sorted(exclude),
        "input_size": args.input_size,
        "letterbox_pad": pad,
        "classes": [c["name"] for c in cats],
        "splits": {},
    }

    for split in args.splits:
        ids = set(int(x) for x in splits.loc[splits["split"] == split, "image_id"])
        print(f"[export] {split}: {len(ids)} images from split manifest")
        coco_stats = export_coco_split(
            master, ids, det_min_side=det_min,
            keep_cat_ids=keep_cat_ids,
            out_path=out / "coco" / f"{split}.json",
        )
        yolo_stats = export_yolo_split(
            master, ids, data_dir=data_dir, out_root=out / "yolo", split=split,
            det_min_side=det_min, cat_id_to_yolo=cat_id_to_yolo,
        )
        summary["splits"][split] = {"coco": coco_stats, "yolo": yolo_stats}
        print(f"  coco: {coco_stats['images']} imgs / {coco_stats['annotations']} boxes "
              f"(dropped {coco_stats['dropped_boxes']})")
        print(f"  yolo: {yolo_stats['images']} imgs / {yolo_stats['labels']} boxes "
              f"(dropped {yolo_stats['dropped_boxes']}, missing {yolo_stats['missing_files']})")

    # Ultralytics data.yaml. `path` must be ABSOLUTE: Ultralytics resolves a
    # relative path against the CWD / datasets dir, not this file's location.
    # Re-run this exporter on each machine (Mac → RunPod) after syncing raw
    # imagery — image symlinks are absolute and machine-local anyway.
    yolo_root = out / "yolo"
    data_yaml = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c["name"] for i, c in enumerate(cats)},
        # role per class id (hostile|benign|non_target) — canonical taxonomy axis.
        # non_target ids are localized but never scored on the hostile/benign axis.
        "roles": {i: c.get("role", "benign") for i, c in enumerate(cats)},
        # Document that labels are native-normalized; Ultralytics letterboxes at train.
        # Our custom Dataset (counterusv.data.eo_dataset) applies the centered-pad-114
        # contract itself for non-YOLO families / QA.
        "note": "Labels are YOLO-normalized in NATIVE image coords. "
                "Harmonization letterbox (centered pad 114) is applied by "
                "counterusv.data.EODetectionDataset for the custom loader path.",
    }
    (yolo_root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    # Record the letterbox contract used by the custom Dataset.
    contract = {
        "input_size": args.input_size,
        "alternate_input_size": 1280,
        "pad_value": pad,
        "padding": "centered",
        "det_min_side_native_px": det_min,
        "scale_formula": "s = S / max(W, H); pad_x=(S-new_w)/2; pad_y=(S-new_h)/2",
        "example_1920x1080_to_640": letterbox_params(1920, 1080, args.input_size).__dict__,
        "seaships_overlay_bands_yx": {"top": [0, 110], "bottom": [980, 1080],
                                       "native": [1920, 1080], "fill": pad},
        "normalize": "imagenet mean/std for pretrained; else [0,1]",
        "augment": {
            "train": "framework mosaic+multiscale ON (Ultralytics); Dataset is letterbox-only",
            "eval": "OFF — letterbox only",
        },
    }
    (out / "letterbox_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[export] wrote {out}")
    print(f"[export] classes ({len(cats)}): {', '.join(c['name'] for c in cats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
