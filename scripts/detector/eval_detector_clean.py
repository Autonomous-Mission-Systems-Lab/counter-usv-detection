#!/usr/bin/env python3
"""Clean-mAP evaluation for detector baselines (augmentation OFF).

Runs each family's ``best.pt`` through the test split with Ultralytics predict
(no TTA), scores with ``pycocotools``, and reports:

  * **shore operational** (headline) — test ∩ ``{seaships, smd, usv}``
    (all shore-viewpoint eval sources; equals the full test split today)
  * **per-source** — seaships / smd / usv
  * **per-class AP** — call out ``usv`` + small benign classes
  * **per-size-bin** — COCO area small/medium/large + shortest-side bins

Evaluated at the models' native training resolution (**640**). Off-resolution
eval (e.g. 1280) is not run by default: it penalises resolution-sensitive
architectures (RT-DETR collapses far off its training size) and muddies the
headline. Pass ``--imgsz`` explicitly only for a deliberate sensitivity study.

Outputs
-------
  ``results/detector_baselines/clean_map/<family>_s{imgsz}.json``
  ``results/detector_baselines/report.md``

Usage
-----
    # After pulling weights from RunPod:
    python scripts/detector/eval_detector_clean.py --all
    python scripts/detector/eval_detector_clean.py --family yolo11s
    python scripts/detector/eval_detector_clean.py --all --dry-run   # wiring only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "detector"))

from counterusv.eval.coco_map import (  # noqa: E402
    filter_detections,
    filter_master,
    master_cat_maps,
    yolo_to_master_cat_ids,
    coco_eval_summary,
    COCO_AREA_BINS,
    SHORTEST_SIDE_BINS,
)
from train_detector import (  # noqa: E402
    load_family_config,
    resolve_device,
    roster_families,
)

FAMILIES_YAML = REPO_ROOT / "configs" / "detector" / "families.yaml"
DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "results" / "detector_baselines"
DEFAULT_OUT = DEFAULT_WEIGHTS_ROOT / "clean_map"

# Classes to highlight in the report (threat + small-craft ambiguity).
HIGHLIGHT_CLASSES = (
    "usv", "military", "small_craft", "fishing", "sailing", "recreational",
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_data_yaml_names(data_yaml: Path) -> dict[int, str]:
    d = _load_yaml(data_yaml)
    names = d.get("names") or {}
    # YAML may load keys as int or str.
    return {int(k): str(v) for k, v in names.items()}


def build_slices(splits: pd.DataFrame, operational_sources: list[str]) -> dict[str, set[int]]:
    """Named image-id sets for the clean-mAP report."""
    test = splits.loc[splits["split"] == "test"]
    test_ids = set(int(x) for x in test["image_id"])
    slices: dict[str, set[int]] = {
        "test_overall": test_ids,
        "shore_operational": set(
            int(x) for x in test.loc[test["source"].isin(operational_sources), "image_id"]
        ),
    }
    for src in sorted(test["source"].unique()):
        slices[f"source:{src}"] = set(
            int(x) for x in test.loc[test["source"] == src, "image_id"]
        )
    return slices


def resolve_weights(weights_root: Path, family: str) -> Path:
    candidates = [
        weights_root / family / "weights" / "best.pt",
        weights_root / family / "best.pt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No best.pt for {family!r} under {weights_root}. "
        f"Pull results from RunPod first (see docs/RUNPOD.md)."
    )


def load_model(arch: str, weights: Path):
    from ultralytics import RTDETR, YOLO

    arch_l = (arch or "yolo11").lower()
    if arch_l in ("rtdetr", "rt-detr", "transformer"):
        return RTDETR(str(weights))
    return YOLO(str(weights))


def predict_coco_detections(
    model,
    master: dict,
    image_ids: set[int],
    *,
    data_dir: Path,
    imgsz: int,
    device: str,
    yolo_to_master: dict[int, int],
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
    batch: int = 8,
) -> list[dict]:
    """Run Ultralytics predict; return COCO-format detections (master cat ids).

    Uses ``stream=True`` so Results are not all held in RAM (critical on MPS /
    for ~1k large maritime images). Paths are also chunked by ``batch``.
    Image ids are matched by **chunk order** (Ultralytics preserves source order),
    not by ``r.path`` string equality (which breaks across resolve/symlink forms).
    """
    id_to_img = {int(im["id"]): im for im in master["images"]}
    # Parallel lists: path string + image_id (same order).
    path_strs: list[str] = []
    path_ids: list[int] = []
    missing = 0
    for iid in sorted(image_ids):
        im = id_to_img.get(iid)
        if im is None:
            continue
        p = data_dir / im["file_name"]
        if not p.is_file():
            missing += 1
            continue
        path_strs.append(str(p))
        path_ids.append(iid)

    if missing:
        print(f"  [predict] WARNING: {missing} images missing on disk")
    if not path_strs:
        return []

    # MPS / unified memory: keep batches small. CUDA can go higher.
    if str(device).lower() in ("mps", "cpu") and batch > 4:
        batch = 4
        print(f"  [predict] capping batch={batch} on device={device}")

    dets: list[dict] = []
    n_done = 0
    n_with_boxes = 0
    for i in range(0, len(path_strs), batch):
        chunk_paths = path_strs[i:i + batch]
        chunk_ids = path_ids[i:i + batch]
        # stream=True → generator; associate results with chunk by index.
        for j, r in enumerate(model.predict(
            source=chunk_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            augment=False,       # clean eval — no TTA
            device=device,
            verbose=False,
            stream=True,
        )):
            if j >= len(chunk_ids):
                print(f"  [predict] WARNING: extra result beyond chunk size")
                break
            iid = chunk_ids[j]
            n_done += 1
            if n_done % 100 == 0 or n_done == len(path_strs):
                print(f"  [predict] {n_done}/{len(path_strs)} images "
                      f"({n_with_boxes} with boxes so far)", flush=True)
            if r.boxes is None or len(r.boxes) == 0:
                continue
            n_with_boxes += 1
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, yi in zip(xyxy, confs, clss):
                if int(yi) not in yolo_to_master:
                    continue
                w = float(x2 - x1)
                h = float(y2 - y1)
                if w <= 0 or h <= 0:
                    continue
                dets.append({
                    "image_id": int(iid),
                    "category_id": int(yolo_to_master[int(yi)]),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(sc),
                })
    if n_done and not dets:
        print(f"  [predict] WARNING: processed {n_done} images but recorded 0 "
              f"detections ({n_with_boxes} had boxes before class remap)")
    return dets


def evaluate_slices(
    master: dict,
    detections: list[dict],
    slices: dict[str, set[int]],
    *,
    keep_cat_ids: set[int],
    det_min_side: float,
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, ids in slices.items():
        gt = filter_master(
            master, ids, keep_cat_ids=keep_cat_ids, det_min_side=det_min_side)
        dets = filter_detections(detections, ids)
        out[name] = coco_eval_summary(gt, dets, id_to_name=id_to_name)

    # Size bins on the headline shore slice.
    for slice_name in ("shore_operational",):
        ids = slices[slice_name]
        size_block: dict[str, Any] = {}
        for bin_name, (a0, a1) in COCO_AREA_BINS.items():
            if bin_name == "all":
                continue
            gt = filter_master(
                master, ids, keep_cat_ids=keep_cat_ids, det_min_side=det_min_side,
                min_area=a0, max_area=a1)
            dets = filter_detections(detections, ids)
            size_block[f"coco_{bin_name}"] = coco_eval_summary(
                gt, dets, id_to_name=id_to_name)
        for bin_name, (s0, s1) in SHORTEST_SIDE_BINS.items():
            gt = filter_master(
                master, ids, keep_cat_ids=keep_cat_ids, det_min_side=det_min_side,
                min_side=s0, max_side=s1)
            dets = filter_detections(detections, ids)
            size_block[bin_name] = coco_eval_summary(
                gt, dets, id_to_name=id_to_name)
        out[f"{slice_name}__size_bins"] = size_block
    return out


def fmt_ap(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def write_report(
    out_path: Path,
    *,
    family_results: dict[str, dict[str, Any]],
    operational_sources: list[str],
    imgsizes: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# Clean-mAP report — detector baselines")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("Augmentation **OFF**. Primary metric: COCO **mAP@[.5:.95]** via "
                 "`pycocotools`. Headline slice: **shore operational** "
                 f"(test ∩ {operational_sources}; all shore-viewpoint eval sources).")
    lines.append("")
    lines.append("## Headline (shore operational)")
    lines.append("")
    lines.append("| family | imgsz | mAP@[.5:.95] | mAP@50 | images | GT boxes |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for fam, by_sz in family_results.items():
        for sz in imgsizes:
            key = f"s{sz}"
            if key not in by_sz:
                continue
            shore = by_sz[key]["slices"].get("shore_operational") or {}
            lines.append(
                f"| {fam} | {sz} | {fmt_ap(shore.get('mAP_50_95'))} | "
                f"{fmt_ap(shore.get('mAP_50'))} | {shore.get('n_images', 0)} | "
                f"{shore.get('n_gt', 0)} |"
            )
    lines.append("")
    lines.append("## Per-source (test split)")
    lines.append("")
    lines.append("| family | imgsz | source | mAP@[.5:.95] | mAP@50 | images |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for fam, by_sz in family_results.items():
        for sz in imgsizes:
            key = f"s{sz}"
            if key not in by_sz:
                continue
            for sk, block in sorted(by_sz[key]["slices"].items()):
                if not sk.startswith("source:"):
                    continue
                src = sk.split(":", 1)[1]
                lines.append(
                    f"| {fam} | {sz} | {src} | {fmt_ap(block.get('mAP_50_95'))} | "
                    f"{fmt_ap(block.get('mAP_50'))} | {block.get('n_images', 0)} |"
                )
    lines.append("")
    lines.append("## Per-class AP (shore operational, highlight classes)")
    lines.append("")
    lines.append("| family | imgsz | class | AP@[.5:.95] | AP@50 |")
    lines.append("|---|---:|---|---:|---:|")
    for fam, by_sz in family_results.items():
        for sz in imgsizes:
            key = f"s{sz}"
            if key not in by_sz:
                continue
            shore = by_sz[key]["slices"].get("shore_operational") or {}
            per = shore.get("per_class") or {}
            # Highlight first, then the rest alphabetically.
            ordered = list(HIGHLIGHT_CLASSES) + sorted(
                c for c in per if c not in HIGHLIGHT_CLASSES)
            for cls in ordered:
                if cls not in per:
                    continue
                ap = per[cls]
                lines.append(
                    f"| {fam} | {sz} | {cls} | {fmt_ap(ap.get('AP_50_95'))} | "
                    f"{fmt_ap(ap.get('AP_50'))} |"
                )
    lines.append("")
    lines.append("## Size bins (shore operational)")
    lines.append("")
    lines.append("| family | imgsz | bin | mAP@[.5:.95] | mAP@50 | GT boxes |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for fam, by_sz in family_results.items():
        for sz in imgsizes:
            key = f"s{sz}"
            if key not in by_sz:
                continue
            bins = by_sz[key]["slices"].get("shore_operational__size_bins") or {}
            for bn, block in bins.items():
                lines.append(
                    f"| {fam} | {sz} | {bn} | {fmt_ap(block.get('mAP_50_95'))} | "
                    f"{fmt_ap(block.get('mAP_50'))} | {block.get('n_gt', 0)} |"
                )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Detector-eligibility floor: native shortest side ≥ 8 px "
                 "(see `data/HARMONIZATION.md`).")
    lines.append("- ABOShips / McShips are train-only and do not appear in these "
                 "eval slices.")
    lines.append("- Thin `usv` test count (~38 images) → high-variance per-class AP; "
                 "still included in the shore headline (same viewpoint) but call out "
                 "`usv` / `source:usv` separately when interpreting.")
    lines.append("- Evaluated at the models' native training resolution (640).")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--family", type=str)
    ap.add_argument("--weights-root", type=Path, default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640],
                    help="input size(s) to evaluate at (default: 640, the "
                         "training resolution)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch", type=int, default=8,
                    help="predict batch size (auto-capped to 4 on mps/cpu)")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low conf for COCOeval (default Ultralytics-style 0.001)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-predict", action="store_true",
                    help="reuse existing detections JSON under --out if present")
    args = ap.parse_args()

    device = resolve_device(args.device)
    data_dir = args.data_dir
    master = json.loads((data_dir / "annotations" / "coco_master.json").read_text())
    splits = pd.read_csv(data_dir / "splits" / "eo_image_splits.csv")
    base = _load_yaml(REPO_ROOT / "configs" / "base.yaml")
    operational = list((base.get("data") or {}).get(
        "eval_operational_sources", ["seaships", "smd", "usv"]))
    det_min = float((base.get("data") or {}).get("det_min_pixels_on_target", 8))
    exclude = set((base.get("detector") or {}).get("exclude_classes") or [])
    non_policy = str((base.get("detector") or {}).get("non_target_policy", "keep"))
    name_to_id, id_to_name, _roles = master_cat_maps(master)
    keep_cat_ids = {
        cid for name, cid in name_to_id.items()
        if name not in exclude
        and not (non_policy == "drop" and name in ("static_aid", "unknown_other"))
    }

    data_yaml = data_dir / "eo_views" / "yolo" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Missing {data_yaml}. Run scripts/data/export_eo_views.py first.")
    yolo_names = load_data_yaml_names(data_yaml)
    yolo_to_master = yolo_to_master_cat_ids(yolo_names, name_to_id)

    slices = build_slices(splits, operational)
    print("[eval] slices:")
    for n, ids in slices.items():
        print(f"  {n}: {len(ids)} images")

    jobs: list[tuple[str, Path, dict]] = []
    if args.family:
        match = [f for f in roster_families() if f.get("name") == args.family]
        if not match:
            ap.error(f"unknown family {args.family!r}")
        cfg_path = (REPO_ROOT / match[0]["config"]).resolve()
        jobs.append((args.family, cfg_path, load_family_config(cfg_path)))
    else:
        for f in roster_families():
            cfg_path = (REPO_ROOT / f["config"]).resolve()
            jobs.append((f["name"], cfg_path, load_family_config(cfg_path)))

    args.out.mkdir(parents=True, exist_ok=True)
    family_results: dict[str, dict[str, Any]] = {}

    # All test ids (union of slices that matter for one predict pass).
    predict_ids = set().union(*slices.values())

    for fam, cfg_path, cfg in jobs:
        det = cfg.get("detector") or {}
        arch = str(det.get("arch", "yolo11"))
        family_results[fam] = {}
        print(f"\n[eval] === {fam} ({arch}) ===")

        if args.dry_run:
            try:
                w = resolve_weights(args.weights_root, fam)
                print(f"  weights={w}")
            except FileNotFoundError as e:
                print(f"  weights MISSING: {e}")
            for sz in args.imgsz:
                family_results[fam][f"s{sz}"] = {
                    "status": "dry_run",
                    "slices": {n: {"n_images": len(ids), "mAP_50_95": None,
                                   "mAP_50": None, "n_gt": None}
                               for n, ids in slices.items()},
                }
            continue

        weights = resolve_weights(args.weights_root, fam)
        model = load_model(arch, weights)

        for sz in args.imgsz:
            det_path = args.out / f"{fam}_s{sz}_detections.json"
            t0 = time.perf_counter()
            if args.skip_predict and det_path.is_file():
                print(f"  imgsz={sz}: reusing {det_path}")
                detections = json.loads(det_path.read_text())
            else:
                print(f"  imgsz={sz}: predicting on {len(predict_ids)} images "
                      f"(device={device})…")
                detections = predict_coco_detections(
                    model, master, predict_ids,
                    data_dir=data_dir, imgsz=sz, device=device,
                    yolo_to_master=yolo_to_master, conf=args.conf,
                    batch=args.batch,
                )
                det_path.write_text(json.dumps(detections))
                print(f"  imgsz={sz}: {len(detections)} detections "
                      f"({time.perf_counter()-t0:.1f}s) → {det_path}")

            slice_metrics = evaluate_slices(
                master, detections, slices,
                keep_cat_ids=keep_cat_ids, det_min_side=det_min,
                id_to_name=id_to_name,
            )
            shore = slice_metrics.get("shore_operational") or {}
            print(f"  imgsz={sz}: shore mAP@[.5:.95]={fmt_ap(shore.get('mAP_50_95'))} "
                  f"mAP@50={fmt_ap(shore.get('mAP_50'))}")

            payload = {
                "family": fam,
                "arch": arch,
                "weights": str(weights),
                "imgsz": sz,
                "device": device,
                "det_min_side": det_min,
                "operational_sources": operational,
                "n_detections": len(detections),
                "slices": slice_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_json = args.out / f"{fam}_s{sz}.json"
            out_json.write_text(json.dumps(payload, indent=2) + "\n")
            family_results[fam][f"s{sz}"] = payload
            print(f"  wrote {out_json}")

    report = args.weights_root / "report.md"
    write_report(
        report,
        family_results=family_results,
        operational_sources=operational,
        imgsizes=list(args.imgsz),
    )
    summary = args.out / "clean_map_summary.json"
    summary.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "imgsz": list(args.imgsz),
        "families": family_results,
    }, indent=2) + "\n")
    print(f"\n[eval] report → {report}")
    print(f"[eval] summary → {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
