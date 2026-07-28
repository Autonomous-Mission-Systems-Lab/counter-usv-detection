#!/usr/bin/env python3
"""USV-recognition capability check for the detector baselines.

Confirms the baselines are **USV-recognition-capable** (the ``docs/METRICS.md``
EO-baseline requirement): on the undisguised-``usv`` test slice it reports each
family's capability **bracketed** between a weak floor and a perfect ceiling:

  * **Presence floor (any-class recall)** — fraction of ``usv`` ground-truth
    boxes localized by *any* detection (IoU ≥ 0.5), regardless of the predicted
    label. This is the EO analogue of the presence-only cross-check: "a contact
    is there" without recognizing it as a USV.
  * **Recognition (class-correct recall + AP)** — fraction localized by a
    detection actually labelled ``usv`` (the real EO-baseline capability), plus
    COCO ``usv`` AP@[.5:.95] / AP@50.
  * **Perfect-EO oracle ceiling = 1.0** — the contact is *told* to carry its
    class (supplied only at eval); the recognition number sits below this by
    construction.

Reuses the cached clean-eval detections (master category ids) so no re-inference
is needed; run ``eval_detector_clean.py`` first.

Usage
-----
    python scripts/detector/usv_capability.py --all
    python scripts/detector/usv_capability.py --all --conf 0.25 --iou 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "detector"))

from counterusv.eval.coco_map import (  # noqa: E402
    coco_eval_summary,
    filter_detections,
    filter_master,
    master_cat_maps,
)
from train_detector import roster_families  # noqa: E402

DEFAULT_WEIGHTS_ROOT = REPO_ROOT / "results" / "detector_baselines"
DEFAULT_CLEAN_MAP = DEFAULT_WEIGHTS_ROOT / "clean_map"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def recall_at_operating_point(
    gt: dict,
    detections: list[dict],
    *,
    usv_cat_id: int,
    conf: float,
    iou_thr: float,
) -> dict[str, Any]:
    """Greedy IoU matching → presence (any-class) vs class-correct recall.

    Both use score ≥ ``conf`` detections and a per-image greedy match (highest
    score first) to unmatched GT at IoU ≥ ``iou_thr``.
    """
    gt_by_img: dict[int, list[list[float]]] = {}
    for a in gt["annotations"]:
        gt_by_img.setdefault(int(a["image_id"]), []).append(list(a["bbox"]))
    det_by_img: dict[int, list[dict]] = {}
    for d in detections:
        if float(d["score"]) < conf:
            continue
        det_by_img.setdefault(int(d["image_id"]), []).append(d)

    total_gt = sum(len(v) for v in gt_by_img.values())
    matched_any = 0
    matched_usv = 0
    usv_tp = 0
    usv_det_total = 0

    for img_id, boxes in gt_by_img.items():
        dets = sorted(det_by_img.get(img_id, []), key=lambda d: -float(d["score"]))
        usv_det_total += sum(1 for d in dets if int(d["category_id"]) == usv_cat_id)
        used_any = [False] * len(boxes)
        used_usv = [False] * len(boxes)
        for d in dets:
            db = list(d["bbox"])
            is_usv = int(d["category_id"]) == usv_cat_id
            best_j, best_iou = -1, iou_thr
            for j, gb in enumerate(boxes):
                if used_any[j]:
                    continue
                v = _iou_xywh(db, gb)
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used_any[best_j] = True
            # class-correct pass (independent greedy over usv-labelled dets)
            if is_usv:
                bj, bi = -1, iou_thr
                for j, gb in enumerate(boxes):
                    if used_usv[j]:
                        continue
                    v = _iou_xywh(db, gb)
                    if v >= bi:
                        bi, bj = v, j
                if bj >= 0:
                    used_usv[bj] = True
                    usv_tp += 1
        matched_any += sum(used_any)
        matched_usv += sum(used_usv)

    return {
        "conf": conf,
        "iou_thr": iou_thr,
        "n_gt": total_gt,
        "presence_recall_any_class": (matched_any / total_gt) if total_gt else None,
        "recognition_recall_usv_class": (matched_usv / total_gt) if total_gt else None,
        "usv_precision": (usv_tp / usv_det_total) if usv_det_total else None,
        "n_usv_detections": usv_det_total,
    }


def fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def build_usv_slice(splits: pd.DataFrame) -> set[int]:
    test = splits.loc[splits["split"] == "test"]
    return set(int(x) for x in test.loc[test["source"] == "usv", "image_id"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--family", type=str)
    ap.add_argument("--clean-map", type=Path, default=DEFAULT_CLEAN_MAP)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="which cached detections to score (default 640)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="operating confidence for recall/precision (default 0.25)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="match IoU for recall/precision (default 0.5)")
    args = ap.parse_args()

    data_dir = args.data_dir
    master = json.loads((data_dir / "annotations" / "coco_master.json").read_text())
    splits = pd.read_csv(data_dir / "splits" / "eo_image_splits.csv")
    base = _load_yaml(REPO_ROOT / "configs" / "base.yaml")
    det_min = float((base.get("data") or {}).get("det_min_pixels_on_target", 8))

    name_to_id, id_to_name, _roles = master_cat_maps(master)
    usv_cat_id = name_to_id["usv"]

    usv_ids = build_usv_slice(splits)
    gt = filter_master(
        master, usv_ids, keep_cat_ids={usv_cat_id}, det_min_side=det_min)
    n_gt = len(gt["annotations"])
    print(f"[usv] slice: {len(usv_ids)} test images, {n_gt} usv GT boxes "
          f"(≥{det_min:.0f}px)")

    if args.family:
        fams = [args.family]
    else:
        fams = [f["name"] for f in roster_families()]

    role_by_fam = {
        f["name"]: f.get("role", f.get("transfer_role", "?"))
        for f in roster_families()
    }

    results: dict[str, Any] = {}
    for fam in fams:
        det_path = args.clean_map / f"{fam}_s{args.imgsz}_detections.json"
        if not det_path.is_file():
            print(f"[usv] {fam}: MISSING {det_path.name} — run eval_detector_clean.py")
            continue
        all_dets = json.loads(det_path.read_text())
        slice_dets = filter_detections(all_dets, usv_ids)
        usv_only = [d for d in slice_dets if int(d["category_id"]) == usv_cat_id]

        ap_summary = coco_eval_summary(gt, usv_only, id_to_name=id_to_name)
        op = recall_at_operating_point(
            gt, slice_dets, usv_cat_id=usv_cat_id,
            conf=args.conf, iou_thr=args.iou)

        results[fam] = {
            "transfer_role": role_by_fam.get(fam, "?"),
            "imgsz": args.imgsz,
            "ap_50_95": ap_summary.get("mAP_50_95"),
            "ap_50": ap_summary.get("mAP_50"),
            "ar_100": ap_summary.get("AR_100"),
            "operating_point": op,
        }
        print(f"[usv] {fam:9s} AP@[.5:.95]={fmt(ap_summary.get('mAP_50_95'))} "
              f"AP@50={fmt(ap_summary.get('mAP_50'))} | "
              f"presence={fmt(op['presence_recall_any_class'], pct=True)} "
              f"recognition={fmt(op['recognition_recall_usv_class'], pct=True)}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "slice": {"n_images": len(usv_ids), "n_usv_gt": n_gt,
                  "det_min_side": det_min},
        "oracle_ceiling": 1.0,
        "families": results,
    }
    out_json = args.clean_map / "usv_capability.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    write_summary(DEFAULT_WEIGHTS_ROOT / "usv_capability.md", payload)
    print(f"\n[usv] json    → {out_json}")
    print(f"[usv] summary → {DEFAULT_WEIGHTS_ROOT / 'usv_capability.md'}")
    return 0


def write_summary(out_path: Path, payload: dict) -> None:
    sl = payload["slice"]
    lines: list[str] = []
    lines.append("# USV-recognition capability — detector baselines")
    lines.append("")
    lines.append(f"Generated: {payload['timestamp']}")
    lines.append("")
    lines.append(
        "Confirms the EO baselines can recognize **undisguised** USVs "
        "(the EO-baseline requirement in `docs/METRICS.md`), so the later "
        "EO-alone vs. EO+consistency comparison is fair. Measured on the "
        f"undisguised-`usv` test slice (**{sl['n_images']} images / "
        f"{sl['n_usv_gt']} boxes**, native shortest side ≥ "
        f"{sl['det_min_side']:.0f}px), at the 640 training resolution.")
    lines.append("")
    lines.append(
        f"Capability is **bracketed**: presence floor (any-class localization) "
        f"≤ recognition (`usv`-labelled) ≤ perfect-EO oracle "
        f"(**{payload['oracle_ceiling']:.1f}**, the contact is told its class at "
        f"eval). Operating point: conf ≥ {payload['conf']}, IoU ≥ {payload['iou']}.")
    lines.append("")
    lines.append("| family | role | AP@[.5:.95] | AP@50 | presence recall | recognition recall | `usv` precision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for fam, r in payload["families"].items():
        op = r["operating_point"]
        lines.append(
            f"| {fam} | {r['transfer_role']} | {fmt(r['ap_50_95'])} | "
            f"{fmt(r['ap_50'])} | {fmt(op['presence_recall_any_class'], pct=True)} | "
            f"{fmt(op['recognition_recall_usv_class'], pct=True)} | "
            f"{fmt(op['usv_precision'], pct=True)} |")
    lines.append("")
    lines.append("## Reading the bracket")
    lines.append("")
    lines.append(
        "- **Presence recall (floor)** — any detection localizes the USV "
        "(IoU ≥ 0.5), regardless of label. The EO analogue of the presence-only "
        "cross-check: a contact is seen but not recognized.")
    lines.append(
        "- **Recognition recall** — a detection labelled `usv` localizes it. "
        "This is the fielded EO-baseline capability the headline comparison "
        "credits to the EO-alone system.")
    lines.append(
        "- **Perfect-EO oracle (ceiling = 1.0)** — the contact is told its class "
        "at eval; supplied later as the upper bound so the improvement claim "
        "does not hinge on baseline strength.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        f"- **Thin slice** — {sl['n_images']} `usv` test images / "
        f"{sl['n_usv_gt']} boxes (val is equally thin). Treat per-family "
        "differences as capability confirmation, not a precise ranking.")
    lines.append(
        "- **Viewpoint/size bias** — the curated `usv` set is shore-viewpoint, "
        "small-craft-scale, and license/provenance-tracked; present it as a "
        "*representative approximation* of a fielded USV set, not a reproduction "
        "(see `docs/THREAT_MODEL.md` EO-baseline note).")
    lines.append(
        "- **Firewall** — `usv` imagery trains only the detector; it is never "
        "seen by the class–kinematics scorer.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
