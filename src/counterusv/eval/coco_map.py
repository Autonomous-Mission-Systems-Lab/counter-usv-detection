"""COCO-style mAP helpers for clean detector evaluation.

Filters the COCO master to a set of images / eligibility floors, runs
``pycocotools.COCOeval``, and extracts headline + per-class + size-bin metrics.
Prediction ``category_id``s must be **master** (1-indexed taxonomy) ids, not
YOLO 0-indexed class indices.
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# COCO area bins (pixel^2).
COCO_AREA_BINS: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e10),
}

# Shortest-side bins aligned with the harmonization eligibility floors.
SHORTEST_SIDE_BINS: dict[str, tuple[float, float]] = {
    "side_8_32": (8.0, 32.0),    # detector-eligible but below patch floor
    "side_32_96": (32.0, 96.0),  # patch-eligible mid
    "side_96_plus": (96.0, 1e10),
}


def master_cat_maps(master: dict) -> tuple[dict[str, int], dict[int, str], dict[int, str]]:
    """Return (name→id, id→name, id→role) from the COCO master categories."""
    name_to_id = {c["name"]: int(c["id"]) for c in master["categories"]}
    id_to_name = {int(c["id"]): c["name"] for c in master["categories"]}
    id_to_role = {int(c["id"]): str(c.get("role", "benign")) for c in master["categories"]}
    return name_to_id, id_to_name, id_to_role


def yolo_to_master_cat_ids(data_yaml_names: dict[int, str], name_to_id: dict[str, int]) -> dict[int, int]:
    """Map YOLO 0-indexed class index → master category_id."""
    out: dict[int, int] = {}
    for yi, name in data_yaml_names.items():
        if name not in name_to_id:
            raise KeyError(f"YOLO class {yi}:{name!r} not in master categories")
        out[int(yi)] = name_to_id[name]
    return out


def filter_master(
    master: dict,
    image_ids: Iterable[int],
    *,
    keep_cat_ids: set[int] | None = None,
    det_min_side: float = 8.0,
    min_area: float | None = None,
    max_area: float | None = None,
    min_side: float | None = None,
    max_side: float | None = None,
) -> dict:
    """Build a COCO-format dict restricted to ``image_ids`` + eligibility filters.

    Box filters apply to GT annotations only (native coords). ``det_min_side`` is
    the detector-eligibility floor (native shortest side); size-bin filters are
    additional optional constraints.
    """
    ids = set(int(i) for i in image_ids)
    images = [im for im in master["images"] if int(im["id"]) in ids]
    keep_imgs = {int(im["id"]) for im in images}
    anns = []
    for a in master["annotations"]:
        if int(a["image_id"]) not in keep_imgs:
            continue
        cid = int(a["category_id"])
        if keep_cat_ids is not None and cid not in keep_cat_ids:
            continue
        x, y, w, h = a["bbox"]
        side = min(float(w), float(h))
        area = float(a.get("area", w * h))
        if side < det_min_side:
            continue
        if min_side is not None and side < min_side:
            continue
        if max_side is not None and side >= max_side:
            continue
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area >= max_area:
            continue
        anns.append(a)
    cats = list(master["categories"])
    if keep_cat_ids is not None:
        cats = [c for c in cats if int(c["id"]) in keep_cat_ids]
    return {
        "info": master.get("info") or {},
        "images": images,
        "annotations": anns,
        "categories": cats,
    }


def coco_eval_summary(
    gt: dict,
    detections: Sequence[dict],
    *,
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Run ``pycocotools`` COCOeval; return headline + per-class APs.

    ``detections`` is a list of COCO-format result dicts
    (``image_id``, ``category_id``, ``bbox`` [x,y,w,h], ``score``).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not gt["images"]:
        return {
            "n_images": 0, "n_gt": 0, "n_det": 0,
            "mAP_50_95": None, "mAP_50": None, "per_class": {},
            "note": "empty image set",
        }
    if not gt["annotations"]:
        return {
            "n_images": len(gt["images"]), "n_gt": 0, "n_det": len(detections),
            "mAP_50_95": None, "mAP_50": None, "per_class": {},
            "note": "no ground-truth boxes after filters",
        }

    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        dt_path = Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(gt))
        # pycocotools loadRes needs a non-empty list; empty → skip.
        if not detections:
            return {
                "n_images": len(gt["images"]), "n_gt": len(gt["annotations"]),
                "n_det": 0, "mAP_50_95": 0.0, "mAP_50": 0.0, "per_class": {},
                "note": "no detections",
            }
        dt_path.write_text(json.dumps(list(detections)))
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(dt_path))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        stats = [float(x) for x in ev.stats]
        per_class: dict[str, dict[str, float | None]] = {}
        # Per-category AP at IoU=.50:.95 and IoU=.50 from precision tensor.
        # precision shape: [T, R, K, A, M] — T=IoU thr, R=recall, K=cat, A=area, M=maxDet
        precision = ev.eval.get("precision")  # may be None if no GT cats matched
        cat_ids = ev.params.catIds
        if precision is not None and id_to_name is not None:
            # area index 0 = all; maxDet index -1 = 100
            for ki, cid in enumerate(cat_ids):
                name = id_to_name.get(int(cid), str(cid))
                # AP@[.5:.95] = mean over IoU thresholds where precision > -1
                p_all = precision[:, :, ki, 0, -1]
                valid = p_all[p_all > -1]
                ap = float(valid.mean()) if valid.size else None
                # AP@50 = IoU index 0
                p50 = precision[0, :, ki, 0, -1]
                valid50 = p50[p50 > -1]
                ap50 = float(valid50.mean()) if valid50.size else None
                per_class[name] = {"AP_50_95": ap, "AP_50": ap50}

    return {
        "n_images": len(gt["images"]),
        "n_gt": len(gt["annotations"]),
        "n_det": len(detections),
        "mAP_50_95": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "mAP_small": stats[3],
        "mAP_medium": stats[4],
        "mAP_large": stats[5],
        "AR_100": stats[8],
        "per_class": per_class,
    }


def group_detections_by_image(detections: Sequence[dict]) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for d in detections:
        by[int(d["image_id"])].append(d)
    return by


def filter_detections(detections: Sequence[dict], image_ids: set[int]) -> list[dict]:
    return [d for d in detections if int(d["image_id"]) in image_ids]
