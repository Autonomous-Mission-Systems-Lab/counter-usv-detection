"""Evaluation harness: clean-mAP, attack×defense matrix, adaptive cost curves,
per-class discriminability, time-to-flag, and real-track FAR validation.
"""

from counterusv.eval.coco_map import (
    COCO_AREA_BINS,
    SHORTEST_SIDE_BINS,
    coco_eval_summary,
    filter_detections,
    filter_master,
    master_cat_maps,
    yolo_to_master_cat_ids,
)

__all__ = [
    "COCO_AREA_BINS",
    "SHORTEST_SIDE_BINS",
    "coco_eval_summary",
    "filter_detections",
    "filter_master",
    "master_cat_maps",
    "yolo_to_master_cat_ids",
]
