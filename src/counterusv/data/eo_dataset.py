"""Harmonized EO detection dataset (COCO master + leakage-controlled splits).

Honors ``data/HARMONIZATION.md`` at load time:
  * native-on-disk pixels; letterbox to a fixed square input with centered pad 114
  * native→input box remap; clip; drop boxes below the detector-eligibility floor
  * SeaShips fixed-region overlay mask (non-destructive)
  * per-channel normalization
  * augmentation OFF for clean eval (letterbox only); train-mode mosaic/multiscale
    are left to the detector framework (Ultralytics etc.) when using the YOLO view

Does **not** re-split: consumes ``data/splits/eo_image_splits.csv`` as-is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from counterusv.data.letterbox import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_PAD_VALUE,
    LetterboxMeta,
    letterbox_image,
    remap_boxes,
)
from counterusv.data.overlay import mask_seaships_overlay

# ImageNet stats for pretrained backbones; otherwise leave [0,1].
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EligibilityStats:
    """Retained vs dropped boxes under the detector-eligibility floor."""

    boxes_total: int = 0
    boxes_kept: int = 0
    boxes_dropped_native: int = 0
    boxes_dropped_letterbox: int = 0
    images: int = 0
    images_with_zero_kept: int = 0

    @property
    def keep_rate(self) -> float:
        return self.boxes_kept / self.boxes_total if self.boxes_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "boxes_total": self.boxes_total,
            "boxes_kept": self.boxes_kept,
            "boxes_dropped_native": self.boxes_dropped_native,
            "boxes_dropped_letterbox": self.boxes_dropped_letterbox,
            "keep_rate": round(self.keep_rate, 6),
            "images": self.images,
            "images_with_zero_kept": self.images_with_zero_kept,
        }


@dataclass
class EODatasetConfig:
    data_dir: Path
    master_json: Path
    splits_csv: Path
    split: str  # train | val | test
    input_size: int = DEFAULT_INPUT_SIZE
    letterbox_pad: int = DEFAULT_PAD_VALUE
    det_min_side: float = 8.0
    mask_overlay_sources: Sequence[str] = ("seaships",)
    # When True: mosaic/multiscale expected from the framework; this Dataset still
    # only letterboxes. When False: clean eval — letterbox only, no photometric.
    augment: bool = False
    normalize: str = "imagenet"  # "imagenet" | "01" | "none"
    drop_non_target: bool = False  # decided finally in label-policy step
    non_target_names: Sequence[str] = ("static_aid", "unknown_other")
    sources: Sequence[str] | None = None  # optional source filter (e.g. shore slice)
    require_file: bool = True


class EODetectionDataset:
    """Indexable EO detection dataset over one split of the COCO master.

    ``__getitem__`` returns a dict::

        {
          "image": float32 CHW tensor-like ndarray in the chosen norm,
          "image_uint8": HxWx3 letterboxed RGB (for QA / viz),
          "boxes": (N,4) float64 COCO xywh in input coords,
          "labels": (N,) int64 canonical category ids (1-indexed, matching master),
          "label_names": list[str],
          "image_id": int,
          "source": str,
          "file_name": str,          # path relative to data_dir
          "meta": LetterboxMeta,
          "n_dropped_native": int,
          "n_dropped_letterbox": int,
        }
    """

    def __init__(self, cfg: EODatasetConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.master = json.loads(Path(cfg.master_json).read_text())
        self.cat_id_to_name = {c["id"]: c["name"] for c in self.master["categories"]}
        self.cat_name_to_id = {n: i for i, n in self.cat_id_to_name.items()}
        self.non_target_ids = {
            self.cat_name_to_id[n] for n in cfg.non_target_names
            if n in self.cat_name_to_id
        }

        splits = pd.read_csv(cfg.splits_csv)
        if "split" not in splits.columns or "image_id" not in splits.columns:
            raise ValueError(f"splits CSV missing required columns: {cfg.splits_csv}")
        sub = splits[splits["split"] == cfg.split]
        if cfg.sources is not None:
            sub = sub[sub["source"].isin(list(cfg.sources))]
        keep_ids = set(int(x) for x in sub["image_id"].tolist())

        id_to_img = {int(im["id"]): im for im in self.master["images"]}
        self.images: list[dict] = []
        for iid in sorted(keep_ids):
            im = id_to_img.get(iid)
            if im is None:
                continue
            if cfg.require_file and not (self.data_dir / im["file_name"]).is_file():
                continue
            self.images.append(im)

        anns_by_img: dict[int, list[dict]] = {int(im["id"]): [] for im in self.images}
        for a in self.master["annotations"]:
            iid = int(a["image_id"])
            if iid not in anns_by_img:
                continue
            if cfg.drop_non_target and int(a["category_id"]) in self.non_target_ids:
                continue
            anns_by_img[iid].append(a)
        self.anns_by_img = anns_by_img
        self._eligibility: EligibilityStats | None = None

    def __len__(self) -> int:
        return len(self.images)

    # ------------------------------------------------------------------ helpers
    def _load_rgb(self, file_name: str) -> np.ndarray:
        path = self.data_dir / file_name
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))

    def _normalize(self, image_uint8: np.ndarray) -> np.ndarray:
        x = image_uint8.astype(np.float32) / 255.0  # HWC [0,1]
        if self.cfg.normalize == "imagenet":
            mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
            std = np.asarray(IMAGENET_STD, dtype=np.float32)
            x = (x - mean) / std
        elif self.cfg.normalize == "01":
            pass
        elif self.cfg.normalize == "none":
            return image_uint8
        else:
            raise ValueError(f"unknown normalize={self.cfg.normalize!r}")
        return np.transpose(x, (2, 0, 1))  # CHW

    def _filter_native(self, anns: list[dict]) -> tuple[list[dict], int]:
        kept, dropped = [], 0
        floor = self.cfg.det_min_side
        for a in anns:
            x, y, w, h = a["bbox"]
            if min(w, h) < floor:
                dropped += 1
                continue
            kept.append(a)
        return kept, dropped

    # ------------------------------------------------------------------ public
    def __getitem__(self, idx: int) -> dict[str, Any]:
        im = self.images[idx]
        rgb = self._load_rgb(im["file_name"])
        source = im.get("source", "")
        if source in self.cfg.mask_overlay_sources:
            rgb = mask_seaships_overlay(rgb, value=self.cfg.letterbox_pad)

        letterboxed, meta = letterbox_image(
            rgb, input_size=self.cfg.input_size, pad_value=self.cfg.letterbox_pad)

        anns, n_drop_native = self._filter_native(self.anns_by_img[int(im["id"])])
        native_boxes = [a["bbox"] for a in anns]
        labels = [int(a["category_id"]) for a in anns]

        # Remap then drop any that shrink below the floor post-letterbox.
        if native_boxes:
            remapped_all = remap_boxes(native_boxes, meta, clip=True, min_side=0.0)
            keep = np.minimum(remapped_all[:, 2], remapped_all[:, 3]) >= self.cfg.det_min_side
            n_drop_lb = int((~keep).sum())
            boxes = remapped_all[keep]
            labels_arr = np.asarray(labels, dtype=np.int64)[keep]
        else:
            boxes = np.zeros((0, 4), dtype=np.float64)
            labels_arr = np.zeros((0,), dtype=np.int64)
            n_drop_lb = 0

        names = [self.cat_id_to_name.get(int(c), str(c)) for c in labels_arr]
        return {
            "image": self._normalize(letterboxed),
            "image_uint8": letterboxed,
            "boxes": boxes,
            "labels": labels_arr,
            "label_names": names,
            "image_id": int(im["id"]),
            "source": source,
            "file_name": im["file_name"],
            "meta": meta,
            "n_dropped_native": n_drop_native,
            "n_dropped_letterbox": n_drop_lb,
        }

    def eligibility_stats(self, max_images: int | None = None) -> EligibilityStats:
        """Sweep the split and report retained vs dropped boxes under the floor."""
        stats = EligibilityStats()
        n = len(self) if max_images is None else min(len(self), max_images)
        for i in range(n):
            item = self[i]
            native_n = len(self.anns_by_img[item["image_id"]])
            if self.cfg.drop_non_target:
                native_n = sum(
                    1 for a in self.anns_by_img[item["image_id"]]
                    if int(a["category_id"]) not in self.non_target_ids
                )
            kept = len(item["boxes"])
            stats.images += 1
            stats.boxes_total += native_n
            stats.boxes_kept += kept
            stats.boxes_dropped_native += item["n_dropped_native"]
            stats.boxes_dropped_letterbox += item["n_dropped_letterbox"]
            if kept == 0 and native_n > 0:
                stats.images_with_zero_kept += 1
        self._eligibility = stats
        return stats

    def iter_records(self) -> Iterable[dict[str, Any]]:
        for i in range(len(self)):
            yield self[i]


def default_config(
    data_dir: Path | str,
    split: str,
    *,
    input_size: int = 640,
    augment: bool = False,
    sources: Sequence[str] | None = None,
) -> EODatasetConfig:
    data_dir = Path(data_dir)
    return EODatasetConfig(
        data_dir=data_dir,
        master_json=data_dir / "annotations" / "coco_master.json",
        splits_csv=data_dir / "splits" / "eo_image_splits.csv",
        split=split,
        input_size=input_size,
        augment=augment,
        sources=sources,
    )
