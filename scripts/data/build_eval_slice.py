#!/usr/bin/env python3
"""Operational small-craft evaluation slice over the EO COCO master .

This is an **evaluation slice, not a new dataset**: a versioned manifest that *selects
and tags* existing annotations/images in the COCO master (via the EO audit manifests).
No images are copied or duplicated. It re-runs cheaply over the audit outputs, so when
the curated `usv` set lands and the EO audit is re-run, this slice updates too.

Purpose (from the plan)
-----------------------
Isolate the **benign small craft a hostile USV plausibly mimics** — canonical classes
`small_craft`, `recreational`, `fishing`, and relevant `sailing` — so they are
explicitly represented in **false-alarm and transfer reporting**. The slice is
**benign-only**: these scene sources contain **no real hostile small USVs**, and the
McShips/ABOShips *military* examples are **large vessels** — they must NOT be presented
as size-matched hostile craft. The small hostile platform is supplied separately by the
curated USV set, for the EO baseline only.

Viewpoint stratification (load-bearing)
---------------------------------------
Most small-craft instances come from **ABOShips**, whose camera is on a *moving
watercraft* → "detection **by** a vessel," the viewpoint this project distinguishes
*against*. The operational, shore-based near-waterline instances are the minority
(SeaShips, SMD on-shore). The slice tags every row with `viewpoint` +
`operational_viewpoint` so reporting can stratify and not over-credit the wrong
viewpoint. (Viewpoint assessment: see `data/INVENTORY.md`.)

Outputs
-------
* ``data/eval_slices/small_craft_eval_annotations.csv`` — per-instance manifest
  (class, size, eligibility, viewpoint, image context, leakage keys).
* ``data/eval_slices/small_craft_eval_images.csv`` — per-image manifest (images with
  >=1 slice instance; per-class counts, viewpoint, chip-like, sequence/dedup keys).
* ``data/eval_slices/small_craft_eval_summary.json`` — counts + slice version.
* ``results/eval_slice/report.md`` (+ figures) — human report with the honest caveats.

Usage
-----
    python scripts/data/build_eval_slice.py
    python scripts/data/build_eval_slice.py --classes small_craft recreational fishing sailing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results" / "eval_slice"

SLICE_CLASSES = ["small_craft", "recreational", "fishing", "sailing"]

# Per-source viewpoint + whether it is the operational (shore near-waterline,
# "detection OF a USV") viewpoint. From data/INVENTORY.md.
VIEWPOINT = {
    "seaships": ("shore_near_waterline", True), # deployed coastline cameras — best match
    "smd": ("shore_fixed_platform", True), # fixed on-shore platform
    "mcships": ("web_in_the_wild", False), # mixed web imagery
    "aboships": ("onboard_moving_vessel", False), # detection BY a vessel — distinguished against
}

# Target-size bins on raw min-side px (context tag; not a filter).
SIZE_BINS = [0, 8, 32, 96, 1e9]
SIZE_LABELS = ["tiny_<8", "small_8-32", "med_32-96", "large_>=96"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _md5(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def viewpoint_of(src: str):
    return VIEWPOINT.get(src, ("unknown", False))


def build(ann_csv: Path, img_csv: Path, classes: list[str]):
    ann = pd.read_csv(ann_csv)
    img = pd.read_csv(img_csv)

    sl = ann[ann["category"].isin(classes)].copy()
    if sl.empty:
        raise SystemExit(f"[error] no annotations in classes {classes}")

    sl["viewpoint"] = sl["source"].map(lambda s: viewpoint_of(s)[0])
    sl["operational_viewpoint"] = sl["source"].map(lambda s: viewpoint_of(s)[1])
    sl["size_bin"] = pd.cut(sl["min_side_px"], SIZE_BINS, labels=SIZE_LABELS,
                            right=False)

    # join per-image context (viewpoint reporting + leakage keys for leakage-controlled splits)
    img_cols = ["image_id", "file_name", "orig_split", "width", "height",
                "sequence_id", "is_sequence_source", "chip_like",
                "exact_dup_group", "near_dup_group"]
    img_cols = [c for c in img_cols if c in img.columns]
    sl = sl.merge(img[img_cols], on="image_id", how="left")

    ann_out_cols = ["ann_id", "image_id", "source", "viewpoint",
                    "operational_viewpoint", "orig_label", "category", "role",
                    "w_px", "h_px", "min_side_px", "area_frac", "center_offset",
                    "size_bin", "detector_eligible", "patch_eligible",
                    "file_name", "orig_split", "sequence_id", "chip_like",
                    "exact_dup_group", "near_dup_group"]
    ann_out_cols = [c for c in ann_out_cols if c in sl.columns]
    ann_slice = sl[ann_out_cols].sort_values(["source", "category", "image_id"])

    # per-image manifest: images that contain >=1 slice instance
    grp = sl.groupby("image_id")
    per_img = grp.agg(
        source=("source", "first"), viewpoint=("viewpoint", "first"),
        operational_viewpoint=("operational_viewpoint", "first"),
        file_name=("file_name", "first"), orig_split=("orig_split", "first"),
        n_slice_objects=("ann_id", "count"),
        n_detector_eligible=("detector_eligible", "sum"),
        n_patch_eligible=("patch_eligible", "sum"),
        classes=("category", lambda s: ",".join(sorted(set(s)))),
        min_side_px_min=("min_side_px", "min"),
    )
    for c in ["sequence_id", "chip_like", "exact_dup_group", "near_dup_group"]:
        if c in sl.columns:
            per_img[c] = grp[c].first()
    per_img = per_img.reset_index().sort_values(["source", "image_id"])
    return ann_slice, per_img


def build_summary(ann_slice, per_img, classes, versions):
    def ct(df, a, b):
        return json.loads(pd.crosstab(df[a], df[b]).to_json())

    by_cls_src = pd.crosstab(ann_slice["category"], ann_slice["source"])
    by_cls_size = pd.crosstab(ann_slice["category"], ann_slice["size_bin"])
    op = ann_slice[ann_slice["operational_viewpoint"]]
    return {
        "slice_version": versions,
        "classes": classes,
        "instances": {
            "total": int(len(ann_slice)),
            "images": int(per_img["image_id"].nunique()),
            "detector_eligible": int(ann_slice["detector_eligible"].sum()),
            "patch_eligible": int(ann_slice["patch_eligible"].sum()),
        },
        "operational_viewpoint": {
            "operational_instances": int(len(op)),
            "operational_images": int(op["image_id"].nunique()),
            "operational_frac": round(len(op) / max(len(ann_slice), 1), 3),
            "by_source": json.loads(
                ann_slice.groupby("source")
                .agg(instances=("ann_id", "count"),
                     operational=("operational_viewpoint", "first")).to_json(orient="index")),
        },
        "per_class_x_source": json.loads(by_cls_src.to_json()),
        "per_class_x_size": json.loads(by_cls_size.to_json()),
        "leakage_keys_present": {
            "sequences": int(ann_slice["sequence_id"].notna().sum())
            if "sequence_id" in ann_slice else 0,
            "exact_dup_rows": int((ann_slice["exact_dup_group"] >= 0).sum())
            if "exact_dup_group" in ann_slice else 0,
            "near_dup_rows": int((ann_slice["near_dup_group"] >= 0).sum())
            if "near_dup_group" in ann_slice else 0,
        },
        "notes": [
            "Benign-only slice: NO real hostile small USVs in these sources. "
            "McShips/ABOShips military examples are LARGE vessels and are excluded here; "
            "they must not be presented as size-matched hostile craft.",
            "The small hostile platform (canonical `usv`, role hostile) is supplied "
            "separately by the curated USV set, for the EO baseline only.",
            "Most instances are ABOShips (onboard/moving-vessel viewpoint = detection BY "
            "a vessel). Report operational (shore) and supplementary viewpoints separately "
            "so the wrong viewpoint is not over-credited.",
            "sequence_id + dedup groups are carried through as leakage-control inputs for "
            "the split; the slice does not itself define train/test.",
        ],
    }


def make_figures(ann_slice, fig_dir):
    import os
    fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(fig_dir.parent / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct = pd.crosstab(ann_slice["category"], ann_slice["viewpoint"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("instances"); ax.set_title("Small-craft eval slice: class x viewpoint")
    ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "class_by_viewpoint.png", dpi=120); plt.close(fig)

    ct2 = pd.crosstab(ann_slice["category"], ann_slice["size_bin"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ct2.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("instances"); ax.set_title("Small-craft eval slice: class x target size")
    ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "class_by_size.png", dpi=120); plt.close(fig)


def write_report(summary, path):
    s = summary; inst = s["instances"]; op = s["operational_viewpoint"]
    cxs = pd.DataFrame(s["per_class_x_source"]).fillna(0).astype(int)
    csz = pd.DataFrame(s["per_class_x_size"]).fillna(0).astype(int)
    L = ["# Operational small-craft evaluation slice \n",
         "Auto-generated by `scripts/data/build_eval_slice.py`. A **versioned manifest over the "
         "existing COCO master** (via the EO audit) — not a new dataset. It selects the "
         "**benign small craft a hostile USV plausibly mimics** for false-alarm / transfer "
         "reporting.\n",
         f"**Slice version:** `{s['slice_version']['version']}` "
         f"(git `{s['slice_version']['git_sha']}`, {s['slice_version']['date']}); "
         f"inputs md5 ann=`{s['slice_version']['ann_csv_md5']}` "
         f"img=`{s['slice_version']['img_csv_md5']}`\n",
         f"**Classes:** {', '.join(s['classes'])}\n",
         "## Totals\n",
         f"- Instances: **{inst['total']:,}** across **{inst['images']:,}** images\n",
         f"- Detector-eligible (>=8px): **{inst['detector_eligible']:,}** · "
         f"patch-eligible (>=32px): **{inst['patch_eligible']:,}**\n",
         "## Viewpoint stratification (load-bearing)\n",
         f"- **Operational (shore near-waterline) instances: {op['operational_instances']:,} "
         f"= {100*op['operational_frac']:.0f}%** of the slice, in {op['operational_images']:,} "
         "images. The remainder is the ABOShips **onboard/moving-vessel** viewpoint "
         "(detection *by* a vessel) — report it separately, do not over-credit it.\n",
         "\n### Instances by class x source\n",
         "| class | " + " | ".join(cxs.columns) + " | total |",
         "|---|" + "|".join(["---"] * len(cxs.columns)) + "|---|"]
    for cls, row in cxs.iterrows():
        L.append(f"| {cls} | " + " | ".join(str(int(v)) for v in row.values)
                 + f" | {int(row.sum())} |")
    L += ["\n### Instances by class x target size (raw min-side px)\n",
          "| class | " + " | ".join(csz.columns) + " |",
          "|---|" + "|".join(["---"] * len(csz.columns)) + "|"]
    for cls, row in csz.iterrows():
        L.append(f"| {cls} | " + " | ".join(str(int(v)) for v in row.values) + " |")
    L += ["\n## Caveats (non-negotiable)\n"]
    for n in s["notes"]:
        L.append(f"- {n}\n")
    L += ["\n## Figures\n",
          "- `figures/class_by_viewpoint.png` · `figures/class_by_size.png`\n"]
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Small-craft eval slice .")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--classes", nargs="+", default=SLICE_CLASSES)
    args = p.parse_args(argv)

    ann_csv = args.data_dir / "audit" / "eo_annotations.csv"
    img_csv = args.data_dir / "audit" / "eo_images.csv"
    if not ann_csv.exists() or not img_csv.exists():
        print(f"[error] run scripts/data/audit_eo.py first (missing {ann_csv} / {img_csv})")
        return 1

    ann_slice, per_img = build(ann_csv, img_csv, args.classes)
    versions = {
        "version": f"{date.today().isoformat()}+{_git_sha()}",
        "date": date.today().isoformat(), "git_sha": _git_sha(),
        "ann_csv_md5": _md5(ann_csv), "img_csv_md5": _md5(img_csv),
    }
    summary = build_summary(ann_slice, per_img, args.classes, versions)

    out_dir = args.data_dir / "eval_slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_slice.to_csv(out_dir / "small_craft_eval_annotations.csv", index=False)
    per_img.to_csv(out_dir / "small_craft_eval_images.csv", index=False)
    (out_dir / "small_craft_eval_summary.json").write_text(json.dumps(summary, indent=2))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        make_figures(ann_slice, args.results_dir / "figures")
    except Exception as e:
        print(f"[warn] figures failed: {e}")
    write_report(summary, args.results_dir / "report.md")

    inst = summary["instances"]; op = summary["operational_viewpoint"]
    print(f"[slice] {inst['total']:,} instances / {inst['images']:,} images "
          f"({', '.join(args.classes)})")
    print(f" detector-eligible {inst['detector_eligible']:,} · "
          f"patch-eligible {inst['patch_eligible']:,}")
    print(f" operational (shore) viewpoint: {op['operational_instances']:,} "
          f"({100*op['operational_frac']:.0f}%) — rest is onboard/moving-vessel (ABOShips)")
    print(f" version {summary['slice_version']['version']}")
    print(f"\nWrote:\n {out_dir}/small_craft_eval_annotations.csv"
          f"\n {out_dir}/small_craft_eval_images.csv"
          f"\n {out_dir}/small_craft_eval_summary.json"
          f"\n {args.results_dir}/report.md + figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
