#!/usr/bin/env python3
"""EO context, quality, and pixels-on-target audit.

Source-agnostic: reads whatever is in the COCO master (``data/annotations/
coco_master.json``) and re-runs cheaply as new sources land — in particular the
curated ``usv`` set passes through this same audit on integration, with no
code change and no exemption from the size floor set here.

What it produces
----------------
* ``data/audit/eo_annotations.csv`` — per-annotation manifest: bbox pixels, area
  fraction, centering, expected post-letterbox target size (640 / 1280), and the
  non-destructive detector / adversarial-patch eligibility flags.
* ``data/audit/eo_images.csv`` — per-image manifest: object count, largest-object
  area fraction, chip-like context tag, sequence/video origin, file existence,
  exact-duplicate group, and perceptual near-duplicate group.
* ``data/audit/eo_audit_summary.json`` — aggregates, threshold sweeps, QA counts.
* ``results/eo_audit/report.md`` + ``results/eo_audit/figures/*.png`` — the human
  readable audit report and figures.

Design principles
----------------------------------
* **Non-destructive.** Nothing is deleted from the master. Low-resolution objects
  get *flags*, not removal; retained counts are reported under every policy.
* **Two separate eligibility policies.** A target may be detectable yet too small
  to carry a bounded adversarial patch, so detector-eligibility and
  patch-eligibility have separately justified floors, each reported across a
  threshold sweep (final floor is locked in the harmonization spec harmonization).
* **Provisional preprocessing assumptions.** Native files are preserved on disk;
  size is interpreted through a letterbox (scale by the long side) to candidate
  network inputs (640 primary; 1280 for small-target recall). Sources are never
  pre-resized to a common native resolution.

Usage
-----
    python scripts/data/audit_eo.py # full audit over the master
    python scripts/data/audit_eo.py --no-hash # skip dedup (context/size only)
    python scripts/data/audit_eo.py --sources smd # audit a subset of sources
    python scripts/data/audit_eo.py --jobs 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results" / "eo_audit"

# --- provisional thresholds (documented; final floor set in the harmonization spec) -------------
DET_MIN_SIDE = 8 # native px shortest side: detector-eligibility default
PATCH_MIN_SIDE = 32 # native px shortest side: patch-eligibility default
DET_SWEEP = [4, 8, 16, 24, 32]
PATCH_SWEEP = [16, 24, 32, 48, 64]
INPUT_SIZES = [640, 1280]
CHIP_AREA_FRAC = 0.40 # largest object fills >=40% of the frame ...
CHIP_CENTER = 0.25 # ... and is centered (normalized center offset <=0.25)
NEAR_HAM = 5 # dHash Hamming radius for perceptual near-duplicates


# ---------------------------------------------------------------------------
# Sequence / video origin (source-agnostic with per-source rules)
# ---------------------------------------------------------------------------
def sequence_id(source: str, orig_id: str) -> tuple[str, bool]:
    """Return (sequence_id, is_sequence_source).

    Frames from the same video/recording must land in the same split , so we
    derive a conservative grouping key. Non-sequence sources get a per-image
    singleton key. New sources (e.g. ``usv``) default to singletons unless a rule
    is added here.
    """
    if source == "smd": # "MVI_1448_VIS_Haze#0" -> video stem
        return f"smd/{str(orig_id).split('#', 1)[0]}", True
    if source == "aboships": # "201806260750_003" -> recording day
        return f"aboships/{str(orig_id)[:8]}", True
    return f"{source}/{orig_id}", False


# ---------------------------------------------------------------------------
# Per-image hashing worker (exact md5 + perceptual dHash + decoded dims)
# ---------------------------------------------------------------------------
def _dhash(gray: np.ndarray, size: int = 8) -> int:
    import cv2
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hash_one(args) -> dict:
    """Worker: returns md5, dhash, decoded (w,h) for one file. Robust to failures."""
    import cv2
    idx, abspath = args
    out = {"i": idx, "md5": None, "dhash": None, "dw": None, "dh": None, "ok": False}
    p = Path(abspath)
    if not p.exists():
        return out
    try:
        md5 = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                md5.update(chunk)
        out["md5"] = md5.hexdigest()
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out["dh"], out["dw"] = int(img.shape[0]), int(img.shape[1])
            out["dhash"] = _dhash(img)
            out["ok"] = True
    except Exception:
        pass
    return out


def compute_hashes(images: list[dict], data_dir: Path, jobs: int,
                   cache_path: Path) -> dict[int, dict]:
    """Hash every image (md5 + dhash + decoded dims), using a size+mtime cache so
    re-runs over an unchanged master are cheap."""
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    todo, results = [], {}
    for im in images:
        rel = im["file_name"]
        ap = data_dir / rel
        st = ap.stat() if ap.exists() else None
        key = rel
        c = cache.get(key)
        sig = [int(st.st_size), int(st.st_mtime)] if st else None
        if c and sig and c.get("sig") == sig:
            results[im["id"]] = c["val"]
        else:
            todo.append((im["id"], str(ap), key, sig))

    if todo:
        payload = [(t[0], t[1]) for t in todo]
        try:
            from tqdm import tqdm
        except Exception: # pragma: no cover
            tqdm = lambda x, **k: x # noqa: E731
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in tqdm(ex.map(hash_one, payload, chunksize=32),
                          total=len(payload), desc="hashing"):
                results[r["i"]] = r
        by_id = {t[0]: t for t in todo}
        for r in results.values():
            if not isinstance(r, dict) or "i" not in r:
                continue
            t = by_id.get(r["i"])
            if t and t[3] is not None:
                cache[t[2]] = {"sig": t[3], "val": {k: r[k] for k in
                               ("md5", "dhash", "dw", "dh", "ok")}}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))

    # normalize to a plain per-image dict
    norm = {}
    for im in images:
        r = results.get(im["id"], {})
        norm[im["id"]] = {k: r.get(k) for k in ("md5", "dhash", "dw", "dh", "ok")}
    return norm


# ---------------------------------------------------------------------------
# Perceptual near-duplicate grouping (dHash + LSH banding + union-find)
# ---------------------------------------------------------------------------
def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_dup_groups(dhashes: dict[int, int], radius: int = NEAR_HAM) -> dict[int, int]:
    """Return image_id -> near-duplicate-group-id (only for ids in groups >1).

    LSH: split the 64-bit dHash into 4 x 16-bit bands; images sharing any band are
    candidate pairs. Candidates within Hamming <= radius are unioned. Avoids the
    O(n^2) all-pairs comparison.
    """
    ids = [i for i, h in dhashes.items() if h is not None]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for band in range(4):
        shift = band * 16
        buckets: dict[int, list[int]] = defaultdict(list)
        for i in ids:
            buckets[(dhashes[i] >> shift) & 0xFFFF].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            base = members[0]
            for j in members[1:]:
                if _hamming(dhashes[base], dhashes[j]) <= radius:
                    union(base, j)
                else:
                    # compare against a few others in the bucket, cheap
                    for k in members:
                        if k != j and _hamming(dhashes[k], dhashes[j]) <= radius:
                            union(k, j)
                            break

    comp: dict[int, list[int]] = defaultdict(list)
    for i in ids:
        comp[find(i)].append(i)
    out: dict[int, int] = {}
    gid = 0
    for members in comp.values():
        if len(members) > 1:
            gid += 1
            for i in members:
                out[i] = gid
    return out


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------
def build_frames(master: dict, hashes: dict[int, dict]):
    """Return (images_df, anns_df) with all audit columns computed."""
    cat_name = {c["id"]: c["name"] for c in master["categories"]}
    cat_role = {c["id"]: c.get("role", "benign") for c in master["categories"]}
    im_by_id = {im["id"]: im for im in master["images"]}

    # ---- per-annotation ----
    arows = []
    for a in master["annotations"]:
        im = im_by_id[a["image_id"]]
        W, H = float(im["width"]), float(im["height"])
        x, y, w, h = (float(v) for v in a["bbox"])
        area = w * h
        img_area = W * H if W > 0 and H > 0 else np.nan
        min_side = min(w, h)
        max_side = max(w, h)
        cx, cy = x + w / 2.0, y + h / 2.0
        off = float(np.hypot((cx - W / 2.0) / W, (cy - H / 2.0) / H)) if W and H else np.nan
        row = {
            "ann_id": a["id"], "image_id": a["image_id"], "source": a["source"],
            "orig_label": a.get("orig_label"),
            "category": cat_name.get(a["category_id"]),
            "role": cat_role.get(a["category_id"]),
            "w_px": round(w, 2), "h_px": round(h, 2),
            "min_side_px": round(min_side, 2), "max_side_px": round(max_side, 2),
            "area_px": round(area, 2),
            "area_frac": round(area / img_area, 6) if img_area else np.nan,
            "aspect": round(max_side / min_side, 3) if min_side > 0 else np.nan,
            "center_offset": round(off, 4),
            "degenerate": bool(min_side < 1.0),
        }
        long_side = max(W, H)
        for S in INPUT_SIZES:
            sc = S / long_side if long_side else np.nan
            row[f"min_side_lb{S}"] = round(min_side * sc, 2)
            row[f"area_frac_lb{S}"] = row["area_frac"] # scale-invariant
        row["detector_eligible"] = bool(min_side >= DET_MIN_SIDE)
        row["patch_eligible"] = bool(min_side >= PATCH_MIN_SIDE)
        arows.append(row)
    anns_df = pd.DataFrame(arows)

    # ---- per-image ----
    grp = anns_df.groupby("image_id") if len(anns_df) else None
    n_obj = grp.size() if grp is not None else pd.Series(dtype=int)
    largest_af = grp["area_frac"].max() if grp is not None else pd.Series(dtype=float)
    n_degen = grp["degenerate"].sum() if grp is not None else pd.Series(dtype=int)
    n_det = grp["detector_eligible"].sum() if grp is not None else pd.Series(dtype=int)
    n_patch = grp["patch_eligible"].sum() if grp is not None else pd.Series(dtype=int)
    # center offset of the single largest object per image (chip-likeness)
    largest_center = {}
    if grp is not None:
        for iid, sub in anns_df.groupby("image_id"):
            k = sub["area_frac"].idxmax()
            largest_center[iid] = float(sub.loc[k, "center_offset"])

    irows = []
    for im in master["images"]:
        iid = im["id"]
        seq, is_seq = sequence_id(im["source"], im["orig_id"])
        no = int(n_obj.get(iid, 0))
        laf = float(largest_af.get(iid, 0.0)) if no else 0.0
        loff = largest_center.get(iid, np.nan)
        chip = bool(no >= 1 and laf >= CHIP_AREA_FRAC and
                    (loff is not np.nan and loff <= CHIP_CENTER))
        hd = hashes.get(iid, {})
        dw, dh = hd.get("dw"), hd.get("dh")
        dim_mismatch = bool(dw is not None and dh is not None and
                            (int(dw) != int(im["width"]) or int(dh) != int(im["height"])))
        irows.append({
            "image_id": iid, "source": im["source"], "orig_id": im["orig_id"],
            "file_name": im["file_name"], "orig_split": im.get("orig_split"),
            "width": im["width"], "height": im["height"],
            "sequence_id": seq, "is_sequence_source": is_seq,
            "n_objects": no, "n_degenerate": int(n_degen.get(iid, 0)),
            "largest_area_frac": round(laf, 6),
            "largest_obj_center_offset": (round(loff, 4) if loff is not np.nan else np.nan),
            "chip_like": chip,
            "n_detector_eligible": int(n_det.get(iid, 0)),
            "n_patch_eligible": int(n_patch.get(iid, 0)),
            "file_exists": bool(hd.get("ok")) or (DEFAULT_DATA_DIR is None),
            "decoded_ok": bool(hd.get("ok")),
            "dim_mismatch": dim_mismatch,
            "md5": hd.get("md5"),
            "dhash": hd.get("dhash"),
        })
    images_df = pd.DataFrame(irows)

    # file existence: rely on decoded_ok if hashing ran; else check path
    if images_df["md5"].isna().all():
        images_df["file_exists"] = [
            (DEFAULT_DATA_DIR / fn).exists() for fn in images_df["file_name"]
        ]

    # ---- exact + perceptual duplicate groups ----
    images_df["exact_dup_group"] = np.nan
    images_df["near_dup_group"] = np.nan
    md5_present = images_df["md5"].notna()
    if md5_present.any():
        counts = images_df.loc[md5_present, "md5"].value_counts()
        dup_md5 = {m: i + 1 for i, m in enumerate(counts[counts > 1].index)}
        images_df.loc[md5_present, "exact_dup_group"] = images_df.loc[
            md5_present, "md5"].map(dup_md5)
        dhashes = {int(r.image_id): int(r.dhash)
                   for r in images_df.itertuples() if pd.notna(r.dhash)}
        nd = near_dup_groups(dhashes)
        images_df["near_dup_group"] = images_df["image_id"].map(nd)

    return images_df, anns_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(images_df: pd.DataFrame, anns_df: pd.DataFrame, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(anns_df["source"].unique())

    # 1. bbox shortest-side distribution per source (log x)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(0, np.log10(max(anns_df["min_side_px"].max(), 10)), 40)
    for s in sources:
        ax.hist(anns_df.loc[anns_df.source == s, "min_side_px"], bins=bins,
                histtype="step", label=s, linewidth=1.5)
    for t in DET_SWEEP:
        ax.axvline(t, color="grey", ls=":", lw=0.6)
    ax.axvline(DET_MIN_SIDE, color="k", ls="--", lw=1, label=f"det floor={DET_MIN_SIDE}px")
    ax.axvline(PATCH_MIN_SIDE, color="r", ls="--", lw=1, label=f"patch floor={PATCH_MIN_SIDE}px")
    ax.set_xscale("log"); ax.set_xlabel("bbox shortest side (native px)")
    ax.set_ylabel("annotations"); ax.set_title("Pixels-on-target by source")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "bbox_min_side_by_source.png", dpi=120)
    plt.close(fig)

    # 2. object-area-fraction distribution per source
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    for s in sources:
        ax.hist(anns_df.loc[anns_df.source == s, "area_frac"], bins=bins,
                histtype="step", label=s, linewidth=1.5)
    ax.axvline(CHIP_AREA_FRAC, color="k", ls="--", lw=1, label=f"chip frac={CHIP_AREA_FRAC}")
    ax.set_xlabel("object area / image area"); ax.set_ylabel("annotations")
    ax.set_title("Object-area fraction by source"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "area_fraction_by_source.png", dpi=120)
    plt.close(fig)

    # 3. post-letterbox shortest side ECDF at each input size
    fig, ax = plt.subplots(figsize=(8, 5))
    for S in INPUT_SIZES:
        vals = np.sort(anns_df[f"min_side_lb{S}"].dropna().values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=f"letterbox {S}")
    ax.axvline(DET_MIN_SIDE, color="k", ls="--", lw=1, label=f"det floor={DET_MIN_SIDE}px")
    ax.set_xscale("log"); ax.set_xlim(1, None)
    ax.set_xlabel("post-letterbox shortest side (px)"); ax.set_ylabel("cumulative fraction")
    ax.set_title("Expected target size after letterbox"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "post_letterbox_ecdf.png", dpi=120)
    plt.close(fig)

    # 4. objects per image per source
    fig, ax = plt.subplots(figsize=(8, 5))
    maxo = int(min(images_df["n_objects"].max(), 20))
    bins = np.arange(0, maxo + 2) - 0.5
    for s in sources:
        ax.hist(images_df.loc[images_df.source == s, "n_objects"].clip(upper=maxo),
                bins=bins, histtype="step", label=s, linewidth=1.5)
    ax.set_xlabel(f"objects per image (clipped at {maxo})"); ax.set_ylabel("images")
    ax.set_title("Scene density by source"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "objects_per_image.png", dpi=120)
    plt.close(fig)

    # 5. chip-like fraction per source
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frac = images_df.groupby("source")["chip_like"].mean().reindex(sources)
    ax.bar(range(len(frac)), frac.values * 100)
    ax.set_xticks(range(len(frac))); ax.set_xticklabels(frac.index, rotation=20)
    ax.set_ylabel("% images chip-like")
    ax.set_title(f"Chip-like context (area>={CHIP_AREA_FRAC}, center<={CHIP_CENTER})")
    fig.tight_layout(); fig.savefig(fig_dir / "chip_like_fraction.png", dpi=120)
    plt.close(fig)

    # 6. eligibility retained counts vs threshold sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    det_ret = [(anns_df["min_side_px"] >= t).sum() for t in DET_SWEEP]
    patch_ret = [(anns_df["min_side_px"] >= t).sum() for t in PATCH_SWEEP]
    ax.plot(DET_SWEEP, det_ret, "o-", label="detector-eligible")
    ax.plot(PATCH_SWEEP, patch_ret, "s-", label="patch-eligible")
    ax.axhline(len(anns_df), color="grey", ls=":", lw=0.8, label=f"total={len(anns_df)}")
    ax.set_xlabel("shortest-side floor (native px)"); ax.set_ylabel("annotations retained")
    ax.set_title("Retained annotations vs eligibility floor"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "eligibility_sweep.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary + report
# ---------------------------------------------------------------------------
def build_summary(images_df, anns_df) -> dict:
    sources = sorted(images_df["source"].unique())

    def sweep(df, col, thr):
        return {int(t): int((df[col] >= t).sum()) for t in thr}

    per_source = {}
    for s in sources:
        ai = anns_df[anns_df.source == s]
        ii = images_df[images_df.source == s]
        per_source[s] = {
            "images": int(len(ii)),
            "annotations": int(len(ai)),
            "sequences": int(ii["sequence_id"].nunique()),
            "is_sequence_source": bool(ii["is_sequence_source"].any()),
            "chip_like_images": int(ii["chip_like"].sum()),
            "min_side_px": {
                "p05": round(float(ai["min_side_px"].quantile(0.05)), 1),
                "p50": round(float(ai["min_side_px"].median()), 1),
                "p95": round(float(ai["min_side_px"].quantile(0.95)), 1),
            } if len(ai) else {},
            "area_frac_median": round(float(ai["area_frac"].median()), 5) if len(ai) else None,
            "detector_eligible_sweep": sweep(ai, "min_side_px", DET_SWEEP),
            "patch_eligible_sweep": sweep(ai, "min_side_px", PATCH_SWEEP),
        }

    qa = {
        "degenerate_boxes": int(anns_df["degenerate"].sum()) if len(anns_df) else 0,
        "images_missing_or_unreadable": int((~images_df["file_exists"]).sum()),
        "dim_mismatch_images": int(images_df["dim_mismatch"].sum()),
        "exact_duplicate_groups": int(images_df["exact_dup_group"].notna().any() and
                                       images_df["exact_dup_group"].nunique()) or 0,
        "images_in_exact_dup_group": int(images_df["exact_dup_group"].notna().sum()),
        "near_duplicate_groups": int(images_df["near_dup_group"].notna().any() and
                                     images_df["near_dup_group"].nunique()) or 0,
        "images_in_near_dup_group": int(images_df["near_dup_group"].notna().sum()),
    }

    return {
        "thresholds": {
            "det_min_side_px": DET_MIN_SIDE, "patch_min_side_px": PATCH_MIN_SIDE,
            "det_sweep": DET_SWEEP, "patch_sweep": PATCH_SWEEP,
            "input_sizes": INPUT_SIZES, "chip_area_frac": CHIP_AREA_FRAC,
            "chip_center_offset": CHIP_CENTER, "near_dup_hamming": NEAR_HAM,
        },
        "totals": {
            "images": int(len(images_df)),
            "annotations": int(len(anns_df)),
            "chip_like_images": int(images_df["chip_like"].sum()),
            "detector_eligible": int(anns_df["detector_eligible"].sum()) if len(anns_df) else 0,
            "patch_eligible": int(anns_df["patch_eligible"].sum()) if len(anns_df) else 0,
            "detector_eligible_sweep": sweep(anns_df, "min_side_px", DET_SWEEP) if len(anns_df) else {},
            "patch_eligible_sweep": sweep(anns_df, "min_side_px", PATCH_SWEEP) if len(anns_df) else {},
        },
        "per_source": per_source,
        "qa": qa,
    }


def write_report(summary: dict, report_path: Path):
    t = summary["thresholds"]
    tot = summary["totals"]
    qa = summary["qa"]
    L = []
    L.append("# EO context, quality & pixels-on-target audit\n")
    L.append("Auto-generated by `scripts/data/audit_eo.py`. Re-runs cheaply over "
             "`data/annotations/coco_master.json`; the curated `usv` set "
             "passes through unchanged on integration.\n")
    L.append("## Provisional preprocessing assumptions\n")
    L.append("- Native resolution retained on disk; size interpreted through a "
             f"letterbox (scale by long side) to candidate inputs {t['input_sizes']} "
             "(640 primary, 1280 for small-target recall). No source is pre-resized "
             "to a common native resolution.\n")
    L.append("## Eligibility policies (non-destructive; final floor set in the harmonization spec)\n")
    L.append(f"- **Detector-eligible** — shortest bbox side >= **{t['det_min_side_px']}px** "
             "native. Below this, annotations are noise-dominated and detection is "
             "unreliable (ABOShips-PLUS used a 16px filter; we keep more and report "
             "the sweep). Nothing is deleted — this is a flag.\n")
    L.append(f"- **Patch-eligible** — shortest bbox side >= **{t['patch_min_side_px']}px** "
             "native. A target may be detectable yet too small to carry a bounded, "
             "marine-EOT-surviving adversarial patch; this is a separate, stricter "
             "floor for adversarial-patch evaluation only.\n")
    L.append("## Totals\n")
    L.append(f"- Images: **{tot['images']:,}** · Annotations: **{tot['annotations']:,}**\n")
    L.append(f"- Chip-like images (context tag, not a separate pool): "
             f"**{tot['chip_like_images']:,}**\n")
    L.append(f"- Detector-eligible annotations (>= {t['det_min_side_px']}px): "
             f"**{tot['detector_eligible']:,}** "
             f"({100*tot['detector_eligible']/max(tot['annotations'],1):.1f}%)\n")
    L.append(f"- Patch-eligible annotations (>= {t['patch_min_side_px']}px): "
             f"**{tot['patch_eligible']:,}** "
             f"({100*tot['patch_eligible']/max(tot['annotations'],1):.1f}%)\n")
    L.append("\n### Retained annotations vs eligibility floor (sweep)\n")
    L.append("| floor px | detector-eligible | patch-eligible |")
    L.append("|---|---|---|")
    allthr = sorted(set(t["det_sweep"]) | set(t["patch_sweep"]))
    for thr in allthr:
        d = tot["detector_eligible_sweep"].get(str(thr), tot["detector_eligible_sweep"].get(thr, ""))
        p = tot["patch_eligible_sweep"].get(str(thr), tot["patch_eligible_sweep"].get(thr, ""))
        L.append(f"| {thr} | {d if d!='' else '—'} | {p if p!='' else '—'} |")
    L.append("\n## Per-source\n")
    L.append("| source | images | anns | seqs | seq-src | chip-like | "
             "min-side p05/p50/p95 | area-frac med | det>=8 | patch>=32 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, d in summary["per_source"].items():
        ms = d.get("min_side_px", {})
        msc = f"{ms.get('p05','?')}/{ms.get('p50','?')}/{ms.get('p95','?')}" if ms else "—"
        det8 = d["detector_eligible_sweep"].get(8, d["detector_eligible_sweep"].get("8", "?"))
        p32 = d["patch_eligible_sweep"].get(32, d["patch_eligible_sweep"].get("32", "?"))
        L.append(f"| {s} | {d['images']:,} | {d['annotations']:,} | {d['sequences']:,} | "
                 f"{'yes' if d['is_sequence_source'] else 'no'} | {d['chip_like_images']:,} | "
                 f"{msc} | {d['area_frac_median']} | {det8:,} | {p32:,} |")
    L.append("\n## Quality / duplicate QA\n")
    L.append(f"- Degenerate boxes (min side < 1px): **{qa['degenerate_boxes']}**\n")
    L.append(f"- Images missing / unreadable: **{qa['images_missing_or_unreadable']}**\n")
    L.append(f"- Images whose decoded dims disagree with the master: "
             f"**{qa['dim_mismatch_images']}**\n")
    L.append(f"- Exact-duplicate groups: **{qa['exact_duplicate_groups']}** "
             f"covering **{qa['images_in_exact_dup_group']}** images\n")
    L.append(f"- Perceptual near-duplicate groups (dHash, Hamming<={t['near_dup_hamming']}): "
             f"**{qa['near_duplicate_groups']}** covering "
             f"**{qa['images_in_near_dup_group']}** images. These groups (and the "
             "sequence ids) constrain the leakage-safe split in the split.\n")
    L.append("\n## Figures\n")
    for f in ["bbox_min_side_by_source", "area_fraction_by_source",
              "post_letterbox_ecdf", "objects_per_image", "chip_like_fraction",
              "eligibility_sweep"]:
        L.append(f"- `figures/{f}.png`")
    L.append("\n## Manifests\n")
    L.append("- `data/audit/eo_annotations.csv` — per-annotation (size, context, "
             "post-letterbox size, eligibility flags).")
    L.append("- `data/audit/eo_images.csv` — per-image (context, chip tag, "
             "sequence id, QA, dedup groups).")
    report_path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    global DEFAULT_DATA_DIR
    p = argparse.ArgumentParser(description="EO audit.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--master", type=Path, default=None,
                   help="default: <data-dir>/annotations/coco_master.json")
    p.add_argument("--sources", nargs="+", help="restrict to these sources")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--audit-dir", type=Path, default=None,
                   help="manifest dir (default: <data-dir>/audit)")
    p.add_argument("--no-hash", action="store_true",
                   help="skip md5/dHash (no dedup/decoded-dim QA); fast")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args(argv)

    DEFAULT_DATA_DIR = args.data_dir
    master_path = args.master or (args.data_dir / "annotations" / "coco_master.json")
    audit_dir = args.audit_dir or (args.data_dir / "audit")
    fig_dir = args.results_dir / "figures"
    audit_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    master = json.loads(master_path.read_text())
    if args.sources:
        keep = set(args.sources)
        master["images"] = [im for im in master["images"] if im["source"] in keep]
        ids = {im["id"] for im in master["images"]}
        master["annotations"] = [a for a in master["annotations"] if a["image_id"] in ids]
    print(f"[audit] {len(master['images'])} images / {len(master['annotations'])} anns "
          f"from {master_path}")

    if args.no_hash:
        hashes = {im["id"]: {"md5": None, "dhash": None, "dw": None, "dh": None,
                             "ok": None} for im in master["images"]}
    else:
        hashes = compute_hashes(master["images"], args.data_dir, args.jobs,
                                audit_dir / "_hash_cache.json")

    images_df, anns_df = build_frames(master, hashes)

    anns_df.to_csv(audit_dir / "eo_annotations.csv", index=False)
    images_df.drop(columns=["dhash"]).to_csv(audit_dir / "eo_images.csv", index=False)

    summary = build_summary(images_df, anns_df)
    (audit_dir / "eo_audit_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        make_figures(images_df, anns_df, fig_dir)
    except Exception as e: # figures are best-effort
        print(f"[warn] figure generation failed: {e}")

    write_report(summary, args.results_dir / "report.md")

    t, tot, qa = summary["thresholds"], summary["totals"], summary["qa"]
    print(f"\n[audit] totals: images={tot['images']} anns={tot['annotations']}")
    print(f" detector-eligible (>= {t['det_min_side_px']}px): {tot['detector_eligible']} "
          f"({100*tot['detector_eligible']/max(tot['annotations'],1):.1f}%)")
    print(f" patch-eligible (>= {t['patch_min_side_px']}px): {tot['patch_eligible']} "
          f"({100*tot['patch_eligible']/max(tot['annotations'],1):.1f}%)")
    print(f" chip-like images: {tot['chip_like_images']}")
    print(f" QA: degenerate={qa['degenerate_boxes']} missing={qa['images_missing_or_unreadable']} "
          f"dim_mismatch={qa['dim_mismatch_images']} "
          f"exact_dup_imgs={qa['images_in_exact_dup_group']} "
          f"near_dup_imgs={qa['images_in_near_dup_group']}")
    print(f"\nWrote:\n {audit_dir}/eo_annotations.csv\n {audit_dir}/eo_images.csv"
          f"\n {audit_dir}/eo_audit_summary.json\n {args.results_dir}/report.md"
          f"\n {fig_dir}/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
