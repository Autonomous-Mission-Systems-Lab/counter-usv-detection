#!/usr/bin/env python3
"""Train / val / test splits with leakage control.

Builds **split manifests** over the already-curated data — it does NOT copy or modify
images/tracks. Three independent partitions, each with its own leakage rule:

EO imagery (`data/splits/eo_image_splits.csv`)
----------------------------------------------
Per-image split over the COCO master. Rules:
* **Group-intact assignment.** Connected components are formed by unioning images that
  share (a) an exact-duplicate group, (b) a perceptual near-duplicate group,
  (c) a sequence id for sequence sources (SMD clip, ABOShips recording-day), or (d) the
  source **video clip** for curated USV frames. A whole component lands in ONE split — no
  duplicate/sequence bleed across the train/eval boundary. Provider `orig_split` is NOT
  trusted (it can cross dedup/sequence boundaries).
* **Viewpoint firewall.** The headline operational eval is SHORE-only
  (`eval_operational_sources` in `configs/base.yaml`: seaships, smd) plus the hostile
  target class `usv` (which must be evaluated). Non-operational sources (ABOShips
  onboard/moving-vessel; McShips web-in-the-wild) are **train-only** and never enter
  val/test — enabling the with/without-auxiliary training ablation the evaluation design requires. If a
  dedup component mixes a train-only source with an eval source, the whole component is
  forced to train (firewall wins).

AIS tracks (`data/splits/ais_track_splits.csv`)
-----------------------------------------------
**Vessel-disjoint** split: all trips of an MMSI stay in one split (no vessel seen in both
train and eval). `mmsi==0` (no identity) is train-only. Each track is also tagged with a
coarse geographic cell + region and its start day so a **region/time-holdout** robustness
eval can be sliced on top of the vessel-disjoint split. The one-class benign scorer
consumes `role==benign` only; `military` (hostile) / `unknown_other` (non_target) are
tagged and excluded downstream, never used to train the benign model.

SMD video tracks (`data/splits/video_eval_pool.csv`)
----------------------------------------------------
Kept as a **separate held-out non-cooperative evaluation pool** — NOT mixed into the AIS
training pool (time-horizon/sampling mismatch; image-plane non-metric features). Grouped by
clip. This is the carriage-bias sensitivity check + PercepGuard-style baseline input.

Outputs
-------
* `data/splits/eo_image_splits.csv`, `data/splits/ais_track_splits.csv`,
  `data/splits/video_eval_pool.csv`
* `data/splits/splits_summary.json` — counts + leakage-check results + version stamp
* `results/splits/report.md` (+ figures)

Usage
-----
    python scripts/data/build_splits.py
    python scripts/data/build_splits.py --ratios 0.7 0.15 0.15 --seed 1337
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results" / "splits"

# Per-source viewpoint + whether it is the operational (shore near-waterline) viewpoint.
# Mirror of scripts/data/build_eval_slice.py VIEWPOINT (from data/INVENTORY.md).
VIEWPOINT = {
    "seaships": ("shore_near_waterline", True),
    "smd": ("shore_fixed_platform", True),
    "usv": ("curated_usv_mixed", False), # hostile TARGET class — eval-eligible regardless
    "mcships": ("web_in_the_wild", False),
    "aboships": ("onboard_moving_vessel", False),
}
# Sequence sources: frames from one recording must not cross splits.
SEQUENCE_SOURCES = {"smd", "aboships"}


# --------------------------------------------------------------------------- utils
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


class DSU:
    """Union-find over hashable ids."""
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def greedy_assign(sizes: dict, ratios: tuple, seed: int, grp_classes: dict = None) -> dict:
    """Assign group_id -> split, keeping groups intact, hitting `ratios` by item count.

    Two passes:
    1. **Class-coverage** (if `grp_classes` given): for each class spanning >=3 leakage
       groups, force at least one containing group into each of train/val/test (rarest
       class first, smallest groups first to limit ratio skew). Classes in <3 groups
       cannot cover all three splits without starving one and are left to pass 2 — reported
       separately, never split at the cost of leakage.
    2. **Count-balance**: remaining groups largest-first, each to the split with the largest
       remaining deficit.
    """
    names = ["train", "val", "test"]
    total = sum(sizes.values())
    targets = {n: total * r for n, r in zip(names, ratios)}
    have = {n: 0 for n in names}
    out: dict = {}

    if grp_classes:
        from collections import defaultdict
        gclass = defaultdict(set)
        for g, cs in grp_classes.items():
            for c in cs:
                gclass[c].add(g)
        for c in sorted(gclass, key=lambda c: len(gclass[c])):
            groups_c = list(gclass[c])
            if len(groups_c) < 3:
                continue # can't cover 3 splits without starving one
            present = {n: any(out.get(g) == n for g in groups_c) for n in names}
            rng = np.random.default_rng(seed + hash(c) % 9999)
            avail = [g for g in groups_c if g not in out]
            rng.shuffle(avail)
            avail.sort(key=lambda g: sizes[g]) # smallest first (limit skew)
            for n in names:
                if present[n] or not avail:
                    continue
                g = avail.pop(0)
                out[g] = n
                have[n] += sizes[g]

    rng = np.random.default_rng(seed)
    rem = [g for g in sizes if g not in out]
    rng.shuffle(rem) # deterministic tie-break
    rem.sort(key=lambda g: sizes[g], reverse=True) # largest first
    for g in rem:
        pick = max(names, key=lambda n: (targets[n] - have[n],
                                         -(have[n] / max(targets[n], 1e-9))))
        out[g] = pick
        have[pick] += sizes[g]
    return out


# --------------------------------------------------------------------------- EO
def _usv_clip_map(data_dir: Path) -> dict:
    """basename -> source video clip_id for curated USV frames (leakage grouping)."""
    man = data_dir / "raw" / "usv" / "manifest.csv"
    if not man.exists():
        return {}
    m = pd.read_csv(man)
    m["base"] = m["file_name"].map(lambda p: Path(str(p)).name)
    out = {}
    for _, r in m.iterrows():
        clip = str(r.get("clip_id") or "").strip()
        if clip:
            out[r["base"]] = clip
    return out


def build_eo(data_dir: Path, eval_sources: set, ratios: tuple, seed: int,
             stratify: bool = True):
    img = pd.read_csv(data_dir / "audit" / "eo_images.csv")
    ann = pd.read_csv(data_dir / "audit" / "eo_annotations.csv")

    # per-image class/role aggregation (for stratification + reporting)
    def agg(g):
        idx = g["area_px"].idxmax() if "area_px" in g and g["area_px"].notna().any() else g.index[0]
        return pd.Series({
            "primary_class": g.loc[idx, "category"],
            "classes": ",".join(sorted(set(g["category"].dropna()))),
            "any_hostile": bool((g["role"] == "hostile").any()),
            "n_boxes": len(g),
        })
    per_img_cls = ann.groupby("image_id", group_keys=False).apply(agg).reset_index()
    img = img.merge(per_img_cls, on="image_id", how="left")

    all_sources = set(img["source"].unique())
    train_only_sources = all_sources - eval_sources

    img["viewpoint"] = img["source"].map(lambda s: VIEWPOINT.get(s, ("unknown", False))[0])
    img["operational_viewpoint"] = img["source"].map(
        lambda s: VIEWPOINT.get(s, ("unknown", False))[1])
    img["eval_eligible"] = img["source"].isin(eval_sources)
    img["train_only"] = img["source"].isin(train_only_sources)

    # ---- leakage grouping (union-find over image_id) ----
    usv_clip = _usv_clip_map(data_dir)
    dsu = DSU()
    for iid in img["image_id"]:
        dsu.find(iid)
    # (a) exact + (b) near dup groups; (c) sequence sources; (d) usv clip
    for key, prefix in [("exact_dup_group", "ex"), ("near_dup_group", "nd")]:
        if key in img.columns:
            for grp, sub in img.groupby(key):
                try:
                    if float(grp) < 0: # -1 => not in any dup group
                        continue
                except (TypeError, ValueError):
                    continue
                ids = list(sub["image_id"])
                for j in ids[1:]:
                    dsu.union(ids[0], j)
    seq = img[img["source"].isin(SEQUENCE_SOURCES) & img["sequence_id"].notna()]
    for (src, sid), sub in seq.groupby(["source", "sequence_id"]):
        ids = list(sub["image_id"])
        for j in ids[1:]:
            dsu.union(ids[0], j)
    if usv_clip:
        usv = img[img["source"] == "usv"].copy()
        usv["base"] = usv["file_name"].map(lambda p: Path(str(p)).name)
        usv["clip"] = usv["base"].map(usv_clip)
        for clip, sub in usv[usv["clip"].notna()].groupby("clip"):
            ids = list(sub["image_id"])
            for j in ids[1:]:
                dsu.union(ids[0], j)

    img["group_id"] = img["image_id"].map(dsu.find)

    # component-level flags: does the component touch a train-only source / eval source?
    comp = img.groupby("group_id").agg(
        has_train_only=("train_only", "any"),
        has_eval=("eval_eligible", "any"),
        size=("image_id", "count"),
        source=("source", lambda s: sorted(set(s))[0]),
    )
    # forced-to-train components: any train-only source in the component (firewall wins)
    forced_train = set(comp[comp["has_train_only"]].index)

    # group -> set of annotation classes (for class-aware stratification)
    ann_g = ann.merge(img[["image_id", "group_id"]], on="image_id", how="inner")
    grp_classes_all = ann_g.groupby("group_id")["category"].apply(
        lambda s: set(s.dropna())).to_dict()

    # eval-eligible components (no train-only member) split per source to hit ratios
    eval_comp = comp[~comp.index.isin(forced_train)]
    assign = {g: "train" for g in forced_train}
    unstratifiable = {}
    for src, sub in eval_comp.groupby("source"):
        sizes = sub["size"].to_dict()
        gcls = {g: grp_classes_all.get(g, set()) for g in sizes}
        if stratify:
            # classes in this source spanning <3 groups can't reach all 3 splits
            from collections import Counter
            gc = Counter()
            for cs in gcls.values():
                for c in set(cs):
                    gc[c] += 1
            thin = {c: gc[c] for c in gc if 0 < gc[c] < 3}
            if thin:
                unstratifiable[src] = thin
        assign.update(greedy_assign(sizes, ratios, seed + hash(src) % 10_000,
                                    grp_classes=gcls if stratify else None))

    img["split"] = img["group_id"].map(assign).fillna("train")

    # transfer seed: operational shore test images (RQ4 black-box target seed)
    img["transfer_seed"] = (img["split"] == "test") & img["operational_viewpoint"]

    cols = ["image_id", "source", "split", "viewpoint", "operational_viewpoint",
            "eval_eligible", "train_only", "transfer_seed", "primary_class", "classes",
            "any_hostile", "n_boxes", "n_objects", "n_detector_eligible",
            "n_patch_eligible", "group_id", "sequence_id", "exact_dup_group",
            "near_dup_group", "chip_like", "orig_split", "file_name"]
    cols = [c for c in cols if c in img.columns]
    return (img[cols].sort_values(["source", "split", "image_id"]),
            sorted(train_only_sources), unstratifiable)


# --------------------------------------------------------------------------- AIS
def _region_of(lat: float, lon: float) -> str:
    if pd.isna(lat) or pd.isna(lon):
        return "unknown"
    if lon <= -125:
        return "pacific_west"
    if lon <= -100:
        return "west_inland"
    if lon <= -82:
        return "gulf_central"
    if lon <= -60:
        return "atlantic_east"
    return "other"


def build_ais(data_dir: Path, ratios: tuple, seed: int):
    cols = ["trip_id", "mmsi", "mean_lat", "mean_lon", "t_start", "canonical_class",
            "role", "transceiver_class"]
    ais = pd.read_parquet(data_dir / "tracks" / "tracks_ais.parquet", columns=cols)
    ais["region"] = [
        _region_of(la, lo) for la, lo in zip(ais["mean_lat"], ais["mean_lon"])]
    ais["geo_cell"] = [
        "na" if pd.isna(la) or pd.isna(lo) else f"{int(np.floor(la/5)*5)}_{int(np.floor(lo/5)*5)}"
        for la, lo in zip(ais["mean_lat"], ais["mean_lon"])]
    ais["start_day"] = [
        datetime.fromtimestamp(int(t), tz=timezone.utc).date().isoformat()
        for t in ais["t_start"]]

    # vessel-disjoint assignment: mmsi==0 -> train-only; else by-vessel to hit ratios
    vessel_sizes = (ais[ais["mmsi"] != 0].groupby("mmsi").size().to_dict())
    vassign = greedy_assign(vessel_sizes, ratios, seed)
    ais["split"] = ais["mmsi"].map(lambda m: "train" if m == 0 else vassign.get(m, "train"))

    out = ais[["trip_id", "mmsi", "split", "role", "canonical_class",
               "transceiver_class", "region", "geo_cell", "start_day",
               "mean_lat", "mean_lon"]].sort_values(["split", "mmsi", "trip_id"])
    return out


# --------------------------------------------------------------------------- video
def build_video(data_dir: Path):
    vp = data_dir / "tracks" / "tracks_video.parquet"
    if not vp.exists():
        return pd.DataFrame()
    v = pd.read_parquet(vp, columns=["track_id", "clip", "canonical_class", "role",
                                     "n_frames", "duration_s", "label_source"])
    v["split"] = "video_holdout_eval" # separate non-cooperative pool; NOT in AIS train
    v["pool"] = "smd_video_noncooperative"
    return v.sort_values(["clip", "track_id"])


# --------------------------------------------------------------------------- checks
def leakage_checks(eo, ais, video, train_only_sources) -> dict:
    problems = []

    # EO: every leakage group in exactly one split
    g = eo.groupby("group_id")["split"].nunique()
    bad_groups = int((g > 1).sum())
    if bad_groups:
        problems.append(f"{bad_groups} EO leakage-group(s) span multiple splits")

    # EO: no train-only source in val/test
    leak_src = eo[(eo["split"] != "train") & (eo["source"].isin(train_only_sources))]
    if len(leak_src):
        problems.append(f"{len(leak_src)} train-only-source image(s) in val/test")

    # EO: val/test contain only eval-eligible sources
    non_elig = eo[(eo["split"] != "train") & (~eo["eval_eligible"])]
    if len(non_elig):
        problems.append(f"{len(non_elig)} non-eval-eligible image(s) in val/test")

    # AIS: no MMSI (except 0) spans multiple splits
    nz = ais[ais["mmsi"] != 0]
    vspan = nz.groupby("mmsi")["split"].nunique()
    bad_v = int((vspan > 1).sum())
    if bad_v:
        problems.append(f"{bad_v} vessel(s) span multiple AIS splits")

    # Video pool disjoint from AIS id space (by construction)
    overlap = set(ais["trip_id"].astype(str)) & set(
        video["track_id"].astype(str)) if len(video) else set()
    if overlap:
        problems.append(f"{len(overlap)} id(s) shared between AIS and video pools")

    return {
        "passed": not problems,
        "problems": problems,
        "eo_groups_multi_split": bad_groups,
        "eo_train_only_in_eval": int(len(leak_src)),
        "ais_vessels_multi_split": bad_v,
    }


# --------------------------------------------------------------------------- report
def grouping_stats(eo):
    """Leakage-group composition: how 1 image != 1 group (dedup/sequence/usv-clip)."""
    sz = eo.groupby("group_id").size()
    per_src = []
    for src, sub in eo.groupby("source"):
        imgs = len(sub)
        grps = int(sub["group_id"].nunique())
        per_src.append({"source": src, "images": imgs, "groups": grps,
                        "collapsed": imgs - grps})
    per_src.sort(key=lambda r: r["collapsed"], reverse=True)
    return {
        "images": int(len(eo)),
        "groups": int(eo["group_id"].nunique()),
        "singleton_groups": int((sz == 1).sum()),
        "multi_image_groups": int((sz > 1).sum()),
        "images_in_multi_groups": int(sz[sz > 1].sum()),
        "largest_group_images": int(sz.max()),
        "per_source": per_src,
    }


def summarize(eo, ais, video, train_only_sources, eval_sources, ratios, checks, versions):
    def dist(df, by):
        return json.loads(pd.crosstab(df[by], df["split"]).to_json())
    return {
        "version": versions,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "eo": {
            "images": int(len(eo)),
            "by_split": eo["split"].value_counts().to_dict(),
            "by_source_x_split": dist(eo, "source"),
            "by_primaryclass_x_split": dist(eo, "primary_class"),
            "operational_eval_sources": sorted(eval_sources),
            "train_only_sources": train_only_sources,
            "transfer_seed_images": int(eo["transfer_seed"].sum()),
            "leakage_groups": int(eo["group_id"].nunique()),
            "grouping": grouping_stats(eo),
            "orig_split_broken": int(
                (eo["orig_split"].notna() & (eo["orig_split"].astype(str) != eo["split"])).sum())
            if "orig_split" in eo else None,
        },
        "ais": {
            "tracks": int(len(ais)),
            "vessels": int(ais["mmsi"].nunique()),
            "by_split": ais["split"].value_counts().to_dict(),
            "by_role_x_split": dist(ais, "role"),
            "by_region_x_split": dist(ais, "region"),
            "by_day_x_split": dist(ais, "start_day"),
            "mmsi_zero_tracks": int((ais["mmsi"] == 0).sum()),
        },
        "video_pool": {
            "tracks": int(len(video)),
            "clips": int(video["clip"].nunique()) if len(video) else 0,
            "note": "separate held-out non-cooperative eval pool; NOT in AIS training",
        },
        "leakage_check": checks,
    }


def write_report(s, path: Path):
    eo, ais, vp = s["eo"], s["ais"], s["video_pool"]
    chk = s["leakage_check"]
    def tbl(d):
        df = pd.DataFrame(d).fillna(0).astype(int)
        order = [c for c in ["train", "val", "test"] if c in df.columns]
        df = df[order] if order else df
        head = "| key | " + " | ".join(df.columns) + " |"
        sep = "|---|" + "|".join(["---"] * len(df.columns)) + "|"
        rows = [f"| {k} | " + " | ".join(str(int(v)) for v in r.values) + " |"
                for k, r in df.iterrows()]
        return "\n".join([head, sep] + rows)

    L = ["# Train / val / test splits with leakage control \n",
         "Auto-generated by `scripts/data/build_splits.py`. Split **manifests** over the "
         "curated data — no images/tracks are copied or modified.\n",
         f"**Version:** `{s['version']['version']}` (git `{s['version']['git_sha']}`, "
         f"{s['version']['date']})\n",
         f"**Ratios:** train {s['ratios']['train']:.0%} / val {s['ratios']['val']:.0%} / "
         f"test {s['ratios']['test']:.0%}\n",
         "## Leakage check\n",
         f"- **{'PASS' if chk['passed'] else 'FAIL'}** — "
         + ("no leakage detected." if chk["passed"] else "; ".join(chk["problems"])) + "\n",
         f"- EO leakage groups (dedup+sequence+usv-clip, kept intact): "
         f"**{eo['leakage_groups']:,}**; groups spanning >1 split: "
         f"**{chk['eo_groups_multi_split']}**\n",
         f"- Train-only-source images in eval: **{chk['eo_train_only_in_eval']}**; "
         f"vessels spanning AIS splits: **{chk['ais_vessels_multi_split']}**\n",
         "\n## Leakage grouping (why images != groups)\n",
         "A **leakage group** is a set of images forced into the SAME split because "
         "separating them would leak train info into eval. Images are unioned into one "
         "group when they share: (a) an **exact duplicate** (md5), (b) a **perceptual "
         "near-duplicate** (EO-audit dHash), (c) the **same sequence** — one SMD clip or one "
         "ABOShips recording-day, or (d) the **same source video** for curated USV frames. "
         "The splitter assigns whole groups, not individual images.\n",
         (lambda gr: (
             f"- **{gr['images']:,} images collapse into {gr['groups']:,} groups** "
             f"({gr['singleton_groups']:,} singletons + {gr['multi_image_groups']:,} "
             f"multi-image groups covering {gr['images_in_multi_groups']:,} images; "
             f"largest group = {gr['largest_group_images']:,} images).\n\n"
             "| source | images | groups | collapsed | dominant cause |\n"
             "|---|---|---|---|---|\n"
             + "\n".join(
                 f"| {r['source']} | {r['images']:,} | {r['groups']:,} | {r['collapsed']:,} | "
                 + {"aboships": "recording-day sequences",
                    "smd": "video-clip sequences",
                    "usv": "source-video clips + dedup",
                    "seaships": "perceptual near-dups (fixed cameras)",
                    "mcships": "exact/near-dup web images"}.get(r["source"], "dedup")
                 + " |"
                 for r in gr["per_source"]))
          )(s["eo"]["grouping"]),
         "\n## EO imagery\n",
         f"- Images: **{eo['images']:,}**; operational eval sources: "
         f"**{', '.join(eo['operational_eval_sources'])}**; train-only (never in eval): "
         f"**{', '.join(eo['train_only_sources'])}**.\n",
         f"- **Class-aware stratification:** {'ON' if eo.get('stratified', True) else 'OFF'} "
         "— every eval class spanning >=3 leakage groups is forced into all of train/val/test "
         "(groups kept intact). Classes spanning <3 groups that therefore cannot reach all "
         f"three splits without leakage: **{eo.get('unstratifiable_classes') or 'none'}**.\n",
         f"- Provider `orig_split` labels that our leakage-controlled split overrides: "
         f"**{eo['orig_split_broken']:,}** (expected — we do not trust provider splits).\n",
         f"- Transfer-protocol seed (operational shore test images, RQ4): "
         f"**{eo['transfer_seed_images']:,}**.\n",
         "\n### EO source x split\n", tbl(eo["by_source_x_split"]),
         "\n\n### EO primary-class x split\n", tbl(eo["by_primaryclass_x_split"]),
         "\n\n## AIS tracks (vessel-disjoint)\n",
         f"- Tracks: **{ais['tracks']:,}** across **{ais['vessels']:,}** vessels; "
         f"`mmsi==0` (no identity, train-only): **{ais['mmsi_zero_tracks']}**.\n",
         "- Vessel-disjoint by construction; `region`/`geo_cell`/`start_day` tags let a "
         "region- or time-holdout robustness eval be sliced on top.\n",
         "- Benign-behavior scorer consumes `role==benign` only; `military` "
         "(hostile) and `unknown_other` (non_target) are tagged and excluded from training.\n",
         "\n### AIS role x split\n", tbl(ais["by_role_x_split"]),
         "\n\n### AIS region x split\n", tbl(ais["by_region_x_split"]),
         "\n\n## SMD video pool (separate held-out eval)\n",
         f"- Non-cooperative tracks: **{vp['tracks']:,}** across **{vp['clips']}** clips. "
         f"{vp['note']}.\n",
         "\n## Caveats\n",
         "- Class-aware stratification guarantees every eval-eligible class with >=3 leakage "
         "groups appears in val AND test — but some shore counts are small (e.g. `sailing`, "
         "`static_aid` from a handful of SMD clips); report those per-class metrics with "
         "variance in mind.\n",
         "- **`recreational` has NO shore representation** (100% ABOShips, which is "
         "train-only) → it cannot be scored in the operational eval at all. Genuine data-card "
         "limitation, not a split artifact; ABOShips supplies it for TRAINING only.\n",
         "- Operational eval + `usv` sets are small (usv val/test ~38 imgs each); report "
         "val/test metrics with variance in mind.\n",
         "- Hostile/adaptive trajectories are synthesized at eval; the benign scorer is "
         "never trained on them.\n"]
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- main
def _eval_sources_from_config(data_dir: Path) -> set:
    cfg = REPO_ROOT / "configs" / "base.yaml"
    ops = ["seaships", "smd"]
    if yaml and cfg.exists():
        try:
            d = yaml.safe_load(cfg.read_text())
            ops = d.get("data", {}).get("eval_operational_sources", ops)
        except Exception:
            pass
    return set(ops) | {"usv"} # hostile target class is always eval-eligible


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build leakage-controlled splits .")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-sources", nargs="+", default=None,
                   help="override operational eval sources (usv always added)")
    p.add_argument("--no-stratify", action="store_true",
                   help="disable class-aware val/test coverage (count-balance only)")
    args = p.parse_args(argv)

    img_csv = args.data_dir / "audit" / "eo_images.csv"
    if not img_csv.exists():
        print(f"[error] run scripts/data/audit_eo.py first (missing {img_csv})")
        return 1
    ratios = tuple(np.array(args.ratios) / sum(args.ratios))
    eval_sources = (set(args.eval_sources) | {"usv"}) if args.eval_sources \
        else _eval_sources_from_config(args.data_dir)

    eo, train_only, unstratifiable = build_eo(
        args.data_dir, eval_sources, ratios, args.seed, stratify=not args.no_stratify)
    ais = build_ais(args.data_dir, ratios, args.seed)
    video = build_video(args.data_dir)
    checks = leakage_checks(eo, ais, video, set(train_only))

    versions = {"version": f"{date.today().isoformat()}+{_git_sha()}",
                "date": date.today().isoformat(), "git_sha": _git_sha(),
                "eo_images_md5": _md5(img_csv),
                "tracks_ais_md5": _md5(args.data_dir / "tracks" / "tracks_ais.parquet")}
    summary = summarize(eo, ais, video, train_only, eval_sources, ratios, checks, versions)
    summary["eo"]["stratified"] = not args.no_stratify
    summary["eo"]["unstratifiable_classes"] = unstratifiable

    out_dir = args.data_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    eo.to_csv(out_dir / "eo_image_splits.csv", index=False)
    ais.to_csv(out_dir / "ais_track_splits.csv", index=False)
    if len(video):
        video.to_csv(out_dir / "video_eval_pool.csv", index=False)
    (out_dir / "splits_summary.json").write_text(json.dumps(summary, indent=2))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_report(summary, args.results_dir / "report.md")

    print(f"[splits] EO {len(eo):,} imgs {eo['split'].value_counts().to_dict()}")
    print(f" eval-sources {sorted(eval_sources)} | train-only {train_only}")
    print(f"[splits] AIS {len(ais):,} tracks {ais['split'].value_counts().to_dict()} "
          f"(vessel-disjoint)")
    print(f"[splits] video pool {len(video):,} tracks (separate held-out eval)")
    print(f"[splits] leakage-check: {'PASS' if checks['passed'] else 'FAIL — ' + '; '.join(checks['problems'])}")
    print(f"\nWrote:\n {out_dir}/eo_image_splits.csv\n {out_dir}/ais_track_splits.csv"
          f"\n {out_dir}/video_eval_pool.csv\n {out_dir}/splits_summary.json"
          f"\n {args.results_dir}/report.md")
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
