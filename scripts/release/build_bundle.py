#!/usr/bin/env python3
"""Build the weight-free data/results release bundle.

Stages derived data (with release transforms), freezes, result summaries, and
paper figures under ``dist/release/``. Does **not** ship model weights, patch
banks, point-level AIS, or McShips annotations.

Usage
-----
    python scripts/release/build_bundle.py
    python scripts/release/build_bundle.py --skip-verify-freezes   # offline/dev
    python scripts/release/build_bundle.py --verify dist/release/stage
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "report"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))

from _common import git_sha, sha256, verify_freezes  # noqa: E402
from package_derived import DERIVED_ARTIFACTS  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "dist" / "release"
STAGE_NAME = "stage"

WEIGHT_SUFFIXES = (
    ".pt",
    ".pth",
    ".joblib",
    ".onnx",
    ".engine",
    ".ckpt",
    ".safetensors",
)

# Relative to repo root — copied as-is (after optional transform).
RESULTS_EXACT = [
    "results/attacks/FROZEN.json",
    "results/attacks/REDACTION.md",
    "results/attacks/RELEASE_NOTES.md",
    "results/defense/FROZEN.json",
    "results/defense/MODEL_CARD.md",
    "results/detector_baselines/FROZEN.json",
    "results/detector_baselines/report.md",
    "results/detector_baselines/train_summary.json",
    "results/detector_baselines/usv_capability.md",
    "results/detector_baselines/clean_map/clean_map_summary.json",
    "results/behavior_model/FROZEN.json",
    "results/behavior_model/MODEL_CARD.md",
    "results/behavior_model/far_summary.json",
    "results/behavior_model/fit_summary.json",
    "results/behavior_model/far_report.md",
    "results/behavior_model/fit_report.md",
    "results/behavior_model/corpus_report.md",
    "results/behavior_model/envelope_map_report.md",
    "results/behavior_model/windows_report.md",
    "results/behavior_model_geometry/FROZEN.json",
    "results/behavior_model_geometry/far_placement_summary.json",
    "results/behavior_model_geometry/fit_summary.json",
    "results/behavior_model_geometry/far_placement_report.md",
    "results/behavior_model_geometry/fit_report.md",
    "results/adversary_motion/FROZEN_SWEEP.json",
    "results/adversary_motion/validity_summary.json",
    "results/adversary_motion/validity_report.md",
    "results/oracle_ddr/oracle_ddr_summary.json",
    "results/oracle_ddr/oracle_ddr_report.md",
    "results/oracle_ddr/oracle_ddr_cells.parquet",
    "results/oracle_ddr_pooled/oracle_ddr_summary.json",
    "results/oracle_ddr_pooled/oracle_ddr_report.md",
    "results/oracle_ddr_pooled/oracle_ddr_cells.parquet",
    "results/adaptive_cost/adaptive_cost_summary.json",
    "results/adaptive_cost/adaptive_cost_report.md",
    "results/adaptive_cost/adaptive_cost_curves.parquet",
    "results/adaptive_cost/adaptive_cost_warning.parquet",
    "results/label_swap/label_swap_summary.json",
    "results/label_swap/label_swap_report.md",
    "results/label_swap/label_swap_cells.parquet",
    "results/label_swap_pooled/label_swap_summary.json",
    "results/label_swap_pooled/label_swap_report.md",
    "results/label_swap_pooled/label_swap_cells.parquet",
]

# Directory trees under results/ copied with extension filter (no weights).
RESULTS_TREES = [
    ("results/paper", None),  # all files
    ("results/attacks/artifact_v1", None),
    ("results/attacks/marine_eot", {".json", ".md", ".png"}),
    ("results/attacks/evasion", {".json", ".md"}),
    ("results/attacks/disguise", {".json", ".md"}),
    ("results/attacks/transfer", {".json", ".md"}),
    ("results/attacks/oracle", {".json", ".md"}),
    ("results/attacks/patch_core", {".json", ".md"}),
]

MMSI_DROP_RELS = {
    "tracks/tracks_ais.parquet",
    "splits/ais_track_splits.csv",
    "behavior/benign_train_manifest.parquet",
}

MCSHIPS_FILTER_RELS = {
    "annotations/coco_master.json",
    "audit/eo_annotations.csv",
    "audit/eo_images.csv",
    "splits/eo_image_splits.csv",
    "eval_slices/small_craft_eval_annotations.csv",
    "eval_slices/small_craft_eval_images.csv",
}

ROOT_META = [
    "LICENSE",
    "LICENSE-DATA",
    "CITATION.cff",
    ".zenodo.json",
    "docs/DATA_LICENSES.md",
    "docs/DUAL_USE.md",
]


def _forbidden_rel(rel: str) -> bool:
    """Return True if a staged relative path violates the redaction policy."""
    rel_posix = rel.replace("\\", "/")
    name = Path(rel_posix).name
    parts = rel_posix.split("/")
    if name == "tracks_ais_points.parquet":
        return True
    if name == "mcships.coco.json":
        return True
    if "patch_bank" in parts:
        return True
    if name in {"patch_optimized.png", "patch_init.png"} and "patch_core" in parts:
        return True
    if "envelopes" in parts or "weights" in parts:
        return True
    return False


def is_weight_path(path: Path) -> bool:
    return path.suffix.lower() in WEIGHT_SUFFIXES


def find_redacted(stage: Path) -> list[str]:
    """Return relative paths inside *stage* that violate the redaction gate."""
    bad: list[str] = []
    for p in stage.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(stage)).replace("\\", "/")
        if is_weight_path(p) or _forbidden_rel(rel):
            bad.append(rel)
    return sorted(bad)


def drop_mmsi(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".parquet":
        df = pd.read_parquet(src)
        if "mmsi" in df.columns:
            df = df.drop(columns=["mmsi"])
        df.to_parquet(dst, index=False)
    elif src.suffix == ".csv":
        df = pd.read_csv(src)
        if "mmsi" in df.columns:
            df = df.drop(columns=["mmsi"])
        df.to_csv(dst, index=False)
    else:
        shutil.copy2(src, dst)


def strip_mcships_coco(src: Path, dst: Path) -> None:
    payload = json.loads(src.read_text())
    keep_ids = {
        img["id"]
        for img in payload.get("images", [])
        if img.get("source") != "mcships"
    }
    payload["images"] = [img for img in payload["images"] if img["id"] in keep_ids]
    payload["annotations"] = [
        ann for ann in payload.get("annotations", []) if ann["image_id"] in keep_ids
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload) + "\n")


def strip_mcships_csv(src: Path, dst: Path) -> None:
    df = pd.read_csv(src)
    if "source" in df.columns:
        df = df[df["source"] != "mcships"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)


def stage_data_file(rel: str, data_dir: Path, stage: Path) -> None:
    src = data_dir / rel
    if not src.is_file():
        raise FileNotFoundError(f"missing derived artifact: {rel}")
    dst = stage / "data" / rel
    if rel in MMSI_DROP_RELS:
        drop_mmsi(src, dst)
    elif rel == "annotations/coco_master.json":
        strip_mcships_coco(src, dst)
    elif rel in MCSHIPS_FILTER_RELS:
        strip_mcships_csv(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_filtered_tree(src_root: Path, dst_root: Path, exts: set[str] | None) -> None:
    if not src_root.is_dir():
        return
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src_root)
        rel_s = str(rel).replace("\\", "/")
        if "patch_bank" in rel_s.split("/"):
            continue
        if is_weight_path(p):
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def stage_results(stage: Path) -> None:
    for rel in RESULTS_EXACT:
        src = REPO_ROOT / rel
        if not src.is_file():
            continue
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for tree_rel, exts in RESULTS_TREES:
        src = REPO_ROOT / tree_rel
        _copy_filtered_tree(src, stage / tree_rel, exts)


def stage_root_meta(stage: Path) -> None:
    for rel in ROOT_META:
        src = REPO_ROOT / rel
        if src.is_file():
            dst = stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_artifact_readme(stage: Path) -> None:
    text = """# Counter-USV — data & results artifact

Weight-free derived-data and evaluation bundle companion to the GitHub code
repository. Code DOI (GitHub tag via Zenodo) and this data DOI are cross-linked
once minted.

All work is **digital and simulated-physical only**. See `docs/DUAL_USE.md`
for the responsible-use statement (weights and patch banks withheld).

## Contents

- `data/` — annotations (McShips omitted), splits, derived track/behavior/defense
  features. Point-level AIS and raw imagery are **not** included.
- `results/` — freeze manifests, evaluation summaries, paper figures/captions.
- `LICENSE-DATA` — CC BY 4.0 for derived data in this deposit (see also
  `docs/DATA_LICENSES.md` for per-source terms).
- `RELEASE.json` / `CHECKSUMS.sha256` — inventory and digests.

## What is withheld

- **Model weights** (detector `*.pt`, defense envelope `*.joblib`) — dual-use;
  available upon reasonable request for bona fide defensive research.
- **Patch-bank tensors** — see `results/attacks/REDACTION.md`.
- **Point-level AIS** and **McShips** annotation exports.

Released trip/split/manifest tables have the `mmsi` column dropped (`trip_id`
retained).

## Verify

```bash
sha256sum -c CHECKSUMS.sha256
# or:
python scripts/release/build_bundle.py --verify path/to/stage
```

To retrain detectors or fit envelopes, clone the code repository, obtain source
imagery/AIS per `docs/DATA_LICENSES.md`, and follow `scripts/README.md`.
"""
    (stage / "ARTIFACT_README.md").write_text(text)


def hash_stage(stage: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(stage.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"CHECKSUMS.sha256", "RELEASE.json"}:
            continue
        rel = str(p.relative_to(stage)).replace("\\", "/")
        rows.append({"path": rel, "sha256": sha256(p), "bytes": p.stat().st_size})
    return rows


def write_checksums(stage: Path, rows: list[dict[str, Any]]) -> None:
    (stage / "CHECKSUMS.sha256").write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in rows)
    )


def write_release(stage: Path, rows: list[dict[str, Any]], *, freezes_ok: bool) -> dict:
    payload = {
        "schema": "counterusv.release.bundle/1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "policy": {
            "includes_model_weights": False,
            "includes_raw_imagery": False,
            "includes_bulk_ais": False,
            "includes_ais_points": False,
            "includes_mcships_annotations": False,
            "includes_patch_banks": False,
            "transforms": [
                "drop_mmsi_column",
                "strip_mcships_from_coco_master_and_audit_csvs",
            ],
        },
        "freezes_verified": freezes_ok,
        "n_files": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "checksums_file": "CHECKSUMS.sha256",
    }
    (stage / "RELEASE.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def make_archive(stage: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname=stage.name)


def verify_stage(stage: Path) -> int:
    checksums = stage / "CHECKSUMS.sha256"
    if not checksums.is_file():
        print(f"[verify] missing {checksums}", file=sys.stderr)
        return 1
    bad_redact = find_redacted(stage)
    if bad_redact:
        print("[verify] redaction gate FAILED:", file=sys.stderr)
        for p in bad_redact:
            print(f"  - {p}", file=sys.stderr)
        return 1
    mismatches = 0
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        path = path.strip()
        fp = stage / path
        if not fp.is_file():
            print(f"[verify] MISSING {path}", file=sys.stderr)
            mismatches += 1
            continue
        got = sha256(fp)
        if got != digest.strip():
            print(f"[verify] MISMATCH {path}", file=sys.stderr)
            mismatches += 1
    if mismatches:
        print(f"[verify] FAILED ({mismatches})", file=sys.stderr)
        return 1
    print(f"[verify] OK ({checksums})")
    return 0


def build(
    *,
    out_dir: Path,
    data_dir: Path,
    skip_verify_freezes: bool,
    skip_archive: bool,
) -> Path:
    stage = out_dir / STAGE_NAME
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    freezes_ok = False
    if not skip_verify_freezes:
        verify_freezes(skip=False)
        freezes_ok = True
    else:
        print("[bundle] skipping freeze digest verification")

    print(f"[bundle] staging {len(DERIVED_ARTIFACTS)} derived data files …")
    for rel in DERIVED_ARTIFACTS:
        stage_data_file(rel, data_dir, stage)

    print("[bundle] staging results …")
    stage_results(stage)
    stage_root_meta(stage)
    write_artifact_readme(stage)

    bad = find_redacted(stage)
    if bad:
        raise SystemExit(
            "redaction gate FAILED — staged tree contains forbidden paths:\n  - "
            + "\n  - ".join(bad)
        )

    rows = hash_stage(stage)
    write_checksums(stage, rows)
    release = write_release(stage, rows, freezes_ok=freezes_ok)
    # re-hash excluding RELEASE/CHECKSUMS already done; checksums file itself
    # is not self-hashed (same as package_derived).

    print(
        f"[bundle] staged {release['n_files']} files "
        f"({release['total_bytes'] / 1e6:.1f} MB) → {stage}"
    )

    if not skip_archive:
        archive = out_dir / "counterusv-data-results.tar.gz"
        make_archive(stage, archive)
        print(f"[bundle] wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    return stage


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory (default: dist/release)",
    )
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument(
        "--skip-verify-freezes",
        action="store_true",
        help="skip attack/defense freeze digest checks",
    )
    ap.add_argument(
        "--skip-archive",
        action="store_true",
        help="stage + checksum only (no tar.gz)",
    )
    ap.add_argument(
        "--verify",
        type=Path,
        nargs="?",
        const=DEFAULT_OUT / STAGE_NAME,
        help="verify an existing stage (checksums + no weights)",
    )
    args = ap.parse_args()

    if args.verify is not None:
        return verify_stage(args.verify.resolve())

    build(
        out_dir=args.out_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        skip_verify_freezes=args.skip_verify_freezes,
        skip_archive=args.skip_archive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
