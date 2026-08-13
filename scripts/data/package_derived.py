#!/usr/bin/env python3
"""Package and checksum the derived data release (no raw imagery / bulk AIS).

Writes:
  * ``data/CHECKSUMS.derived.sha256`` — sha256 of every redistributable derived artifact
  * ``data/RELEASE.json`` — version stamp, counts, and the file inventory

Does **not** package ``data/raw/`` imagery or bulk AIS feeds. The curated USV
provenance manifest is included; image pixels are not.

Usage
-----
    python scripts/data/package_derived.py
    python scripts/data/package_derived.py --data-dir data --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Relative to data_dir. Order is the release inventory order.
# Excludes: point-level AIS (cleaned feed copy, not a derived feature set);
# McShips annotations (no stated license — omit from permissive slice).
DERIVED_ARTIFACTS = [
    # Public contracts / cards
    "taxonomy.yaml",
    "HARMONIZATION.md",
    "DATACARD.md",
    "DATACARD_EO.md",
    "DATACARD_TRACKS.md",
    # EO annotations (McShips omitted — regenerate locally via fetch_data)
    "annotations/coco_master.json",
    "annotations/convert_summary.json",
    "annotations/seaships.coco.json",
    "annotations/aboships.coco.json",
    "annotations/smd.coco.json",
    "annotations/usv.coco.json",
    # Audit + eval manifests
    "audit/eo_annotations.csv",
    "audit/eo_images.csv",
    "audit/eo_audit_summary.json",
    "eval_slices/small_craft_eval_annotations.csv",
    "eval_slices/small_craft_eval_images.csv",
    "eval_slices/small_craft_eval_summary.json",
    # Splits
    "splits/eo_image_splits.csv",
    "splits/ais_track_splits.csv",
    "splits/splits_summary.json",
    # Trajectory features (derived; not raw AIS / not point-level feeds)
    "tracks/tracks_ais.parquet",
    # Behavior-model inputs (windowed kinematics + train manifest)
    "behavior/features_window_60s.parquet",
    "behavior/features_window_120s.parquet",
    "behavior/features_window_180s.parquet",
    "behavior/features_window_300s.parquet",
    "behavior/features_window_600s.parquet",
    "behavior/features_whole_track.parquet",
    "behavior/benign_train_manifest.parquet",
    "behavior/benign_corpus_summary.json",
    "behavior/features_windows_summary.json",
    "behavior/envelope_coverage.json",
    # Defense geometry features + placements
    "defense/features_geometry_window_120s.parquet",
    "defense/features_geometry_window_180s.parquet",
    "defense/features_geometry_window_300s.parquet",
    "defense/features_geometry_window_600s.parquet",
    "defense/placements.parquet",
    "defense/placements_digest.json",
    "defense/placements_report.md",
    "defense/geometry_coverage_report.json",
    "defense/geometry_coverage_report.md",
    # USV provenance only (pixels stay under raw/ and are not redistributed)
    "raw/usv/manifest.csv",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--strict", action="store_true",
                    help="fail if any expected derived artifact is missing")
    args = ap.parse_args()
    data_dir: Path = args.data_dir

    rows = []
    missing = []
    for rel in DERIVED_ARTIFACTS:
        path = data_dir / rel
        if not path.is_file():
            missing.append(rel)
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": rel, "sha256": digest, "bytes": size})

    if missing:
        msg = f"[package] missing {len(missing)} derived artifact(s):\n  - " + "\n  - ".join(missing)
        if args.strict:
            print(msg, file=sys.stderr)
            return 1
        print(msg + "\n[package] continuing (omit --strict to allow partial).")

    # sha256sum-compatible lines (hash  two-spaces  path)
    checksum_path = data_dir / "CHECKSUMS.derived.sha256"
    checksum_path.write_text(
        "".join(f"{r['sha256']}  {r['path']}\n" for r in rows))

    convert = load_json(data_dir / "annotations" / "convert_summary.json") or {}
    splits = load_json(data_dir / "splits" / "splits_summary.json") or {}
    audit = load_json(data_dir / "audit" / "eo_audit_summary.json") or {}
    master = convert.get("master", {})
    totals = audit.get("totals", {})

    version = {
        "version": f"{date.today().isoformat()}+{_git_sha()}",
        "date": date.today().isoformat(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
    }
    release = {
        "version": version,
        "policy": {
            "includes_raw_imagery": False,
            "includes_bulk_ais": False,
            "includes_usv_pixels": False,
            "includes_ais_points": False,
            "includes_mcships_annotations": False,
            "includes_model_weights": False,
            "redistributes": (
                "annotations (McShips omitted), manifests, derived track/behavior/"
                "defense features, taxonomy, contracts"
            ),
        },
        "counts": {
            "eo_images": master.get("images"),
            "eo_annotations": master.get("annotations"),
            "eo_per_class": master.get("per_class"),
            "detector_eligible": totals.get("detector_eligible"),
            "patch_eligible": totals.get("patch_eligible"),
            "eo_split": (splits.get("eo") or {}).get("by_split"),
            "ais_tracks": (splits.get("ais") or {}).get("tracks"),
        },
        "artifacts": rows,
        "n_artifacts": len(rows),
        "n_missing": len(missing),
        "missing": missing,
        "checksums_file": "CHECKSUMS.derived.sha256",
        "data_cards": ["DATACARD.md", "DATACARD_EO.md", "DATACARD_TRACKS.md"],
    }
    release_path = data_dir / "RELEASE.json"
    release_path.write_text(json.dumps(release, indent=2) + "\n")

    # Re-hash card/checksums are already in rows for DATACARD; RELEASE.json is the
    # stamp itself (not self-hashed). Append a note file size for the stamp.
    total_bytes = sum(r["bytes"] for r in rows)
    print(f"[package] version {version['version']}")
    print(f"[package] wrote {checksum_path.relative_to(data_dir)} "
          f"({len(rows)} files, {total_bytes / 1e6:.1f} MB)")
    print(f"[package] wrote {release_path.relative_to(data_dir)}")
    if master:
        print(f"[package] EO master: {master.get('images')} imgs / "
              f"{master.get('annotations')} boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
