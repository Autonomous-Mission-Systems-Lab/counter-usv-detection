#!/usr/bin/env python3
"""Extract observation-window kinematic features (scorer 3.3).

Policy (locked): **one last-W-seconds window per track per window length** —
no sliding / tiling. Sweep ``observation_window_sweep_s`` from
``configs/defense/scorer_features.yaml``. Also emits whole-track features as
the long-horizon limit (from ``data/tracks/tracks_ais.parquet``, same definitions).

Outputs under ``data/behavior/``:
  * ``features_window_{W}s.parquet`` — one row per trip_id
  * ``features_whole_track.parquet`` — whole-track scorer columns + identity
  * ``features_windows_summary.json``
  * ``results/behavior_model/windows_report.md``

Usage
-----
    python scripts/behavior/extract_windowed_features.py
    python scripts/behavior/extract_windowed_features.py --windows 300
    python scripts/behavior/extract_windowed_features.py --smoke   # 2k trips only
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

from counterusv.kinematics import extract_last_windows  # noqa: E402

DEFAULT_FEATURES = REPO_ROOT / "configs" / "defense" / "scorer_features.yaml"
DEFAULT_POINTS = REPO_ROOT / "data" / "tracks" / "tracks_ais_points.parquet"
DEFAULT_TRACKS = REPO_ROOT / "data" / "tracks" / "tracks_ais.parquet"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits" / "ais_track_splits.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "behavior"
DEFAULT_REPORT = REPO_ROOT / "results" / "behavior_model" / "windows_report.md"

SCORER_COLS = [
    "sog_mean", "sog_med", "sog_p95", "sog_max", "sog_std",
    "loiter_frac", "straightness",
    "accel_mean_abs", "accel_std",
    "turn_rate_mean_dps", "turn_rate_p95_dps", "cog_circ_std_deg",
]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def whole_track_table(tracks: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Whole-track features from the AIS summary parquet (+ split tags)."""
    id_cols = [
        "trip_id", "mmsi", "canonical_class", "role", "transceiver_class",
        "source", "n_points", "duration_s", "heading_avail_frac",
    ]
    keep = [c for c in id_cols + SCORER_COLS if c in tracks.columns]
    out = tracks[keep].copy()
    out["window_s"] = float("inf")  # sentinel: whole track
    out["window_complete"] = True
    out["span_s"] = out["duration_s"].astype("float64")
    sp = splits[["trip_id", "split", "region", "geo_cell", "start_day"]]
    out = out.merge(sp, on="trip_id", how="left")
    return out


def attach_meta(feat: pd.DataFrame, tracks: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    meta = tracks[[
        "trip_id", "mmsi", "canonical_class", "role", "transceiver_class", "source",
    ]].drop_duplicates("trip_id")
    sp = splits[["trip_id", "split", "region", "geo_cell", "start_day"]]
    out = feat.merge(meta, on="trip_id", how="left", suffixes=("", "_meta"))
    if "mmsi_meta" in out.columns:
        out["mmsi"] = out["mmsi"].fillna(out["mmsi_meta"])
        out = out.drop(columns=["mmsi_meta"])
    out = out.merge(sp, on="trip_id", how="left")
    return out


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Observation-window features")
    lines.append("")
    lines.append(f"Generated: {summary['timestamp']}")
    lines.append("")
    lines.append(
        "Partial-history kinematic features for the consistency scorer. "
        "**Policy:** one **last-W-seconds** window per track per length "
        "(no sliding/tiling). Whole-track features kept as the long-horizon "
        "limit. Spec: `configs/defense/scorer_features.yaml`."
    )
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    pol = summary["window_policy"]
    lines.append(f"- placement: **{pol.get('placement')}**")
    lines.append(f"- per track per length: **{pol.get('per_track_per_length')}**")
    lines.append(f"- min_points: {pol.get('min_points')}")
    lines.append(f"- min_span_frac: {pol.get('min_span_frac')}")
    lines.append(f"- incomplete: {pol.get('incomplete_policy')}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| window_s | rows | complete | complete % | median n_points | median span_s |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for w, block in summary["windows"].items():
        lines.append(
            f"| {w} | {block['n_rows']:,} | {block['n_complete']:,} | "
            f"{100 * block['complete_frac']:.1f}% | "
            f"{block['median_n_points']:.0f} | {block['median_span_s']:.0f} |"
        )
    wt = summary.get("whole_track") or {}
    lines.append(
        f"| whole | {wt.get('n_rows', 0):,} | {wt.get('n_rows', 0):,} | "
        f"100% | — | — |"
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for p in summary.get("artifacts") or []:
        lines.append(f"- `{p}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    ap.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--windows", type=float, nargs="+", default=None,
                    help="override sweep (seconds); default from feature spec")
    ap.add_argument("--smoke", action="store_true",
                    help="limit to 2000 trip_ids for a fast wiring check")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = _load_yaml(args.features)
    pol = spec.get("window_policy") or {}
    windows = args.windows or list(spec.get("observation_window_sweep_s") or [300])
    min_points = int(pol.get("min_points", 3))
    min_span_frac = float(pol.get("min_span_frac", 0.5))

    print(f"[windows] lengths={windows}  min_points={min_points}  "
          f"min_span_frac={min_span_frac}")
    print(f"[windows] loading points {args.points} …")
    t0 = time.perf_counter()
    cols = ["trip_id", "t", "lat", "lon", "sog", "cog", "heading"]
    # mmsi is on tracks, not always on points — check
    import pyarrow.parquet as pq
    schema_names = set(pq.read_schema(args.points).names)
    if "mmsi" in schema_names:
        cols.append("mmsi")
    points = pd.read_parquet(args.points, columns=cols)
    print(f"[windows] points={len(points):,} in {time.perf_counter()-t0:.1f}s")

    tracks = pd.read_parquet(args.tracks)
    splits = pd.read_csv(args.splits)

    if args.smoke:
        keep_ids = set(tracks["trip_id"].drop_duplicates().head(2000))
        points = points.loc[points["trip_id"].isin(keep_ids)]
        tracks = tracks.loc[tracks["trip_id"].isin(keep_ids)]
        print(f"[windows] smoke: {len(keep_ids)} trips / {len(points):,} points")

    args.out.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    window_stats: dict[str, Any] = {}

    # --- whole track ---
    whole = whole_track_table(tracks, splits)
    whole_path = args.out / "features_whole_track.parquet"
    if not args.dry_run:
        whole.to_parquet(whole_path, index=False)
        artifacts.append(str(whole_path.relative_to(REPO_ROOT)))
        print(f"[windows] wrote {whole_path.name} ({len(whole):,} rows)")

    # --- per window length ---
    print(f"[windows] extracting last-W windows …")
    t1 = time.perf_counter()
    feat_all = extract_last_windows(
        points, windows, min_points=min_points, min_span_frac=min_span_frac)
    print(f"[windows] extracted {len(feat_all):,} rows in "
          f"{time.perf_counter()-t1:.1f}s")

    if feat_all.empty and not args.dry_run:
        raise RuntimeError("No windowed features produced")

    if not feat_all.empty:
        feat_all = attach_meta(feat_all, tracks, splits)

    for w in windows:
        w = float(w)
        sub = feat_all.loc[feat_all["window_s"] == w].copy() if not feat_all.empty else feat_all
        n_complete = int(sub["window_complete"].sum()) if len(sub) else 0
        window_stats[str(int(w) if w == int(w) else w)] = {
            "n_rows": int(len(sub)),
            "n_complete": n_complete,
            "complete_frac": (n_complete / len(sub)) if len(sub) else 0.0,
            "median_n_points": float(sub["n_points"].median()) if len(sub) else None,
            "median_span_s": float(sub["span_s"].median()) if len(sub) else None,
        }
        out_path = args.out / f"features_window_{int(w)}s.parquet"
        if not args.dry_run and len(sub):
            sub.to_parquet(out_path, index=False)
            artifacts.append(str(out_path.relative_to(REPO_ROOT)))
            print(f"[windows] wrote {out_path.name} ({len(sub):,} rows, "
                  f"{n_complete:,} complete)")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_spec": str(args.features.relative_to(REPO_ROOT)),
        "window_policy": pol,
        "windows_s": [float(w) for w in windows],
        "windows": window_stats,
        "whole_track": {"n_rows": int(len(whole))},
        "smoke": bool(args.smoke),
        "artifacts": artifacts,
        "n_points_input": int(len(points)),
        "n_trips_input": int(points["trip_id"].nunique()),
    }
    if args.dry_run:
        print("[windows] dry-run — no files written")
        print(json.dumps(summary, indent=2))
        return 0

    sum_path = args.out / "features_windows_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    artifacts.append(str(sum_path.relative_to(REPO_ROOT)))
    summary["artifacts"] = artifacts
    write_report(args.report, summary)
    print(f"[windows] wrote {sum_path}")
    print(f"[windows] wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
