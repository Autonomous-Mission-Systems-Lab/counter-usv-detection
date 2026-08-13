#!/usr/bin/env python3
"""Extract asset-relative geometry features for encounter-paired windows.

Region-only pairing against the digested placements table. Emits one parquet
per observation-window length under ``data/defense/``, leaving the kinematics
window tables untouched. Writes a coverage report.

Usage
-----
    python scripts/defense/extract_geometry_features.py
    python scripts/defense/extract_geometry_features.py --smoke
    python scripts/defense/extract_geometry_features.py --windows 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402
from counterusv.defense.geometry_features import (  # noqa: E402
    GEOMETRY_FEATURE_KEYS,
    geometry_features_from_points,
    range_bearing_nm,
)
from counterusv.kinematics.features import (  # noqa: E402
    haversine_km,
    last_window_mask,
)

DEFAULT_POINTS = REPO_ROOT / "data" / "tracks" / "tracks_ais_points.parquet"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits" / "ais_track_splits.csv"
DEFAULT_PLACEMENTS = REPO_ROOT / "data" / "defense" / "placements.parquet"
DEFAULT_FEATURES_CFG = REPO_ROOT / "configs" / "defense" / "scorer_features.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "defense"
NM_PER_KM = 1.0 / 1.852


def _load_windows(cfg_path: Path) -> tuple[list[float], dict]:
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    windows = [float(w) for w in raw.get("observation_window_sweep_s", [300])]
    policy = dict(raw.get("window_policy") or {})
    return windows, policy


def _seed_approx(cfg) -> dict[str, tuple[float, float]]:
    return {p.id: (p.approx_lat, p.approx_lon) for p in cfg.port_regions()}


def _tracks_near_seeds(
    splits: pd.DataFrame,
    seed_approx: dict[str, tuple[float, float]],
    pairing_radius_nm: float,
) -> pd.DataFrame:
    """Annotate each track with the seed(s) whose approx it falls near."""
    lat = splits["mean_lat"].to_numpy(dtype="float64")
    lon = splits["mean_lon"].to_numpy(dtype="float64")
    assigned: list[dict] = []
    for seed_id, (slat, slon) in seed_approx.items():
        r = haversine_km(slat, slon, lat, lon) * NM_PER_KM
        mask = r <= pairing_radius_nm
        if not mask.any():
            continue
        sub = splits.loc[mask, [
            "trip_id", "split", "role", "canonical_class", "mean_lat", "mean_lon",
        ]].copy()
        sub["port_region"] = seed_id
        sub["dist_to_seed_nm"] = r[mask]
        assigned.append(sub)
    if not assigned:
        return pd.DataFrame()
    # One row per (trip, seed); a track near two seeds may pair with both.
    return pd.concat(assigned, ignore_index=True)


def extract_for_window(
    points: pd.DataFrame,
    track_seeds: pd.DataFrame,
    placements: pd.DataFrame,
    window_s: float,
    *,
    annulus: dict,
    inbound_leg: dict,
    min_points: int,
    min_span_frac: float,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Emit geometry feature rows for one window length."""
    place = placements.copy()
    if valid_only:
        place = place.loc[place["valid"]].copy()
    if place.empty or track_seeds.empty:
        return pd.DataFrame()

    # Restrict points to candidate trips.
    trip_ids = track_seeds["trip_id"].unique()
    pts = points.loc[points["trip_id"].isin(trip_ids)].copy()
    if pts.empty:
        return pd.DataFrame()
    pts = pts.sort_values(["trip_id", "t"], kind="mergesort").reset_index(drop=True)
    mask = last_window_mask(pts, window_s)
    win = pts.loc[mask]
    if win.empty:
        return pd.DataFrame()

    # Completeness by trip (same rule as kinematics windows).
    g = win.groupby("trip_id", sort=False)
    n_pts = g.size()
    span = g["t"].max() - g["t"].min()
    complete = (n_pts >= min_points) & (span >= min_span_frac * window_s)

    meta = track_seeds.merge(
        complete.rename("window_complete").reset_index(),
        on="trip_id", how="left",
    )
    meta["window_complete"] = meta["window_complete"].fillna(False)

    rows: list[dict] = []
    place_by_seed = {
        seed: gdf for seed, gdf in place.groupby("port_region", sort=False)
    }
    win_by_trip = {tid: gdf for tid, gdf in win.groupby("trip_id", sort=False)}
    max_r = float(annulus.get("max_range_nm", 6.0))

    for _, tr in meta.iterrows():
        seed = tr["port_region"]
        assets = place_by_seed.get(seed)
        if assets is None or assets.empty:
            continue
        sub = win_by_trip.get(tr["trip_id"])
        if sub is None or len(sub) < int(inbound_leg.get("min_points", 4)):
            continue
        slat = sub["lat"].to_numpy(dtype="float64")
        slon = sub["lon"].to_numpy(dtype="float64")
        for _, asset in assets.iterrows():
            # Cheap reject: no window point within max range.
            rr, _ = range_bearing_nm(slat, slon, float(asset["lat"]), float(asset["lon"]))
            if not (rr <= max_r).any():
                continue
            feat = geometry_features_from_points(
                sub[["t", "lat", "lon"]],
                float(asset["lat"]), float(asset["lon"]),
                annulus=annulus, inbound_leg=inbound_leg,
            )
            if feat is None:
                continue
            row = {
                "trip_id": int(tr["trip_id"]),
                "window_s": float(window_s),
                "window_complete": bool(tr["window_complete"]),
                "asset_id": asset["asset_id"],
                "port_region": seed,
                "placement_class": asset["placement_class"],
                "role": asset["role"],
                "split": tr.get("split"),
                "track_role": tr.get("role"),
                "canonical_class": tr.get("canonical_class"),
            }
            row.update({k: feat[k] for k in GEOMETRY_FEATURE_KEYS})
            rows.append(row)
    return pd.DataFrame(rows)


def _coverage_report(
    frames: dict[float, pd.DataFrame],
    track_seeds: pd.DataFrame,
    fit_population: list[str],
    out_md: Path,
    out_json: Path,
) -> dict:
    summary: dict = {"windows": {}, "fit_population": fit_population}
    lines = [
        "# Geometry feature coverage",
        "",
        "Encounter-paired asset-relative features. Tracks with no usable "
        "encounter stay kinematics-only (not listed here).",
        "",
    ]
    for w, df in sorted(frames.items()):
        key = f"{int(w)}s"
        n = len(df)
        train = df.loc[df["split"] == "train"] if n else df
        fit = train.loc[train["placement_class"].isin(fit_population)] if n else train
        thin = []
        if n and len(fit):
            counts = (
                fit.groupby(["canonical_class", "port_region", "placement_class"])
                .size()
                .reset_index(name="n")
            )
            thin = counts.loc[counts["n"] < 30].to_dict(orient="records")
        cell = {
            "n_rows": int(n),
            "n_train": int(len(train)),
            "n_train_fit_population": int(len(fit)),
            "n_unique_trips": int(df["trip_id"].nunique()) if n else 0,
            "n_candidate_trips_near_seeds": int(track_seeds["trip_id"].nunique()),
            "thin_cells_n_lt_30": thin,
        }
        if n and len(train):
            cell["frac_candidate_trips_with_geometry"] = (
                train["trip_id"].nunique() / max(track_seeds.loc[
                    track_seeds["split"] == "train", "trip_id"
                ].nunique(), 1)
            )
        summary["windows"][key] = cell
        lines.append(f"## Window {key}")
        lines.append("")
        lines.append(f"- Rows: **{cell['n_rows']}**")
        lines.append(f"- Train rows: **{cell['n_train']}** "
                     f"(fit-population: **{cell['n_train_fit_population']}**)")
        lines.append(f"- Unique trips: **{cell['n_unique_trips']}** / "
                     f"{cell['n_candidate_trips_near_seeds']} near seeds")
        if thin:
            lines.append(f"- Thin fit cells (n<30): **{len(thin)}**")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--placements", type=Path, default=DEFAULT_PLACEMENTS)
    ap.add_argument("--features-config", type=Path, default=DEFAULT_FEATURES_CFG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--windows", type=float, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="One window (300s), miami_approach placements only, "
                         "cap candidate trips")
    ap.add_argument("--include-invalid-placements", action="store_true")
    args = ap.parse_args()

    if not args.placements.is_file():
        print(f"placements not found: {args.placements}\n"
              f"Run scripts/defense/materialize_placements.py first.",
              file=sys.stderr)
        return 2

    eng = load_engagement_geometry()
    windows, policy = _load_windows(args.features_config)
    if args.windows:
        windows = [float(w) for w in args.windows]
    if args.smoke:
        windows = [300.0]

    pairing_r = float(eng.placement_policy.get("pairing_radius_nm", 100))
    placements = pd.read_parquet(args.placements)
    if args.smoke:
        placements = placements.loc[
            placements["port_region"] == "miami_approach"
        ].copy()

    splits = pd.read_csv(args.splits)
    track_seeds = _tracks_near_seeds(splits, _seed_approx(eng), pairing_r)
    if args.smoke and not track_seeds.empty:
        # Cap for fast iteration.
        keep = track_seeds["trip_id"].drop_duplicates().head(500)
        track_seeds = track_seeds.loc[track_seeds["trip_id"].isin(keep)]

    print(f"candidate (trip,seed) pairs: {len(track_seeds):,} · "
          f"placements: {len(placements)} · windows: {windows}")

    print(f"loading points from {args.points} …")
    points = pd.read_parquet(
        args.points, columns=["trip_id", "t", "lat", "lon", "sog", "cog"],
    )
    # Restrict early.
    points = points.loc[points["trip_id"].isin(track_seeds["trip_id"].unique())]
    print(f"  {len(points):,} points for candidate trips")

    min_points = int(policy.get("min_points", 2))
    min_span_frac = float(policy.get("min_span_frac", 0.5))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[float, pd.DataFrame] = {}
    for w in windows:
        print(f"extracting window={w}s …")
        df = extract_for_window(
            points, track_seeds, placements, w,
            annulus=eng.annulus, inbound_leg=eng.inbound_leg,
            min_points=min_points, min_span_frac=min_span_frac,
            valid_only=not args.include_invalid_placements,
        )
        frames[w] = df
        out = args.out_dir / f"features_geometry_window_{int(w)}s.parquet"
        df.to_parquet(out, index=False)
        print(f"  wrote {out} ({len(df):,} rows)")

    cov = _coverage_report(
        frames, track_seeds, eng.fit_population(),
        args.out_dir / "geometry_coverage_report.md",
        args.out_dir / "geometry_coverage_report.json",
    )
    print(f"coverage: {json.dumps(cov['windows'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
