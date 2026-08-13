#!/usr/bin/env python3
"""Materialize the adversary-motion sweep and freeze the grid digest.

Builds the sweep cell table, SHA-256 freezes it (no DDR numbers), then
generates thinned world-frame point streams under results/adversary_motion/.
Never writes into data/behavior or data/defense training tables.

Usage
-----
    python scripts/attacks/generate_adversary_tracks.py
    python scripts/attacks/generate_adversary_tracks.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.kinematics import (  # noqa: E402
    build_sweep_cells,
    cells_to_frame,
    digest_cells_frame,
    freeze_sweep_manifest,
    generate_two_phase_track,
    load_kinematics_config,
    load_placements,
    platforms_from_config,
    refuse_benign_corpus_write,
)


def _smoke_cfg(cfg: dict) -> dict:
    """Tiny grid for wiring checks — not the frozen headline sweep."""
    cfg = dict(cfg)
    cfg["sweep"] = {
        "v_mimic_kn": [8, 25],
        "commit_range_nm": [2.0],
        "bearing_offset_deg": [0, 30],
        "mimicked_classes": ["fishing", "recreational"],
        "platforms": ["magura_v5"],
        "include_unconstrained": True,
    }
    cfg["include_far_only"] = False
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid + one asset (wiring only; do not freeze as headline)")
    ap.add_argument("--max-assets", type=int, default=None,
                    help="limit number of fit-population assets")
    ap.add_argument("--skip-points", action="store_true",
                    help="freeze cell table only (no track generation)")
    args = ap.parse_args()

    cfg = load_kinematics_config(args.config)
    if args.smoke:
        cfg = _smoke_cfg(cfg)

    out = cfg.get("output") or {}
    cells_rel = out.get("cells") or "results/adversary_motion/sweep_cells.parquet"
    points_rel = out.get("points") or "results/adversary_motion/tracks_points.parquet"
    freeze_rel = out.get("freeze") or "results/adversary_motion/FROZEN_SWEEP.json"
    cells_path = REPO_ROOT / cells_rel
    points_path = REPO_ROOT / points_rel
    freeze_path = REPO_ROOT / freeze_rel
    for p in (cells_path, points_path, freeze_path):
        refuse_benign_corpus_write(p)

    print("[adv-motion] loading placements …")
    assets, placements_sha = load_placements(cfg)
    if "role" in assets.columns and not cfg.get("include_far_only", False):
        roles = set(cfg.get("headline_placement_roles") or ["fit"])
        assets = assets.loc[assets["role"].isin(roles)].copy()
    if "valid" in assets.columns:
        assets = assets.loc[assets["valid"]].copy()
    if args.smoke:
        assets = assets.head(1)
    elif args.max_assets is not None:
        assets = assets.head(int(args.max_assets))
    print(f"[adv-motion] assets: {len(assets)}  placements_sha={placements_sha[:16]}…")

    print("[adv-motion] building sweep cells …")
    cells = build_sweep_cells(cfg, assets)
    cells_df = cells_to_frame(cells)
    # Annotate plausibility on the cell table (platform peak speed).
    plats = platforms_from_config(cfg)
    peaks = []
    in_band = []
    for c in cells:
        peak = c.peak_speed_kn(plats[c.platform])
        peaks.append(peak)
        from counterusv.attacks.kinematics import in_plausibility_band
        in_band.append(in_plausibility_band(peak, cfg=cfg))
    cells_df["peak_speed_kn"] = peaks
    cells_df["in_plausibility_band"] = in_band
    cells_df["extrapolation"] = ~cells_df["in_plausibility_band"]

    cells_path.parent.mkdir(parents=True, exist_ok=True)
    cells_df.to_parquet(cells_path, index=False)
    cell_sha = digest_cells_frame(cells_df)
    print(f"[adv-motion] cells: n={len(cells_df):,}  sha256={cell_sha[:16]}…")
    print(f"[adv-motion] wrote {cells_path.relative_to(REPO_ROOT)}")

    if args.smoke:
        # Smoke must not overwrite the headline freeze with a tiny grid.
        smoke_freeze = freeze_path.with_name("FROZEN_SWEEP_smoke.json")
        freeze_sweep_manifest(
            cfg=cfg, cells=cells_df, placements_sha=placements_sha,
            out_path=smoke_freeze,
        )
        print(f"[adv-motion] smoke freeze → {smoke_freeze.relative_to(REPO_ROOT)}")
    else:
        freeze = freeze_sweep_manifest(
            cfg=cfg, cells=cells_df, placements_sha=placements_sha,
            out_path=freeze_path,
        )
        print(f"[adv-motion] FROZEN sweep digest={freeze['digests']['sweep_cells'][:16]}…")
        print(f"[adv-motion] wrote {freeze_path.relative_to(REPO_ROOT)}")

    if args.skip_points:
        print("[adv-motion] --skip-points: done")
        return 0

    asset_xy = {
        str(r.asset_id): (float(r.lat), float(r.lon))
        for r in assets.itertuples()
    }
    print("[adv-motion] generating tracks …")
    t0 = time.perf_counter()
    point_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []
    for i, cell in enumerate(cells):
        lat, lon = asset_xy[str(cell.asset_id)]
        track = generate_two_phase_track(
            lat, lon, cell, plats[cell.platform], cfg=cfg, trip_id=cell.cell_id,
        )
        pts = track.points.copy()
        pts["cell_id"] = cell.cell_id
        pts["asset_id"] = cell.asset_id
        pts["mimicked_class"] = cell.mimicked_class
        pts["platform"] = cell.platform
        pts["negative_control"] = cell.negative_control
        pts["unconstrained"] = cell.unconstrained
        point_parts.append(pts)
        meta_rows.append(track.meta)
        if (i + 1) % 200 == 0 or (i + 1) == len(cells):
            print(f"  … {i + 1}/{len(cells)}")

    points = pd.concat(point_parts, ignore_index=True)
    refuse_benign_corpus_write(points_path)
    points.to_parquet(points_path, index=False)
    meta_path = points_path.with_name("tracks_meta.json")
    meta_path.write_text(json.dumps(meta_rows, indent=2, default=str) + "\n")
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"[adv-motion] wrote {points_path.relative_to(REPO_ROOT)}  "
          f"n_points={len(points):,}  n_trips={points['trip_id'].nunique():,}")
    print(f"[adv-motion] wrote {meta_path.relative_to(REPO_ROOT)}")
    print(f"[adv-motion] done in {elapsed}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
