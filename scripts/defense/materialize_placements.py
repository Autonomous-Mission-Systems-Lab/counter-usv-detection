#!/usr/bin/env python3
"""Materialize defended-asset placements from AIS traffic + engagement policy.

Writes ``data/defense/placements.parquet``, a SHA-256 digest pin, and a short
report. Digests the table *before* any envelope fit or FAR number is computed.

Usage
-----
    python scripts/defense/materialize_placements.py
    python scripts/defense/materialize_placements.py --smoke
    python scripts/defense/materialize_placements.py --seed miami_approach
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402
from counterusv.defense.placements import materialize_placements  # noqa: E402

DEFAULT_POINTS = REPO_ROOT / "data" / "tracks" / "tracks_ais_points.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "defense"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "defense" / "engagement_geometry.yaml"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Asset placements",
        "",
        f"Rows: **{len(df)}** · valid: **{int(df['valid'].sum())}** · "
        f"invalid: **{int((~df['valid']).sum())}**",
        "",
        "| asset_id | role | lat | lon | inbound_legs | min_range_nm | valid | reject |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| `{r.asset_id}` | {r.role} | {r.lat:.4f} | {r.lon:.4f} | "
            f"{int(r.n_inbound_legs_in_annulus)} | {r.min_observed_range_nm:.2f} | "
            f"{bool(r.valid)} | {r.reject_reason or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=str, default=None,
                    help="Restrict to one seed port id")
    ap.add_argument("--smoke", action="store_true",
                    help="One seed (miami_approach) for fast iteration")
    ap.add_argument("--search-nm", type=float, default=None,
                    help="Override materialize_search_nm")
    args = ap.parse_args()

    cfg = load_engagement_geometry(args.config)
    seeds = cfg.port_regions()
    if args.smoke:
        seeds = [s for s in seeds if s.id == "miami_approach"] or seeds[:1]
    elif args.seed:
        seeds = [s for s in seeds if s.id == args.seed]
        if not seeds:
            print(f"unknown seed: {args.seed}", file=sys.stderr)
            return 2

    print(f"loading points from {args.points} …")
    cols = ["trip_id", "t", "lat", "lon", "sog"]
    points = pd.read_parquet(args.points, columns=cols)
    print(f"  {len(points):,} points · seeds={[s.id for s in seeds]}")

    df = materialize_placements(
        points, cfg, seeds=seeds, search_nm=args.search_nm,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "placements.parquet"
    df.to_parquet(parquet, index=False)
    report = out_dir / "placements_report.md"
    _write_report(df, report)

    digest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "n_rows": int(len(df)),
        "n_valid": int(df["valid"].sum()),
        "seeds": [s.id for s in seeds],
        "fit_population": cfg.fit_population(),
        "digests": {
            "placements.parquet": _sha256_file(parquet),
            "engagement_geometry.yaml": _sha256_file(Path(args.config)),
        },
        "paths": {
            "placements": str(parquet.relative_to(REPO_ROOT)),
            "report": str(report.relative_to(REPO_ROOT)),
            "config": str(Path(args.config).resolve().relative_to(REPO_ROOT)
                          if Path(args.config).is_absolute()
                          else args.config),
        },
    }
    digest_path = out_dir / "placements_digest.json"
    digest_path.write_text(json.dumps(digest, indent=2) + "\n")
    print(f"wrote {parquet} ({len(df)} rows, {digest['n_valid']} valid)")
    print(f"wrote {digest_path}")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
