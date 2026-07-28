#!/usr/bin/env python3
"""Assemble the benign training corpus + report retained counts (scorer 3.1).

Joins ``data/tracks/tracks_ais.parquet`` to ``data/splits/ais_track_splits.csv``, keeps
the train split with ``role==benign`` only (hard-excludes hostile / non_target /
``usv``), and writes:

  * ``data/behavior/benign_train_manifest.parquet`` — training corpus
  * ``data/behavior/benign_corpus_summary.json`` — retained / excluded counts
  * ``results/behavior_model/corpus_report.md`` — human summary

The feature contract lives in ``configs/defense/scorer_features.yaml`` (frozen
alongside this corpus). This script does not fit a model — it only freezes
*who* trains and *which* columns the scorer may see.

Usage
-----
    python scripts/behavior/build_benign_corpus.py
    python scripts/behavior/build_benign_corpus.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = REPO_ROOT / "configs" / "defense" / "scorer_features.yaml"
DEFAULT_OUT_DATA = REPO_ROOT / "data" / "behavior"
DEFAULT_OUT_RESULTS = REPO_ROOT / "results" / "behavior_model"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def feature_columns(spec: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (core cols, course cols, all feature cols) from the frozen spec."""
    core = [f["name"] for f in (spec.get("core") or [])]
    course = [f["name"] for f in (spec.get("course") or [])]
    return core, course, core + course


def assemble(
    tracks: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    keep_roles: list[str],
    train_split: str = "train",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join tracks↔splits; return (benign_train, summary dict)."""
    split_cols = ["trip_id", "split", "region", "geo_cell", "start_day"]
    missing = [c for c in split_cols if c not in splits.columns]
    if missing:
        raise KeyError(f"ais_track_splits missing columns: {missing}")

    # Prefer track identity columns; pull split provenance from the split file.
    drop_from_tracks = [c for c in ("split", "region", "geo_cell", "start_day")
                        if c in tracks.columns]
    base = tracks.drop(columns=drop_from_tracks) if drop_from_tracks else tracks
    merged = base.merge(splits[split_cols], on="trip_id", how="inner",
                        validate="one_to_one")
    if len(merged) != len(tracks):
        raise RuntimeError(
            f"tracks↔splits join size mismatch: tracks={len(tracks)} "
            f"joined={len(merged)} (expected 1:1 on trip_id)"
        )

    # Firewall attestation inputs.
    excluded = merged.loc[
        (merged["split"] == train_split) & (~merged["role"].isin(keep_roles))
    ]
    excluded_by_role = excluded["role"].value_counts().to_dict()
    excluded_by_class = excluded["canonical_class"].value_counts().to_dict()

    benign_train = merged.loc[
        (merged["split"] == train_split) & (merged["role"].isin(keep_roles))
    ].copy()

    # Refuse usv source if it ever appears (AIS currently has none).
    if "source" in benign_train.columns:
        n_usv = int((benign_train["source"] == "usv").sum())
        if n_usv:
            raise RuntimeError(
                f"Firewall violation: {n_usv} source==usv rows in benign train"
            )

    per_class = (
        benign_train.groupby("canonical_class").size()
        .sort_values(ascending=False).to_dict()
    )
    per_tx = (
        benign_train.groupby("transceiver_class").size().to_dict()
        if "transceiver_class" in benign_train.columns else {}
    )
    per_class_tx_df = (
        benign_train.groupby(["canonical_class", "transceiver_class"]).size()
        .unstack(fill_value=0)
        if "transceiver_class" in benign_train.columns
        else None
    )
    per_class_tx: dict[str, dict[str, int]] = {}
    if per_class_tx_df is not None:
        for cls, row in per_class_tx_df.iterrows():
            per_class_tx[str(cls)] = {
                str(tx): int(row[tx]) for tx in per_class_tx_df.columns
            }

    # Held-out benign counts (not in the training corpus; for later FAR).
    benign_held = {}
    for sp in ("val", "test"):
        bh = merged.loc[
            (merged["split"] == sp) & (merged["role"].isin(keep_roles))
        ]
        benign_held[sp] = {
            "n": int(len(bh)),
            "per_class": bh.groupby("canonical_class").size()
            .sort_values(ascending=False).to_dict(),
        }

    # split × role contingency for the attestation.
    split_role = (
        merged.groupby(["split", "role"]).size().unstack(fill_value=0)
    )
    split_role_counts = {
        str(sp): {str(role): int(split_role.loc[sp, role])
                  for role in split_role.columns}
        for sp in split_role.index
    }

    summary: dict[str, Any] = {
        "n_tracks_total": int(len(merged)),
        "n_train_total": int((merged["split"] == train_split).sum()),
        "n_benign_train": int(len(benign_train)),
        "n_excluded_train": int(len(excluded)),
        "excluded_train_by_role": {str(k): int(v) for k, v in excluded_by_role.items()},
        "excluded_train_by_class": {
            str(k): int(v) for k, v in excluded_by_class.items()
        },
        "benign_train_per_class": {str(k): int(v) for k, v in per_class.items()},
        "benign_train_per_transceiver": {
            str(k): int(v) for k, v in per_tx.items()
        },
        "benign_train_per_class_transceiver": per_class_tx,
        "benign_held_out": {
            sp: {
                "n": v["n"],
                "per_class": {str(k): int(n) for k, n in v["per_class"].items()},
            }
            for sp, v in benign_held.items()
        },
        "split_role_counts": split_role_counts,
        "n_vessels_benign_train": int(benign_train["mmsi"].nunique()),
        "firewall": {
            "keep_roles": list(keep_roles),
            "train_split": train_split,
            "usv_in_benign_train": 0,
            "hostile_in_benign_train": int(
                (benign_train["role"] == "hostile").sum()
            ),
            "non_target_in_benign_train": int(
                (benign_train["role"] == "non_target").sum()
            ),
        },
    }
    return benign_train, summary


def feature_null_report(
    df: pd.DataFrame, feature_cols: list[str], course_cols: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(df)
    for col in feature_cols:
        if col not in df.columns:
            out[col] = {"present": False}
            continue
        n_null = int(df[col].isna().sum())
        out[col] = {
            "present": True,
            "n_null": n_null,
            "null_frac": round(n_null / n, 4) if n else None,
            "gated": col in course_cols,
        }
    if "heading_avail_frac" in df.columns:
        out["heading_avail_frac"] = {
            "present": True,
            "mean": float(df["heading_avail_frac"].mean()),
            "frac_below_0.25": float((df["heading_avail_frac"] < 0.25).mean()),
            "role": "quality_flag",
        }
    return out


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    nulls: dict[str, Any],
    spec: dict,
    feature_cols: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Benign training corpus — kinematic scorer")
    lines.append("")
    lines.append(f"Generated: {summary['timestamp']}")
    lines.append("")
    lines.append(
        "Training pool for the one-class benign-behavior model. "
        "AIS train split ∩ `role==benign` only. Hostile / non_target / `usv` "
        "are hard-excluded (firewall). Feature contract: "
        "`configs/defense/scorer_features.yaml`."
    )
    lines.append("")
    lines.append("## Retained counts")
    lines.append("")
    lines.append(
        f"- **Benign train tracks:** {summary['n_benign_train']:,} "
        f"(of {summary['n_train_total']:,} train / "
        f"{summary['n_tracks_total']:,} AIS total)"
    )
    lines.append(
        f"- **Vessels (MMSI):** {summary['n_vessels_benign_train']:,} "
        f"(vessel-disjoint from val/test by construction)"
    )
    lines.append(
        f"- **Excluded from train:** {summary['n_excluded_train']:,} "
        f"— {summary['excluded_train_by_role']}"
    )
    lines.append("")
    lines.append("### Per-class (benign train)")
    lines.append("")
    lines.append("| class | n | Class-A | Class-B |")
    lines.append("|---|---:|---:|---:|")
    ctx = summary.get("benign_train_per_class_transceiver") or {}
    for cls, n in summary["benign_train_per_class"].items():
        row = ctx.get(cls) or {}
        a = int(row.get("A", 0))
        b = int(row.get("B", 0))
        lines.append(f"| {cls} | {n:,} | {a:,} | {b:,} |")
    lines.append("")
    lines.append(
        f"**Transceiver mix:** {summary['benign_train_per_transceiver']} "
        "(Class-B = small-craft regime a hostile USV mimics)."
    )
    lines.append("")
    lines.append("### Held-out benign (not in training; FAR later)")
    lines.append("")
    for sp, block in (summary.get("benign_held_out") or {}).items():
        lines.append(f"- **{sp}:** {block['n']:,} tracks")
    lines.append("")
    lines.append("## Firewall attestation")
    lines.append("")
    fw = summary["firewall"]
    lines.append(
        f"- keep_roles = {fw['keep_roles']}; train_split = `{fw['train_split']}`"
    )
    lines.append(
        f"- usv / hostile / non_target in benign train: "
        f"{fw['usv_in_benign_train']} / {fw['hostile_in_benign_train']} / "
        f"{fw['non_target_in_benign_train']} (all must be 0)"
    )
    lines.append(
        f"- Excluded train by class: {summary['excluded_train_by_class']}"
    )
    lines.append("")
    lines.append("## Feature contract")
    lines.append("")
    lines.append(
        f"Frozen features ({len(feature_cols)}): "
        + ", ".join(f"`{c}`" for c in feature_cols)
    )
    lines.append("")
    lines.append(
        f"- Observation window: **{spec.get('observation_window_s')} s** "
        f"(sweep {spec.get('observation_window_sweep_s')}); "
        f"resample **{spec.get('resample_seconds')} s**."
    )
    lines.append(
        "- `range_to_shore` / `pos_speed_mean_kn` / heading-derived: excluded "
        "(see `scorer_features.yaml` `excluded`)."
    )
    lines.append(
        "- Course features (`turn_rate_*`, `cog_circ_std_deg`) gate on "
        "**non-null** (COG-derived; ~15–20% null when no course change). "
        "NaN = mask, never fill 0."
    )
    lines.append("")
    lines.append("### Null rates on benign train")
    lines.append("")
    lines.append("| feature | null % | gated |")
    lines.append("|---|---:|---|")
    for col, info in nulls.items():
        if not info.get("present"):
            continue
        if "null_frac" in info:
            pct = f"{100 * info['null_frac']:.1f}%"
            gated = "yes" if info.get("gated") else "no"
            lines.append(f"| `{col}` | {pct} | {gated} |")
        elif col == "heading_avail_frac":
            lines.append(
                f"| `{col}` (quality) | mean={info['mean']:.3f}; "
                f"<{0.25} = {100*info['frac_below_0.25']:.1f}% | flag |"
            )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `data/behavior/benign_train_manifest.parquet`")
    lines.append("- `data/behavior/benign_corpus_summary.json`")
    lines.append("- `configs/defense/scorer_features.yaml`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--out-data", type=Path, default=DEFAULT_OUT_DATA)
    ap.add_argument("--out-results", type=Path, default=DEFAULT_OUT_RESULTS)
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts only; do not write parquet/json/md")
    args = ap.parse_args()

    spec = _load_yaml(args.features)
    corpus = spec.get("corpus") or {}
    keep_roles = list(corpus.get("keep_roles") or ["benign"])
    train_split = str(corpus.get("train_split") or "train")
    core_cols, course_cols, feat_cols = feature_columns(spec)

    tracks_path = args.data_dir / "tracks" / "tracks_ais.parquet"
    splits_path = args.data_dir / "splits" / "ais_track_splits.csv"
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)
    if not splits_path.is_file():
        raise FileNotFoundError(splits_path)

    print(f"[corpus] loading {tracks_path.name} + {splits_path.name}")
    tracks = pd.read_parquet(tracks_path)
    splits = pd.read_csv(splits_path)

    # Confirm feature columns exist.
    missing_feats = [c for c in feat_cols if c not in tracks.columns]
    if missing_feats:
        raise KeyError(
            f"Feature columns missing from tracks/tracks_ais.parquet: {missing_feats}"
        )

    benign_train, summary = assemble(
        tracks, splits, keep_roles=keep_roles, train_split=train_split)

    # Sanity: firewall zeros.
    fw = summary["firewall"]
    if (fw["hostile_in_benign_train"] or fw["non_target_in_benign_train"]
            or fw["usv_in_benign_train"]):
        raise RuntimeError(f"Firewall attestation failed: {fw}")

    nulls = feature_null_report(benign_train, feat_cols, course_cols)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["feature_spec"] = str(args.features.relative_to(REPO_ROOT))
    summary["feature_columns"] = feat_cols
    summary["feature_nulls_benign_train"] = nulls
    summary["manifest"] = "data/behavior/benign_train_manifest.parquet"

    print(f"[corpus] benign train: {summary['n_benign_train']:,} tracks / "
          f"{summary['n_vessels_benign_train']:,} vessels")
    print(f"[corpus] excluded train: {summary['n_excluded_train']:,} "
          f"{summary['excluded_train_by_role']}")
    print(f"[corpus] per-class: {summary['benign_train_per_class']}")
    print(f"[corpus] transceiver: {summary['benign_train_per_transceiver']}")

    if args.dry_run:
        print("[corpus] dry-run — no files written")
        return 0

    # Manifest: identity + split provenance + frozen feature columns + quality.
    id_cols = [
        "trip_id", "mmsi", "canonical_class", "role", "transceiver_class",
        "source", "vessel_type_code", "n_points", "duration_s",
        "t_start", "t_end", "split", "region", "geo_cell", "start_day",
    ]
    quality_cols = ["heading_avail_frac"]
    keep = [c for c in id_cols + feat_cols + quality_cols if c in benign_train.columns]
    manifest = benign_train[keep].reset_index(drop=True)

    args.out_data.mkdir(parents=True, exist_ok=True)
    args.out_results.mkdir(parents=True, exist_ok=True)
    man_path = args.out_data / "benign_train_manifest.parquet"
    sum_path = args.out_data / "benign_corpus_summary.json"
    report_path = args.out_results / "corpus_report.md"

    manifest.to_parquet(man_path, index=False)
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        report_path, summary=summary, nulls=nulls, spec=spec,
        feature_cols=feat_cols,
    )
    print(f"[corpus] wrote {man_path} ({len(manifest):,} rows)")
    print(f"[corpus] wrote {sum_path}")
    print(f"[corpus] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
