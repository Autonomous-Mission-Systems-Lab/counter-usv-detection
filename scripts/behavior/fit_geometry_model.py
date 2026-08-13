#!/usr/bin/env python3
"""Fit kinematics_geometry arm envelopes (class-conditional + pooled ablation).

Joins kinematics window features with encounter-paired geometry rows at the
fit-population placements (berth_approach, anchorage). Leaves the frozen
kinematics_only arm under ``results/behavior_model/`` untouched.

Usage
-----
    python scripts/behavior/fit_geometry_model.py
    python scripts/behavior/fit_geometry_model.py --smoke
    python scripts/behavior/fit_geometry_model.py --envelope recreational
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.kinematics.behavior_model import (  # noqa: E402
    EnvelopeModel,
    MultiHorizonEnvelope,
    envelope_summary,
    fit_envelope,
    save_envelope,
    select_envelope_rows,
)

# Reuse FAR helpers from the kinematics fit script.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "behavior"))
from fit_behavior_model import multihorizon_test, score_split_far  # noqa: E402

DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "behavior_model_geometry.yaml"
DEFAULT_OUT = REPO_ROOT / "results" / "behavior_model_geometry"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


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


def load_joined_window(
    cfg: dict,
    window_s: int,
    *,
    require_complete: bool,
    placement_classes: list[str] | None = None,
) -> pd.DataFrame:
    """Join kinematics + geometry for one horizon.

    Default ``placement_classes=None`` uses the fit population
    (``fit_placement_classes`` / berth+anchorage). Pass an explicit list
    (or empty → all classes present) for placement-swept FAR validation.
    """
    kin_tmpl = cfg.get("kinematics_path_template") or (
        "data/behavior/features_window_{w}s.parquet"
    )
    geom_tmpl = cfg.get("geometry_path_template") or (
        "data/defense/features_geometry_window_{w}s.parquet"
    )
    kin_path = (REPO_ROOT / kin_tmpl.format(w=window_s)).resolve()
    geom_path = (REPO_ROOT / geom_tmpl.format(w=window_s)).resolve()
    if not kin_path.is_file():
        raise FileNotFoundError(kin_path)
    if not geom_path.is_file():
        raise FileNotFoundError(geom_path)

    kin = pd.read_parquet(kin_path)
    geom = pd.read_parquet(geom_path)

    if placement_classes is None:
        placement_classes = list(
            cfg.get("fit_placement_classes")
            or ["berth_approach", "anchorage"]
        )
    if placement_classes:
        geom = geom.loc[geom["placement_class"].isin(placement_classes)].copy()
    else:
        geom = geom.copy()
    if "geometry_usable" in geom.columns:
        geom = geom.loc[geom["geometry_usable"]].copy()

    # Vessel role: geometry table uses track_role; kinematics uses role.
    if "track_role" in geom.columns:
        geom = geom.loc[geom["track_role"] == "benign"].copy()
    if "role" in kin.columns:
        kin = kin.loc[kin["role"] == "benign"].copy()

    if require_complete:
        if "window_complete" in kin.columns:
            kin = kin.loc[kin["window_complete"]].copy()
        if "window_complete" in geom.columns:
            geom = geom.loc[geom["window_complete"]].copy()

    geom_meta = [
        "trip_id", "window_s", "asset_id", "port_region", "placement_class",
        "range_min_nm", "closing_rate_med_kn", "closing_rate_p90_kn",
        "bearing_rate_std_dps", "dcpa_nm", "tcpa_s", "closing_frac",
        "inbound_leg_persistence_s", "n_points_in_annulus", "cpa_range_nm",
        "geometry_usable",
    ]
    geom_meta = [c for c in geom_meta if c in geom.columns]
    # Prefer kinematics class / split / transceiver; keep geometry placement tags.
    merged = geom[geom_meta].merge(
        kin,
        on=["trip_id", "window_s"],
        how="inner",
        suffixes=("", "_kin"),
    )
    # Ensure split present.
    if "split" not in merged.columns or merged["split"].isna().all():
        splits = pd.read_csv(
            REPO_ROOT / (cfg.get("splits") or "data/splits/ais_track_splits.csv")
        )
        merged = merged.drop(columns=[c for c in merged.columns if c == "split"],
                             errors="ignore")
        merged = merged.merge(
            splits[["trip_id", "split"]], on="trip_id", how="left",
        )
    # Canonical class: prefer kinematics label.
    if "canonical_class_kin" in merged.columns:
        merged["canonical_class"] = merged["canonical_class_kin"].fillna(
            merged.get("canonical_class")
        )
    return merged


def write_report(path: Path, payload: dict) -> None:
    primary_w = payload["primary_window_s"]
    windows = payload["windows_s"]
    geom = payload.get("geometry_features") or []
    lines: list[str] = [
        "# Kinematics + geometry envelope fit",
        "",
        f"Generated: {payload['timestamp']}",
        "",
        "Extended arm (`kinematics_geometry`). Fit population = encounter pairs "
        "at **berth_approach** / **anchorage** only. Kinematics-only arm left "
        "frozen and untouched.",
        "",
        f"Horizons: **{windows} s** (primary **{primary_w} s**). "
        "120 s omitted — AIS cadence yields no usable inbound legs at that horizon.",
        "",
        f"Features core: `{payload['core_features']}`",
        f"Features course: `{payload['course_features']}`",
        f"Features geometry: `{geom}`",
        "",
        f"## Class-conditional GMM (primary horizon {primary_w} s, core+geometry)",
        "",
        "| envelope | n_train | n_val | k | val loglik | thr FAR@5% | test FAR@5% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def _row(env_name: str, block: dict) -> str:
        core = ((block.get("subspaces") or {}).get("core") or {}).get("gmm") or {}
        tf = ((block.get("test_far") or {}).get("gmm") or {})
        thr = (core.get("thresholds") or {}).get("far_0.05")
        ll, k = core.get("val_loglik"), core.get("n_components")
        if isinstance(ll, (int, float)) and isinstance(thr, (int, float)):
            far = tf.get("far")
            far_s = f"{100 * far:.1f}%" if isinstance(far, (int, float)) else "—"
            return (
                f"| `{env_name}` | {core.get('n_train', 0):,} | "
                f"{core.get('n_val', 0):,} | {k if k is not None else '—'} | "
                f"{ll:.3f} | {thr:.3f} | {far_s} |"
            )
        return f"| `{env_name}` | — | — | — | — | — | — |"

    for env_name, block in payload["envelopes"].items():
        if env_name.startswith("pooled"):
            continue
        lines.append(_row(env_name, block))

    lines += [
        "",
        "## Pooled-vs-class-conditional ablation",
        "",
        "Pooled envelope ignores asserted class (geometry as track anomaly). "
        "Class-conditional is the primary claim. Compare test FAR@5% and "
        "whether class conditioning absorbs routine fairway-like closures.",
        "",
    ]
    pooled = payload["envelopes"].get(payload.get("pooled_name") or "pooled_benign")
    if pooled:
        lines.append(_row(payload.get("pooled_name") or "pooled_benign", pooled))
        lines.append("")
        lines.append(
            f"Pooled test FAR@5%: "
            f"**{((pooled.get('test_far') or {}).get('gmm') or {}).get('far')}**"
        )
        lines.append(
            f"Pooled multi-horizon FAR@5%: "
            f"**{(pooled.get('multi_horizon_test') or {}).get('far')}**"
        )
    else:
        lines.append("_Pooled ablation not fitted._")

    lines += [
        "",
        "## Multi-horizon coverage + FAR (longest complete window, GMM)",
        "",
        "| envelope | test contacts | scored | coverage | FAR@5% |",
        "|---|---:|---:|---:|---:|",
    ]
    for env_name, block in payload["envelopes"].items():
        mh = block.get("multi_horizon_test") or {}
        cov = mh.get("coverage")
        far = mh.get("far")
        cov_s = f"{100 * cov:.1f}%" if isinstance(cov, (int, float)) else "—"
        far_s = f"{100 * far:.1f}%" if isinstance(far, (int, float)) else "—"
        lines.append(
            f"| `{env_name}` | {mh.get('n_universe', 0):,} | "
            f"{mh.get('n_scored', 0):,} | {cov_s} | {far_s} |"
        )

    cov_note = payload.get("coverage_note") or ""
    lines += [
        "",
        "## Coverage / n by class (train, primary horizon, fit-population)",
        "",
        "```",
        cov_note.rstrip() or "(see fit_summary.json → train_counts)",
        "```",
        "",
        "## Notes",
        "",
        "- Geometry features are appended to both `core` and `core_course` "
        "subspaces (extended vector).",
        "- Skipped envelopes had fewer than `min_train_rows` usable encounter "
        "rows at every horizon — reported, not imputed.",
        "- Bundles: `results/behavior_model_geometry/envelopes/<name>.joblib`.",
        "- Digested placements pin recorded in `FROZEN.json`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_freeze(
    out: Path,
    cfg_path: Path,
    payload: dict,
    envelope_names: list[str],
) -> Path:
    """Write a freeze manifest compatible with ConsistencyScorer.from_freeze."""
    env_dir = out / "envelopes"
    map_path = "configs/defense/class_envelope_map.yaml"
    feat_path = "configs/defense/scorer_features.yaml"
    eng_path = "configs/defense/engagement_geometry.yaml"
    cfg_rel = str(cfg_path.relative_to(REPO_ROOT))

    # YAML configs: path only (git is source of truth). Data pins keep digests.
    configs: dict[str, dict[str, Any]] = {
        "class_envelope_map": {"path": map_path},
        "scorer_features": {"path": feat_path},
        "behavior_model": {"path": cfg_rel},
        "engagement_geometry": {"path": eng_path},
    }
    placements = REPO_ROOT / "data" / "defense" / "placements.parquet"
    placements_digest = REPO_ROOT / "data" / "defense" / "placements_digest.json"
    if placements.is_file():
        configs["placements"] = {
            "path": "data/defense/placements.parquet",
            "sha256": _sha256_file(placements),
        }
    if placements_digest.is_file():
        configs["placements_digest"] = {
            "path": "data/defense/placements_digest.json",
            "sha256": _sha256_file(placements_digest),
        }

    envelopes: dict[str, dict[str, str]] = {}
    for name in envelope_names:
        rel = f"results/behavior_model_geometry/envelopes/{name}.joblib"
        p = REPO_ROOT / rel
        if p.is_file():
            envelopes[name] = {"path": rel, "sha256": _sha256_file(p)}

    freeze = {
        "version": datetime.now(timezone.utc).date().isoformat(),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_arm": "kinematics_geometry",
        "primary_model": "gmm",
        "primary_window_s": payload["primary_window_s"],
        "windows_s": payload["windows_s"],
        "configs": configs,
        "features": {
            "core": payload.get("core_features"),
            "course": payload.get("course_features"),
            "geometry": payload.get("geometry_features"),
        },
        "envelopes": envelopes,
        "far_floor": {"default_far": 0.05},
        "firewall": {
            "note": "Fitted on role==benign encounter pairs only; "
                    "hostile/synth never in training.",
        },
        "pooled_ablation": payload.get("pooled_name"),
    }
    path = out / "FROZEN.json"
    path.write_text(json.dumps(freeze, indent=2) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--envelope", type=str, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-pooled", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    emap = _load_yaml(
        REPO_ROOT / (cfg.get("envelope_map")
                     or "configs/defense/class_envelope_map.yaml")
    )
    envelopes = dict(emap.get("envelopes") or {})
    if args.envelope:
        if args.envelope not in envelopes:
            ap.error(f"unknown envelope {args.envelope!r}")
        envelopes = {args.envelope: envelopes[args.envelope]}

    pooled_cfg = cfg.get("pooled_ablation") or {}
    pooled_name = str(pooled_cfg.get("name") or "pooled_benign")
    do_pooled = (
        bool(pooled_cfg.get("enabled", True))
        and not args.skip_pooled
        and args.envelope is None
    )

    windows = [int(w) for w in (cfg.get("windows_s") or [600])]
    primary_w = int(cfg.get("window_s") or max(windows))
    if primary_w not in windows:
        windows = sorted(set(windows) | {primary_w})
    models_cfg = cfg.get("models") or {}
    primary = str(models_cfg.get("primary") or "gmm")
    baselines = [b for b in (models_cfg.get("baselines") or []) if b != primary]
    families = [primary] + baselines
    feat_cfg = cfg.get("features") or {}

    print(f"[fit-geometry] horizons: {windows} s (primary {primary_w} s)")
    print("[fit-geometry] joining kinematics × geometry …")
    complete_by_w = {
        w: load_joined_window(cfg, w, require_complete=True) for w in windows
    }
    testfull_by_w = {
        w: load_joined_window(cfg, w, require_complete=False) for w in windows
    }
    for w in windows:
        d = complete_by_w[w]
        print(
            f"  {w:>4} s complete: n={len(d):,} "
            f"train={int((d.split == 'train').sum()):,} "
            f"val={int((d.split == 'val').sum()):,} "
            f"test={int((d.split == 'test').sum()):,}"
        )

    if args.smoke:
        for w in windows:
            parts = []
            for s, n in (("train", 800), ("val", 200), ("test", 200)):
                sub = complete_by_w[w].loc[complete_by_w[w].split == s]
                if sub.empty:
                    continue
                parts.append(sub.sample(n=min(n, len(sub)), random_state=0))
            if parts:
                complete_by_w[w] = pd.concat(parts)
        print("[fit-geometry] smoke subsample applied")

    # Train counts note for report.
    primary_train = complete_by_w[primary_w].loc[
        complete_by_w[primary_w].split == "train"
    ]
    counts = (
        primary_train.groupby(["canonical_class", "placement_class"])
        .size()
        .unstack(fill_value=0)
        if len(primary_train) else pd.DataFrame()
    )
    coverage_note = counts.to_string() if len(counts) else "(empty)"

    out_env = args.out / "envelopes"
    out_env.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.relative_to(REPO_ROOT)),
        "feature_arm": "kinematics_geometry",
        "primary_window_s": primary_w,
        "windows_s": windows,
        "core_features": list(feat_cfg.get("core") or []),
        "course_features": list(feat_cfg.get("course") or []),
        "geometry_features": list(feat_cfg.get("geometry") or []),
        "pooled_name": pooled_name if do_pooled else None,
        "train_counts": counts.to_dict() if len(counts) else {},
        "coverage_note": coverage_note,
        "envelopes": {},
    }

    fit_targets: list[tuple[str, dict]] = [
        (name, dict(spec.get("members") or {}))
        for name, spec in envelopes.items()
    ]
    if do_pooled:
        fit_targets.append(
            (pooled_name, dict(pooled_cfg.get("members") or {}))
        )

    fitted_names: list[str] = []
    for name, members in fit_targets:
        print(f"\n[fit-geometry] envelope={name}")
        horizons: dict[int, EnvelopeModel] = {}
        horizon_meta: dict[str, Any] = {}
        for w in windows:
            d = complete_by_w[w]
            tr = d.loc[d.split == "train"]
            va = d.loc[d.split == "val"]
            model = fit_envelope(
                name, members, tr, va, cfg={**cfg, "window_s": w},
            )
            if model is None:
                continue
            horizons[w] = model
            gmm_core = ((model.subspaces.get("core") or {}).get("gmm"))
            horizon_meta[str(w)] = {
                "gmm_k": gmm_core.n_components if gmm_core else None,
                "n_train_core": gmm_core.n_train if gmm_core else 0,
            }

        if not horizons:
            print(f"  [skip] {name}: no horizon produced a fit")
            continue

        bundle = MultiHorizonEnvelope(
            name=name,
            members=members,
            primary_window_s=primary_w if primary_w in horizons else max(horizons),
            core_features=list(feat_cfg.get("core") or [])
            + list(feat_cfg.get("geometry") or []),
            course_features=list(feat_cfg.get("course") or []),
            horizons=horizons,
            primary=primary,  # type: ignore[arg-type]
        )
        save_envelope(bundle, out_env / f"{name}.joblib")
        fitted_names.append(name)

        # Primary-horizon summary + test FAR.
        prim = horizons.get(primary_w) or horizons[max(horizons)]
        block = envelope_summary(prim)
        block["horizons"] = horizon_meta
        te = complete_by_w[prim.window_s].loc[
            complete_by_w[prim.window_s].split == "test"
        ]
        te = select_envelope_rows(te, members)
        block["test_far"] = {
            fam: score_split_far(prim, te, model_name=fam)
            for fam in families
        }
        # Multi-horizon test uses incomplete-kept tables.
        block["multi_horizon_test"] = multihorizon_test(
            bundle, testfull_by_w, members, model_name=primary,
        )
        payload["envelopes"][name] = block
        print(
            f"  saved {name}.joblib  horizons={sorted(horizons)}  "
            f"test_FAR@5%={block['test_far'].get('gmm', {}).get('far')}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fit_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    write_report(args.out / "fit_report.md", payload)
    freeze_path = _write_freeze(args.out, args.config, payload, fitted_names)
    print(f"\n[fit-geometry] wrote {args.out / 'fit_report.md'}")
    print(f"[fit-geometry] wrote {freeze_path}")
    print(f"[fit-geometry] envelopes: {fitted_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
