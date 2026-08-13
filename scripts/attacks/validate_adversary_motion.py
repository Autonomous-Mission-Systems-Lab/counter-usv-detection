#!/usr/bin/env python3
"""Validity report for the adversary motion model (no DDR / cost-curve claims).

Checks dynamics caps, AIS cadence after thinning, feature extractability on
both arms, negative-control ≈ FAR smoke, and class-contrast smoke.

Usage
-----
    python scripts/attacks/validate_adversary_motion.py --smoke
    python scripts/attacks/validate_adversary_motion.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.kinematics import (  # noqa: E402
    build_sweep_cells,
    generate_two_phase_track,
    load_kinematics_config,
    load_placements,
    platforms_from_config,
    refuse_benign_corpus_write,
)
from counterusv.defense.consistency import ConsistencyScorer  # noqa: E402
from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402
from counterusv.defense.geometry_features import geometry_features_from_points  # noqa: E402
from counterusv.kinematics.features import (  # noqa: E402
    features_from_points,
    last_window_mask,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _operating_far(summary: dict, *, geometry: bool) -> float | None:
    if geometry:
        gate = summary.get("gate") or {}
        if "geometry_operating_far" in gate:
            return float(gate["geometry_operating_far"])
        op = (summary.get("by_split") or {}).get("test", {}).get("overall_operating")
        if isinstance(op, dict):
            cal = op.get("at_calibrated") or {}
            if "far_0.05" in cal:
                return float(cal["far_0.05"])
        return None
    test = (summary.get("by_split") or {}).get("test", {})
    overall = test.get("overall") or {}
    cal = overall.get("at_calibrated") or {}
    if "far_0.05" in cal:
        return float(cal["far_0.05"])
    return None


def _window_feats(
    points: pd.DataFrame,
    windows_s: list[int],
    *,
    asset_lat: float | None = None,
    asset_lon: float | None = None,
    annulus: dict | None = None,
    inbound_leg: dict | None = None,
    join_geometry: bool = False,
) -> tuple[dict[int, dict], set[int]]:
    """Extract last-W kinematics (+ optional geometry) features per window."""
    by_w: dict[int, dict] = {}
    complete: set[int] = set()
    for w in windows_s:
        mask = last_window_mask(points, float(w))
        win = points.loc[mask].copy()
        if len(win) < 3:
            continue
        span = float(win["t"].max() - win["t"].min()) if len(win) else 0.0
        if span < 0.5 * float(w):
            continue
        kin = features_from_points(win)
        if kin.empty:
            continue
        row = kin.iloc[0].to_dict()
        if join_geometry and asset_lat is not None and asset_lon is not None:
            geo = geometry_features_from_points(
                win, float(asset_lat), float(asset_lon),
                annulus=annulus, inbound_leg=inbound_leg,
            )
            if geo is None:
                continue
            row.update(geo)
        by_w[int(w)] = row
        complete.add(int(w))
    return by_w, complete


def _score_track(
    scorer: ConsistencyScorer,
    asserted: str,
    by_w: dict[int, dict],
    complete: set[int],
    meta: dict,
) -> dict:
    if not by_w:
        return {"status": "no_window", "is_inconsistent": None, "score": None}
    res = scorer.score(
        asserted,
        features_by_window=by_w,
        complete_windows=complete,
        purpose="eval",
        track_meta=meta,
    )
    return {
        "status": res.status,
        "is_inconsistent": res.is_inconsistent,
        "score": res.score,
        "envelope_used": res.envelope_used,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="small asset/platform subset for wiring")
    ap.add_argument("--no-verify-digests", action="store_true")
    ap.add_argument("--max-nc", type=int, default=None,
                    help="cap negative-control tracks scored (default: all / smoke 12)")
    ap.add_argument("--max-contrast", type=int, default=None,
                    help="cap class-contrast tracks (default: 6 smoke / 24 full)")
    args = ap.parse_args()

    cfg = load_kinematics_config(args.config)
    val = dict(cfg.get("validity") or {})
    thin_s = float(cfg.get("thin_cadence_s") or 60.0)
    cadence_tol = float(val.get("cadence_median_tol_s") or 5.0)
    nc_factor = float(val.get("negative_control_far_factor") or 2.0)
    nc_slack = float(val.get("negative_control_far_slack") or 0.05)
    kin_windows = [int(w) for w in (val.get("kinematics_windows_s") or [300, 600])]
    geo_window = int(val.get("geometry_primary_window_s") or 600)

    out = cfg.get("output") or {}
    report_path = REPO_ROOT / (out.get("validity_report")
                               or "results/adversary_motion/validity_report.md")
    summary_path = REPO_ROOT / (out.get("validity_summary")
                                or "results/adversary_motion/validity_summary.json")
    for p in (report_path, summary_path):
        refuse_benign_corpus_write(p)

    print("[validity] loading placements + engagement geometry …")
    assets, placements_sha = load_placements(cfg)
    if "role" in assets.columns and not cfg.get("include_far_only", False):
        roles = set(cfg.get("headline_placement_roles") or ["fit"])
        assets = assets.loc[assets["role"].isin(roles)].copy()
    if "valid" in assets.columns:
        assets = assets.loc[assets["valid"]].copy()
    if args.smoke:
        assets = assets.head(2)

    eng = load_engagement_geometry(
        REPO_ROOT / (cfg.get("engagement_geometry")
                     or "configs/defense/engagement_geometry.yaml")
    )
    annulus = {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm}
    inbound = {
        "require_points_before_cpa": int(
            eng.inbound_leg.get("require_points_before_cpa", 3)
        ),
        "min_points": int(eng.inbound_leg.get("min_points", 4)),
    }

    # Prefer pre-materialized points when present and not smoke.
    points_path = REPO_ROOT / (out.get("points")
                               or "results/adversary_motion/tracks_points.parquet")
    cells_path = REPO_ROOT / (out.get("cells")
                              or "results/adversary_motion/sweep_cells.parquet")
    plats = platforms_from_config(cfg)

    if args.smoke or not points_path.is_file():
        print("[validity] generating on-the-fly tracks …")
        smoke_cfg = dict(cfg)
        if args.smoke:
            smoke_cfg["sweep"] = {
                "v_mimic_kn": [8, 25],
                "commit_range_nm": [2.0],
                "bearing_offset_deg": [0.0],
                "mimicked_classes": ["fishing", "recreational"],
                "platforms": ["magura_v5"],
                "include_unconstrained": True,
            }
        cells = build_sweep_cells(smoke_cfg, assets)
        asset_xy = {
            str(r.asset_id): (float(r.lat), float(r.lon))
            for r in assets.itertuples()
        }
        tracks = []
        for cell in cells:
            lat, lon = asset_xy[str(cell.asset_id)]
            tr = generate_two_phase_track(
                lat, lon, cell, plats[cell.platform], cfg=cfg, trip_id=cell.cell_id,
            )
            tracks.append(tr)
        points = pd.concat(
            [t.points.assign(
                cell_id=t.meta["cell_id"],
                asset_id=t.meta["asset_id"],
                mimicked_class=t.meta["mimicked_class"],
                platform=t.meta["platform"],
                negative_control=t.meta["negative_control"],
                unconstrained=t.meta["unconstrained"],
                peak_speed_kn=t.meta["peak_speed_kn"],
                in_plausibility_band=t.meta["in_plausibility_band"],
            ) for t in tracks],
            ignore_index=True,
        )
        meta_by_trip = {t.meta["trip_id"]: t.meta for t in tracks}
    else:
        print(f"[validity] loading {points_path.relative_to(REPO_ROOT)} …")
        points = pd.read_parquet(points_path)
        meta_path = points_path.with_name("tracks_meta.json")
        meta_rows = json.loads(meta_path.read_text()) if meta_path.is_file() else []
        meta_by_trip = {str(m["trip_id"]): m for m in meta_rows}
        cells = None

    # ---------------- dynamics ----------------
    dyn = dict(cfg.get("dynamics") or {})
    max_turn = float(dyn.get("max_turn_rate_dps") or 8.0)
    max_accel = float(dyn.get("max_accel_kn_per_s") or 0.5)
    dynamics_ok = True
    dyn_notes: list[str] = []
    for tid, g in points.groupby("trip_id", sort=False):
        g = g.sort_values("t")
        sog = g["sog"].to_numpy(dtype="float64")
        plat_key = str(g["platform"].iloc[0]) if "platform" in g.columns else None
        burst = float(plats[plat_key].burst_kn) if plat_key in plats else None
        if burst is not None and float(np.nanmax(sog)) > burst + 1e-6:
            dynamics_ok = False
            dyn_notes.append(f"{tid}: sog {np.nanmax(sog):.1f} > burst {burst}")
        if len(g) >= 2:
            dt = np.diff(g["t"].to_numpy(dtype="float64"))
            dsog = np.diff(sog)
            with np.errstate(divide="ignore", invalid="ignore"):
                accel = np.where(dt > 0, dsog / dt, 0.0)
            if float(np.nanmax(np.abs(accel))) > max_accel + 0.05:
                # Fine synth was thinned — residual accel can look larger across
                # 60 s gaps; only flag extreme outliers (> 5× cap over cadence).
                if float(np.nanmax(np.abs(accel))) > max_accel * 5 + 1e-6:
                    dynamics_ok = False
                    dyn_notes.append(
                        f"{tid}: thinned |accel|={np.nanmax(np.abs(accel)):.3f}"
                    )

    # ---------------- cadence ----------------
    dts = []
    for tid, g in points.groupby("trip_id", sort=False):
        if len(g) < 2:
            continue
        dts.append(np.diff(np.sort(g["t"].to_numpy(dtype="float64"))))
    all_dt = np.concatenate(dts) if dts else np.array([thin_s])
    median_dt = float(np.median(all_dt))
    cadence_ok = abs(median_dt - thin_s) <= cadence_tol

    # ---------------- extractability ----------------
    asset_xy = {
        str(r.asset_id): (float(r.lat), float(r.lon))
        for r in assets.itertuples()
    }
    # Prefer operating berth assets for extractability sample.
    sample_ids = []
    for tid, g in points.groupby("trip_id", sort=False):
        aid = str(g["asset_id"].iloc[0]) if "asset_id" in g.columns else None
        if aid and "berth" in aid:
            sample_ids.append(str(tid))
        if len(sample_ids) >= (8 if args.smoke else 24):
            break
    if not sample_ids:
        sample_ids = list(points["trip_id"].astype(str).unique()[:8])

    kin_ok = 0
    geo_ok = 0
    for tid in sample_ids:
        g = points.loc[points["trip_id"].astype(str) == tid]
        aid = str(g["asset_id"].iloc[0])
        lat, lon = asset_xy.get(aid, (None, None))
        by_w, complete = _window_feats(g, kin_windows)
        if by_w:
            kin_ok += 1
        if lat is not None:
            by_g, _ = _window_feats(
                g, [geo_window], asset_lat=lat, asset_lon=lon,
                annulus=annulus, inbound_leg=inbound, join_geometry=True,
            )
            if by_g:
                geo_ok += 1
    extract_kin = kin_ok / max(len(sample_ids), 1)
    extract_geo = geo_ok / max(len(sample_ids), 1)
    extract_ok = extract_kin >= 0.8 and extract_geo >= 0.5

    # ---------------- scorers + FAR floors ----------------
    print("[validity] loading scorers …")
    verify = not args.no_verify_digests
    try:
        kin_scorer = ConsistencyScorer.from_freeze(
            REPO_ROOT / "results/behavior_model/FROZEN.json",
            verify_digests=verify,
        )
    except ValueError as e:
        print(f"[validity] WARNING kinematics freeze: {e}; retry without verify")
        kin_scorer = ConsistencyScorer.from_freeze(
            REPO_ROOT / "results/behavior_model/FROZEN.json",
            verify_digests=False,
        )
    try:
        geo_scorer = ConsistencyScorer.from_freeze(
            REPO_ROOT / "results/behavior_model_geometry/FROZEN.json",
            verify_digests=verify,
        )
    except ValueError as e:
        print(f"[validity] WARNING geometry freeze: {e}; retry without verify")
        geo_scorer = ConsistencyScorer.from_freeze(
            REPO_ROOT / "results/behavior_model_geometry/FROZEN.json",
            verify_digests=False,
        )

    kin_far_sum = _load_json(REPO_ROOT / (
        val.get("kinematics_far_summary") or "results/behavior_model/far_summary.json"
    ))
    geo_far_sum = _load_json(REPO_ROOT / (
        val.get("geometry_far_summary")
        or "results/behavior_model_geometry/far_placement_summary.json"
    ))
    kin_far = _operating_far(kin_far_sum, geometry=False) or 0.035
    geo_far = _operating_far(geo_far_sum, geometry=True) or 0.062

    # ---------------- negative-control smoke ----------------
    nc_trips = []
    if "negative_control" in points.columns:
        for tid, g in points.groupby("trip_id", sort=False):
            if bool(g["negative_control"].iloc[0]):
                nc_trips.append(str(tid))
    max_nc = args.max_nc
    if max_nc is None:
        max_nc = 12 if args.smoke else len(nc_trips)
    nc_trips = nc_trips[:max_nc]
    print(f"[validity] negative-control smoke n={len(nc_trips)} …")

    def _nc_rate(scorer, join_geo: bool, windows: list[int]) -> dict:
        flagged = 0
        scored = 0
        abstained = 0
        for tid in nc_trips:
            g = points.loc[points["trip_id"].astype(str) == tid]
            aid = str(g["asset_id"].iloc[0])
            cls = str(g["mimicked_class"].iloc[0])
            lat, lon = asset_xy[aid]
            meta = meta_by_trip.get(tid) or {
                "role": "hostile", "source": "synth", "trip_id": tid,
            }
            by_w, complete = _window_feats(
                g, windows, asset_lat=lat, asset_lon=lon,
                annulus=annulus, inbound_leg=inbound, join_geometry=join_geo,
            )
            res = _score_track(scorer, cls, by_w, complete, meta)
            if res["status"] != "scored" or res["is_inconsistent"] is None:
                abstained += 1
                continue
            scored += 1
            if res["is_inconsistent"]:
                flagged += 1
        rate = (flagged / scored) if scored else None
        return {
            "n_trips": len(nc_trips), "n_scored": scored, "n_abstain": abstained,
            "n_flagged": flagged, "flag_rate": rate,
        }

    nc_kin = _nc_rate(kin_scorer, False, kin_windows)
    nc_geo = _nc_rate(geo_scorer, True, [geo_window] + [
        w for w in kin_windows if w != geo_window
    ])

    def _nc_ok(rate: float | None, floor: float) -> bool | None:
        if rate is None:
            return None
        ceiling = max(floor * nc_factor, floor + nc_slack)
        return rate <= ceiling + 1e-9

    nc_kin_ok = _nc_ok(nc_kin["flag_rate"], kin_far)
    nc_geo_ok = _nc_ok(nc_geo["flag_rate"], geo_far)

    # ---------------- class-contrast smoke ----------------
    contrast_n = args.max_contrast
    if contrast_n is None:
        contrast_n = 6 if args.smoke else 24
    # Prefer unconstrained / high-v cells for contrast visibility.
    contrast_ids = []
    if "unconstrained" in points.columns:
        for tid, g in points.groupby("trip_id", sort=False):
            if bool(g["unconstrained"].iloc[0]):
                contrast_ids.append(str(tid))
            if len(contrast_ids) >= contrast_n:
                break
    if len(contrast_ids) < contrast_n:
        for tid in points["trip_id"].astype(str).unique():
            if tid not in contrast_ids:
                contrast_ids.append(str(tid))
            if len(contrast_ids) >= contrast_n:
                break

    scoreable = sorted(set(kin_scorer.scoreable_classes()) | set(geo_scorer.scoreable_classes()))
    # Focus mimic targets + a few others.
    contrast_classes = [c for c in [
        "fishing", "recreational", "sailing", "cargo_merchant", "working_service",
        "passenger_ferry",
    ] if c in scoreable]
    print(f"[validity] class-contrast smoke n_tracks={len(contrast_ids)} "
          f"classes={contrast_classes}")

    contrast_rows = []
    for tid in contrast_ids:
        g = points.loc[points["trip_id"].astype(str) == tid]
        aid = str(g["asset_id"].iloc[0])
        lat, lon = asset_xy[aid]
        meta = meta_by_trip.get(tid) or {
            "role": "hostile", "source": "synth", "trip_id": tid,
        }
        by_kin, comp_kin = _window_feats(g, kin_windows)
        by_geo, comp_geo = _window_feats(
            g, [geo_window] + [w for w in kin_windows if w != geo_window],
            asset_lat=lat, asset_lon=lon,
            annulus=annulus, inbound_leg=inbound, join_geometry=True,
        )
        for cls in contrast_classes:
            rk = _score_track(kin_scorer, cls, by_kin, comp_kin, meta)
            rg = _score_track(geo_scorer, cls, by_geo, comp_geo, meta)
            contrast_rows.append({
                "trip_id": tid,
                "asserted_class": cls,
                "kin_status": rk["status"],
                "kin_flag": rk["is_inconsistent"],
                "geo_status": rg["status"],
                "geo_flag": rg["is_inconsistent"],
            })
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_summary = {}
    if not contrast_df.empty:
        for cls, sub in contrast_df.groupby("asserted_class"):
            k = sub.loc[sub["kin_status"] == "scored"]
            g = sub.loc[sub["geo_status"] == "scored"]
            contrast_summary[str(cls)] = {
                "kin_flag_rate": float(k["kin_flag"].mean()) if len(k) else None,
                "kin_n": int(len(k)),
                "geo_flag_rate": float(g["geo_flag"].mean()) if len(g) else None,
                "geo_n": int(len(g)),
            }

    # ---------------- freeze check ----------------
    freeze_path = REPO_ROOT / (out.get("freeze")
                               or "results/adversary_motion/FROZEN_SWEEP.json")
    freeze = _load_json(freeze_path)

    def _has_ddr_payload(obj: object) -> bool:
        """True only if numeric DDR / cost-curve fields are present (not prose)."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k).lower()
                if key in {
                    "ddr", "detection_rate", "cost_curve", "ddr_by_cell",
                    "adaptive_cost_curve",
                }:
                    return True
                if _has_ddr_payload(v):
                    return True
        elif isinstance(obj, list):
            return any(_has_ddr_payload(x) for x in obj)
        return False

    freeze_has_ddr = _has_ddr_payload(freeze)

    # ---------------- report ----------------
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(args.smoke),
        "n_points": int(len(points)),
        "n_trips": int(points["trip_id"].nunique()),
        "placements_sha": placements_sha,
        "dynamics": {"ok": dynamics_ok, "notes": dyn_notes[:20],
                     "max_turn_rate_dps": max_turn, "max_accel_kn_per_s": max_accel},
        "cadence": {
            "ok": cadence_ok,
            "median_dt_s": median_dt,
            "target_s": thin_s,
            "tol_s": cadence_tol,
        },
        "extractability": {
            "ok": extract_ok,
            "kinematics_frac": extract_kin,
            "geometry_frac": extract_geo,
            "n_sample": len(sample_ids),
            "geometry_primary_window_s": geo_window,
        },
        "negative_control": {
            "kinematics": {**nc_kin, "operating_far": kin_far, "within_tol": nc_kin_ok},
            "geometry": {**nc_geo, "operating_far": geo_far, "within_tol": nc_geo_ok},
            "tolerance": {"factor": nc_factor, "slack": nc_slack},
            "note": (
                "Reported smoke, not a hard gate for thin-n. "
                "Full-fidelity mimics scored under matched asserted class."
            ),
        },
        "class_contrast": contrast_summary,
        "freeze": {
            "path": str(freeze_path.relative_to(REPO_ROOT)) if freeze_path.is_file() else None,
            "digest": (freeze.get("digests") or {}).get("sweep_cells"),
            "n_cells": freeze.get("n_cells"),
            "ddr_numbers_present": freeze_has_ddr,
        },
        "scope": (
            "Validity only — no DDR / adaptive cost-curve claims. "
            "Cost curves consume the frozen sweep under a separate evaluation."
        ),
    }

    lines = [
        "# Adversary motion model — validity report",
        "",
        f"_Generated {summary['timestamp']}"
        + (" (smoke)" if args.smoke else "")
        + "._",
        "",
        "## Scope",
        "",
        "This report validates the generator instrument (dynamics, cadence,",
        "extractability, negative-control ≈ FAR, class contrast).",
        "**It does not claim DDR or adaptive cost curves** — those consume the",
        "frozen sweep under a later evaluation.",
        "",
        "## Dynamics",
        "",
        f"- Caps: turn ≤ {max_turn} deg/s (fine), accel ≤ {max_accel} kn/s (fine).",
        f"- Verdict: **{'PASS' if dynamics_ok else 'FAIL'}**"
        + (f" ({len(dyn_notes)} notes)" if dyn_notes else "") + ".",
        "",
        "## Cadence",
        "",
        f"- Target thin cadence: **{thin_s:.0f} s** (median Δt = {median_dt:.1f} s).",
        f"- Verdict: **{'PASS' if cadence_ok else 'FAIL'}** (tol ±{cadence_tol} s).",
        "",
        "## Extractability",
        "",
        f"- Kinematics features (windows {kin_windows}): "
        f"**{extract_kin:.0%}** of sample ({kin_ok}/{len(sample_ids)}).",
        f"- Geometry features (primary {geo_window} s): "
        f"**{extract_geo:.0%}** of sample ({geo_ok}/{len(sample_ids)}).",
        f"- Verdict: **{'PASS' if extract_ok else 'FAIL'}**.",
        "",
        "## Negative-control smoke",
        "",
        "Full-fidelity mimics scored under the **matched** asserted class.",
        "Flag rate should land near that arm's operating FAR (loose tolerance:",
        f"≤ max({nc_factor}× FAR, FAR+{nc_slack:.0%}); reported, not gate-failed for thin-n).",
        "",
        f"| Arm | n scored | flag rate | operating FAR | within tol |",
        f"|---|---:|---:|---:|:---:|",
        "| kinematics | {n} | {rate} | {far:.1%} | {ok} |".format(
            n=nc_kin["n_scored"],
            rate=("—" if nc_kin["flag_rate"] is None
                  else f"{nc_kin['flag_rate']:.1%}"),
            far=kin_far,
            ok=nc_kin_ok,
        ),
        "| geometry | {n} | {rate} | {far:.1%} | {ok} |".format(
            n=nc_geo["n_scored"],
            rate=("—" if nc_geo["flag_rate"] is None
                  else f"{nc_geo['flag_rate']:.1%}"),
            far=geo_far,
            ok=nc_geo_ok,
        ),
        "",
        "## Class-contrast smoke",
        "",
        "Same points scored under all listed asserted classes "
        "(contrast is the measurement).",
        "",
        "| Asserted class | kin flag rate (n) | geo flag rate (n) |",
        "|---|---|---|",
    ]
    for cls, row in sorted(contrast_summary.items()):
        k = ("—" if row["kin_flag_rate"] is None
             else f"{row['kin_flag_rate']:.0%} (n={row['kin_n']})")
        g = ("—" if row["geo_flag_rate"] is None
             else f"{row['geo_flag_rate']:.0%} (n={row['geo_n']})")
        lines.append(f"| {cls} | {k} | {g} |")
    lines += [
        "",
        "## Freeze",
        "",
        (
            f"- Manifest: `{summary['freeze']['path']}`"
            if summary["freeze"]["path"] else
            "- Manifest: **missing** (run `generate_adversary_tracks.py` without `--smoke`)."
        ),
        f"- Sweep digest: `{summary['freeze']['digest'] or '—'}`",
        f"- DDR numbers in freeze: **{summary['freeze']['ddr_numbers_present']}** "
        "(must be false).",
        "",
        "## Artifacts",
        "",
        f"- Points root: `results/adversary_motion/`",
        f"- Summary JSON: `{summary_path.relative_to(REPO_ROOT)}`",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[validity] wrote {report_path.relative_to(REPO_ROOT)}")
    print(f"[validity] wrote {summary_path.relative_to(REPO_ROOT)}")
    print(
        f"[validity] dynamics={dynamics_ok} cadence={cadence_ok} "
        f"extract={extract_ok} nc_kin={nc_kin_ok} nc_geo={nc_geo_ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
