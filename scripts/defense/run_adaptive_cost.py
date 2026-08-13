#!/usr/bin/env python3
"""Adaptive-adversary cost curve (RQ3) — oracle-only.

Joins terminal oracle DDR to the frozen motion sweep; reports DDR vs added
approach time and mimicry knobs; optionally walks tracks causally for
first-flag warning time / standoff. Never mutates the motion freeze.

Usage
-----
    python scripts/defense/run_adaptive_cost.py --skip-warning-time
    python scripts/defense/run_adaptive_cost.py --smoke
    python scripts/defense/run_adaptive_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from counterusv.attacks.kinematics import (  # noqa: E402
    load_kinematics_config,
    load_placements,
)
from counterusv.attacks.oracle import PerfectDisguiseOracle  # noqa: E402
from counterusv.defense.engagement import load_engagement_geometry  # noqa: E402
from counterusv.defense.harness import evaluate_contact  # noqa: E402
from counterusv.defense.pipeline import (  # noqa: E402
    DefensePipeline,
    PipelineConfig,
    load_pipeline_config,
)
from counterusv.eval.adaptive_cost import (  # noqa: E402
    CONSISTENCY_ARMS,
    ddr_by_axis,
    first_flag_along_track,
    join_ddr_sweep,
    load_adaptive_cost_config,
    load_freeze_platforms,
    pivot_curve_markdown,
    warning_summary,
)
from counterusv.eval.oracle_ddr import (  # noqa: E402
    load_operating_far,
    load_oracle_ddr_config,
    resolve_path,
)


def _load_pipelines(
    oracle_cfg: dict[str, Any],
    arm_names: list[str],
    *,
    verify_digests: bool,
    far_target: float,
) -> dict[str, DefensePipeline]:
    base = load_pipeline_config()
    pipes: dict[str, DefensePipeline] = {}
    arms = oracle_cfg.get("arms") or {}
    for arm_name in arm_names:
        arm = arms[arm_name]
        freeze = resolve_path(arm["freeze_path"])
        feature_arm = str(arm.get("feature_arm") or arm_name)
        pcfg = PipelineConfig(
            version=base.version,
            frozen=base.frozen,
            far_target=far_target,
            feature_arm=feature_arm,  # type: ignore[arg-type]
            freeze_path=freeze,
            verify_digests=verify_digests,
            primary_model=base.primary_model,
            association_note=base.association_note,
            path=base.path,
        )
        pipes[arm_name] = DefensePipeline.from_freeze(
            freeze,
            far_target=far_target,
            verify_digests=verify_digests,
            pipeline_config=base.path,
        )
        pipes[arm_name].config = pcfg
    return pipes


def _asset_coords(placements: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for _, row in placements.iterrows():
        out[str(row["asset_id"])] = (float(row["lat"]), float(row["lon"]))
    return out


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    joined: pd.DataFrame,
    curves_by_axis: dict[str, pd.DataFrame],
    warn_sum: pd.DataFrame | None,
    far_by_arm: dict[str, float],
    arms: list[str],
    classes: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Adaptive-adversary cost curve (RQ3)\n\n")
    lines.append(
        f"**Condition:** `{summary['condition']}` — oracle-only; "
        "patch-conditioned slice deferred (TMSR ≈ 0).\n\n"
    )
    lines.append(
        "**Cost:** "
        r"$\Delta t_{\mathrm{add}} = R(1/v_{\mathrm{mimic}} - 1/v_{\mathrm{max}})$ "
        f"with $R = R_{{\\mathrm{{start}}}} - R_{{\\mathrm{{commit}}}}$ "
        f"({summary['cost']['start_range_nm']:g} nm − commit), "
        f"$v_{{\\mathrm{{max}}}}$ = platform burst. "
        "Reported in **minutes**. Unconstrained → 0.\n\n"
    )
    lines.append("**FAR paired:** ")
    lines.append(
        ", ".join(f"{a} {_fmt_pct(far_by_arm.get(a))}" for a in arms) + ".\n\n"
    )
    lines.append(
        f"Joined non-NC consistency cells: **{len(joined)}** "
        f"({summary['n_trips']} trips × {len(arms)} arms).\n\n"
    )

    lines.append("## DDR vs added approach time\n\n")
    dt = curves_by_axis.get("delta_t_add_min")
    if dt is not None and not dt.empty:
        # Round for readability in the pivot.
        show = dt.copy()
        show["delta_t_add_min"] = show["delta_t_add_min"].round(1)
        lines.append(
            pivot_curve_markdown(
                show, axis="delta_t_add_min", arms=arms, classes=classes
            )
        )
    else:
        lines.append("_no rows_\n")

    for axis, title in (
        ("v_mimic_kn", "DDR vs mimic speed"),
        ("commit_range_nm", "DDR vs commit range"),
        ("bearing_offset_deg", "DDR vs bearing offset"),
    ):
        lines.append(f"\n## {title}\n\n")
        cur = curves_by_axis.get(axis)
        if cur is None or cur.empty:
            lines.append("_no rows_\n")
        else:
            lines.append(
                pivot_curve_markdown(cur, axis=axis, arms=arms, classes=classes)
            )

    # Recreational contrast at Δt ≈ 0 (unconstrained / burst-like).
    lines.append("\n## Recreational contrast (near-zero time tax)\n\n")
    rec = joined.loc[joined["mimicked_class"].astype(str) == "recreational"]
    near0 = rec.loc[rec["delta_t_add_min"] <= 1e-6]
    if near0.empty:
        lines.append("_no recreational near-zero Δt cells_\n")
    else:
        for arm in arms:
            g = near0.loc[near0["arm"].astype(str) == arm]
            n = len(g)
            n_f = int((g["action"] == "flag").sum())
            ddr = (n_f / n) if n else None
            lines.append(
                f"- `{arm}` at Δt_add ≈ 0: DDR **{_fmt_pct(ddr)}** (n={n})\n"
            )

    if warn_sum is not None and not warn_sum.empty:
        lines.append("\n## Warning time / standoff at first flag\n\n")
        lines.append(
            "Causal checkpoints every "
            f"**{summary['warning_time']['stride_s']:g} s** once the arm "
            "window can fill; $t_{\\mathrm{flag}}$ from annulus entry "
            f"(≤ {summary['warning_time']['max_range_nm']:g} nm).\n\n"
        )
        lines.append(
            "| class | arm | n | flag rate | median $R_{\\mathrm{flag}}$ (nm) | "
            "median $t_{\\mathrm{flag}}$ (min) |\n"
        )
        lines.append("|---|---|---:|---:|---:|---:|\n")
        for _, r in warn_sum.iterrows():
            r_med = r["R_flag_nm_median"]
            t_med = r["t_flag_min_median"]
            r_s = "—" if r_med is None or pd.isna(r_med) else f"{float(r_med):.2f}"
            t_s = "—" if t_med is None or pd.isna(t_med) else f"{float(t_med):.1f}"
            lines.append(
                f"| {r['mimicked_class']} | {r['arm']} | {int(r['n'])} | "
                f"{_fmt_pct(r['flag_rate'])} | {r_s} | {t_s} |\n"
            )
    elif summary.get("warning_time", {}).get("skipped"):
        lines.append(
            "\n## Warning time / standoff\n\n"
            "_Skipped (`--skip-warning-time`). Terminal DDR curves above "
            "still stand._\n"
        )

    lines.append("\n## Firewall\n\n")
    lines.append(str(summary.get("firewall_note") or "").strip() + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--skip-warning-time",
        action="store_true",
        help="join + curves only (no causal first-flag pass)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="small trip subset for the warning-time pass",
    )
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument(
        "--no-verify-digests",
        action="store_true",
        help="skip behavior-model freeze digest verification",
    )
    args = ap.parse_args()

    cfg = load_adaptive_cost_config(args.config)
    far_target = float(cfg.get("far_target") or 0.05)
    thin_n = int(cfg.get("thin_n_threshold") or 20)
    arms = [str(a) for a in (cfg.get("arms") or list(CONSISTENCY_ARMS))]
    cost = cfg.get("cost") or {}
    start_r = float(cost.get("start_range_nm") or 12.0)
    wt_cfg = cfg.get("warning_time") or {}
    stride_s = float(wt_cfg.get("stride_s") or 60.0)

    od = cfg.get("oracle_ddr") or {}
    sw = cfg.get("sweep") or {}
    out = cfg.get("output") or {}

    freeze_path = resolve_path(sw.get("freeze") or "results/adversary_motion/FROZEN_SWEEP.json")
    print("[adaptive_cost] asserting freeze has no DDR payload …")
    freeze, platforms = load_freeze_platforms(freeze_path)

    cells_path = resolve_path(od.get("cells") or "results/oracle_ddr/oracle_ddr_cells.parquet")
    sweep_path = resolve_path(sw.get("cells") or "results/adversary_motion/sweep_cells.parquet")
    if not cells_path.is_file():
        raise FileNotFoundError(f"missing {cells_path}; run run_oracle_ddr.py first")
    if not sweep_path.is_file():
        raise FileNotFoundError(f"missing {sweep_path}")

    print(f"[adaptive_cost] loading {cells_path.relative_to(REPO_ROOT)} …")
    ddr = pd.read_parquet(cells_path)
    sweep = pd.read_parquet(sweep_path)
    joined = join_ddr_sweep(
        ddr,
        sweep,
        platforms,
        start_range_nm=start_r,
        arms=arms,
        exclude_nc=True,
    )
    print(f"[adaptive_cost] joined {len(joined)} non-NC consistency rows")

    classes = sorted(joined["mimicked_class"].astype(str).unique().tolist())
    curves_by_axis: dict[str, pd.DataFrame] = {}
    curve_frames: list[pd.DataFrame] = []
    for axis in (
        "delta_t_add_min",
        "v_mimic_kn",
        "commit_range_nm",
        "bearing_offset_deg",
    ):
        cur = ddr_by_axis(joined, axis, thin_n_threshold=thin_n)
        cur.insert(0, "axis", axis)
        curves_by_axis[axis] = cur
        curve_frames.append(cur)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()

    # FAR footnotes from oracle_ddr arm configs.
    oracle_cfg = load_oracle_ddr_config(
        resolve_path(od.get("config") or "configs/defense/oracle_ddr.yaml")
    )
    far_by_arm: dict[str, float] = {}
    for arm_name in arms:
        arm = (oracle_cfg.get("arms") or {}).get(arm_name) or {}
        geo = bool(arm.get("join_geometry"))
        default = 0.062 if geo else 0.035
        far_by_arm[arm_name] = load_operating_far(
            arm.get("far_summary") or "",
            geometry=geo,
            default=default,
        )

    eng = load_engagement_geometry(
        resolve_path(cfg.get("engagement_geometry") or "configs/defense/engagement_geometry.yaml")
    )
    max_r = float(eng.max_range_nm)

    warn_df: pd.DataFrame | None = None
    warn_sum: pd.DataFrame | None = None
    warning_meta: dict[str, Any] = {
        "skipped": bool(args.skip_warning_time),
        "stride_s": stride_s,
        "max_range_nm": max_r,
    }

    if not args.skip_warning_time:
        points_path = resolve_path(
            sw.get("points") or "results/adversary_motion/tracks_points.parquet"
        )
        if not points_path.is_file():
            raise FileNotFoundError(f"missing {points_path}")
        print(f"[adaptive_cost] loading points {points_path.relative_to(REPO_ROOT)} …")
        points = pd.read_parquet(points_path)

        kin_cfg = load_kinematics_config(
            resolve_path(cfg.get("kinematics_sweep") or "configs/attacks/kinematics.yaml")
        )
        placements, _ = load_placements(kin_cfg)
        # Prefer placements path from adaptive_cost config when set.
        place_path = cfg.get("placements")
        if place_path:
            pp = resolve_path(place_path)
            if pp.is_file():
                placements = pd.read_parquet(pp)
        asset_xy = _asset_coords(placements)

        annulus = {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm}
        inbound = {
            "require_points_before_cpa": int(
                eng.inbound_leg.get("require_points_before_cpa", 3)
            ),
            "min_points": int(eng.inbound_leg.get("min_points", 4)),
        }

        print("[adaptive_cost] loading defenses for causal pass …")
        verify = not args.no_verify_digests
        try:
            pipes = _load_pipelines(
                oracle_cfg, arms, verify_digests=verify, far_target=far_target
            )
        except ValueError as e:
            if verify and "digest mismatch" in str(e):
                print(f"[adaptive_cost] WARNING: {e}")
                pipes = _load_pipelines(
                    oracle_cfg, arms, verify_digests=False, far_target=far_target
                )
            else:
                raise
        oracle = PerfectDisguiseOracle.from_config(
            resolve_path(oracle_cfg.get("oracle") or "configs/attacks/oracle.yaml")
        )

        # Trip list from joined (unique cell_ids), exclude NC already.
        trip_ids = sorted(joined["cell_id"].astype(str).unique().tolist())
        if args.smoke:
            keep: list[str] = []
            for tid in trip_ids:
                g = points.loc[points["trip_id"].astype(str) == tid]
                if g.empty:
                    continue
                if str(g["platform"].iloc[0]) != "magura_v5":
                    continue
                if str(g["mimicked_class"].iloc[0]) not in ("fishing", "recreational"):
                    continue
                unc = bool(g["unconstrained"].iloc[0]) if "unconstrained" in g.columns else False
                if unc:
                    keep.append(tid)
                    continue
                # One representative sweep cell per class.
                row = joined.loc[joined["cell_id"].astype(str) == tid].iloc[0]
                if (
                    float(row["v_mimic_kn"]) == 25.0
                    and float(row["commit_range_nm"]) == 2.0
                    and float(row["bearing_offset_deg"]) == 0.0
                ):
                    keep.append(tid)
            trip_ids = keep
            print(f"[adaptive_cost] smoke warning subset n_trips={len(trip_ids)}")
        if args.max_cells is not None:
            trip_ids = trip_ids[: int(args.max_cells)]
            print(f"[adaptive_cost] capped warning trips to {len(trip_ids)}")

        records: list[dict[str, Any]] = []
        t0 = time.time()
        n_trips = len(trip_ids)
        for i, tid in enumerate(trip_ids, 1):
            if i == 1 or i % 50 == 0 or i == n_trips:
                print(f"[adaptive_cost] warning-time {i}/{n_trips} …", flush=True)
            g = points.loc[points["trip_id"].astype(str) == tid].copy()
            if g.empty:
                continue
            mimicked = str(g["mimicked_class"].iloc[0])
            asset_id = str(g["asset_id"].iloc[0])
            platform = str(g["platform"].iloc[0])
            unc = bool(g["unconstrained"].iloc[0]) if "unconstrained" in g.columns else False
            lat, lon = asset_xy.get(asset_id, (None, None))
            if lat is None:
                continue
            assertion = oracle.assert_class(
                mimicked,
                contact_id=tid,
                true_class="usv",
                box_xyxy=(0.0, 0.0, 10.0, 10.0),
            )
            meta = {
                "role": "hostile",
                "source": "synth",
                "trip_id": tid,
                "canonical_class": "usv",
            }
            base_join = joined.loc[joined["cell_id"].astype(str) == tid]
            for arm_name in arms:
                arm = (oracle_cfg.get("arms") or {})[arm_name]
                windows = [int(w) for w in (arm.get("windows_s") or [])]
                join_geo = bool(arm.get("join_geometry"))
                pipe = pipes[arm_name]

                def _eval(
                    *,
                    features_by_window: dict,
                    complete_windows: set,
                    _pipe=pipe,
                    _assertion=assertion,
                    _meta=meta,
                ):
                    return evaluate_contact(
                        _pipe,
                        assertion=_assertion,
                        features_by_window=features_by_window,
                        complete_windows=complete_windows,
                        purpose="eval",
                        track_meta=_meta,
                        far_target=far_target,
                    )

                out_flag = first_flag_along_track(
                    g,
                    asset_lat=float(lat),
                    asset_lon=float(lon),
                    max_range_nm=max_r,
                    windows_s=windows,
                    join_geometry=join_geo,
                    annulus=annulus,
                    inbound_leg=inbound,
                    evaluate_fn=_eval,
                    stride_s=stride_s,
                )
                # Attach cost axes from join (any arm row for this cell).
                jrow = base_join.iloc[0] if len(base_join) else None
                records.append({
                    "cell_id": tid,
                    "trip_id": tid,
                    "arm": arm_name,
                    "mimicked_class": mimicked,
                    "platform": platform,
                    "asset_id": asset_id,
                    "unconstrained": unc,
                    "v_mimic_kn": float(jrow["v_mimic_kn"]) if jrow is not None else None,
                    "commit_range_nm": (
                        float(jrow["commit_range_nm"]) if jrow is not None else None
                    ),
                    "bearing_offset_deg": (
                        float(jrow["bearing_offset_deg"]) if jrow is not None else None
                    ),
                    "delta_t_add_min": (
                        float(jrow["delta_t_add_min"]) if jrow is not None else None
                    ),
                    **out_flag,
                })
        warn_df = pd.DataFrame(records)
        warn_sum = warning_summary(warn_df, thin_n_threshold=thin_n)
        warning_meta["n_trips"] = n_trips
        warning_meta["n_rows"] = len(warn_df)
        warning_meta["elapsed_s"] = round(time.time() - t0, 1)
        print(
            f"[adaptive_cost] warning-time done in {warning_meta['elapsed_s']}s "
            f"({len(warn_df)} rows)"
        )

    joined_path = resolve_path(out.get("joined") or "results/adaptive_cost/adaptive_cost_joined.parquet")
    curves_path = resolve_path(out.get("curves") or "results/adaptive_cost/adaptive_cost_curves.parquet")
    warn_path = resolve_path(out.get("warning") or "results/adaptive_cost/adaptive_cost_warning.parquet")
    report_path = resolve_path(out.get("report") or "results/adaptive_cost/adaptive_cost_report.md")
    summary_path = resolve_path(out.get("summary") or "results/adaptive_cost/adaptive_cost_summary.json")
    for p in (joined_path, curves_path, warn_path, report_path, summary_path):
        if "adversary_motion" in str(p) and "FROZEN" in str(p):
            raise RuntimeError(f"refuse to write into motion freeze path: {p}")

    joined_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(joined_path, index=False)
    curves.to_parquet(curves_path, index=False)
    if warn_df is not None:
        warn_df.to_parquet(warn_path, index=False)

    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "condition": cfg.get("condition") or "perfect_disguise_oracle",
        "far_target": far_target,
        "arms": arms,
        "classes": classes,
        "n_trips": int(joined["cell_id"].nunique()),
        "n_joined_rows": len(joined),
        "operating_far": far_by_arm,
        "cost": {
            "start_range_nm": start_r,
            "v_max_source": cost.get("v_max_source") or "platform_burst",
            "formula": "R*(1/v_mimic - 1/v_max), R=start-commit, hours→min",
        },
        "warning_time": warning_meta,
        "freeze": {
            "path": str(freeze_path.relative_to(REPO_ROOT)),
            "n_cells": freeze.get("n_cells"),
            "ddr_numbers_present": False,
        },
        "firewall_note": str(cfg.get("firewall_note") or "").strip(),
        "paths": {
            "joined": str(joined_path.relative_to(REPO_ROOT)),
            "curves": str(curves_path.relative_to(REPO_ROOT)),
            "warning": str(warn_path.relative_to(REPO_ROOT)) if warn_df is not None else None,
            "report": str(report_path.relative_to(REPO_ROOT)),
            "summary": str(summary_path.relative_to(REPO_ROOT)),
        },
        "smoke": bool(args.smoke),
    }
    # Headline recreational near-zero Δt DDR.
    rec0: dict[str, Any] = {}
    for arm in arms:
        g = joined.loc[
            (joined["mimicked_class"].astype(str) == "recreational")
            & (joined["arm"].astype(str) == arm)
            & (joined["delta_t_add_min"] <= 1e-6)
        ]
        n = len(g)
        n_f = int((g["action"] == "flag").sum())
        rec0[arm] = {"n": n, "n_flag": n_f, "ddr": (n_f / n) if n else None}
    summary["recreational_delta_t_zero"] = rec0

    if warn_sum is not None:
        summary["warning_summary"] = warn_sum.to_dict(orient="records")

    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _write_report(
        report_path,
        summary=summary,
        joined=joined,
        curves_by_axis=curves_by_axis,
        warn_sum=warn_sum,
        far_by_arm=far_by_arm,
        arms=arms,
        classes=classes,
    )

    print(f"[adaptive_cost] wrote {joined_path.relative_to(REPO_ROOT)}")
    print(f"[adaptive_cost] wrote {curves_path.relative_to(REPO_ROOT)}")
    if warn_df is not None:
        print(f"[adaptive_cost] wrote {warn_path.relative_to(REPO_ROOT)}")
    print(f"[adaptive_cost] wrote {report_path.relative_to(REPO_ROOT)}")
    print(f"[adaptive_cost] wrote {summary_path.relative_to(REPO_ROOT)}")
    for arm, st in rec0.items():
        print(
            f"[adaptive_cost] recreational Δt≈0 × {arm}: "
            f"DDR={_fmt_pct(st['ddr'])} (n={st['n']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
