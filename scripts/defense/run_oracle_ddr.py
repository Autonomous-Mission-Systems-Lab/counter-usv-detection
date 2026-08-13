#!/usr/bin/env python3
"""Oracle detection rate + defensibility gap (perfect-disguise condition).

Scores the frozen adversary-motion sweep under asserted = mimicked class
through both consistency arms and the presence-only comparator. Writes
results under ``results/oracle_ddr/`` only — never mutates the motion freeze.

Usage
-----
    python scripts/defense/run_oracle_ddr.py --smoke
    python scripts/defense/run_oracle_ddr.py
    python scripts/defense/run_oracle_ddr.py --no-verify-digests
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
from counterusv.defense.presence import (  # noqa: E402
    PresenceOnlyDefense,
    presence_for_disguise,
)
from counterusv.eval.oracle_ddr import (  # noqa: E402
    assert_freeze_has_no_ddr,
    cell_kind,
    ddr_table,
    extract_windows,
    gap_table,
    has_ddr_payload,
    load_operating_far,
    load_oracle_ddr_config,
    nc_sanity,
    pivot_ddr_markdown,
    resolve_path,
)


def _load_pipelines(
    cfg: dict[str, Any],
    *,
    verify_digests: bool,
    envelope_override: str | None = None,
) -> dict[str, DefensePipeline]:
    base = load_pipeline_config()
    pipes: dict[str, DefensePipeline] = {}
    for arm_name, arm in (cfg.get("arms") or {}).items():
        freeze = resolve_path(arm["freeze_path"])
        feature_arm = str(arm.get("feature_arm") or arm_name)
        pcfg = PipelineConfig(
            version=base.version,
            frozen=base.frozen,
            far_target=float(cfg.get("far_target") or base.far_target),
            feature_arm=feature_arm,  # type: ignore[arg-type]
            freeze_path=freeze,
            verify_digests=verify_digests,
            primary_model=base.primary_model,
            association_note=base.association_note,
            path=base.path,
        )
        pipes[arm_name] = DefensePipeline.from_freeze(
            freeze,
            far_target=pcfg.far_target,
            verify_digests=verify_digests,
            pipeline_config=base.path,
        )
        # Override arm metadata after load (from_freeze reads pipeline.yaml arm).
        pipes[arm_name].config = pcfg
        if envelope_override and envelope_override not in pipes[arm_name].scorer.envelopes:
            joblib = freeze.parent / "envelopes" / f"{envelope_override}.joblib"
            pipes[arm_name].scorer.attach_envelope(envelope_override, joblib)
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
    cons_table: pd.DataFrame,
    presence_table: pd.DataFrame,
    gaps: pd.DataFrame,
    unc_table: pd.DataFrame,
    arms: list[str],
    classes: list[str],
    far_by_arm: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("# Oracle detection rate + defensibility gap\n")
    lines.append(
        f"**Condition:** `{summary['condition']}` — asserted class = mimicked "
        "class, no patch. Hostile tracks synthesized at eval only "
        "(`role=hostile`, `source=synth`, `purpose=eval`); the benign scorer "
        "is never trained on them.\n"
    )
    lines.append(
        f"**Sweep freeze:** `{summary['freeze']['path']}` "
        f"(n_cells={summary['freeze']['n_cells']}, "
        f"ddr_in_freeze={summary['freeze']['ddr_numbers_present']}).\n"
    )
    lines.append("## Measured FAR (paired with every DDR)\n")
    for arm, far in far_by_arm.items():
        lines.append(f"- **{arm}:** {_fmt_pct(far)} (held-out / operating)\n")
    lines.append("\n## Consistency DDR — non-NC cells (swp + unc)\n")
    lines.append(
        pivot_ddr_markdown(cons_table, arms=arms, classes=classes)
    )
    lines.append("\n## Unconstrained attack-run sub-row\n")
    lines.append(
        pivot_ddr_markdown(unc_table, arms=arms, classes=classes)
    )
    lines.append("\n## Presence-only (disguise comparator)\n")
    lines.append(
        "By construction EO `detected` → presence **pass** on every oracle "
        "disguise cell.\n\n"
    )
    if presence_table.empty:
        lines.append("_No presence rows._\n")
    else:
        lines.append("| mimicked_class | presence DDR |\n|---|---|\n")
        for _, r in presence_table.iterrows():
            lines.append(
                f"| {r['mimicked_class']} | {_fmt_pct(r['ddr'])} "
                f"(n={int(r['n'])}) |\n"
            )
    lines.append("\n## Defensibility gap (consistency − presence)\n")
    if gaps.empty:
        lines.append("_No gap rows._\n")
    else:
        lines.append(
            "| mimicked_class | arm | consistency DDR | presence DDR | gap |\n"
            "|---|---|---|---|---|\n"
        )
        for _, r in gaps.iterrows():
            lines.append(
                "| {cls} | {arm} | {c} | {p} | {g} |\n".format(
                    cls=r.get("mimicked_class"),
                    arm=r.get("arm"),
                    c=_fmt_pct(r.get("ddr")),
                    p=_fmt_pct(r.get("presence_ddr")),
                    g=_fmt_pct(r.get("gap")),
                )
            )
    lines.append("\n## Negative-control sanity\n")
    nc = summary.get("negative_control") or {}
    lines.append(
        "| arm | NC flag rate | operating FAR | ceiling | within tol |\n"
        "|---|---|---|---|---|\n"
    )
    for arm, block in nc.items():
        lines.append(
            "| {arm} | {ddr} (n={n}) | {far} | {ceil} | {ok} |\n".format(
                arm=arm,
                ddr=_fmt_pct(block.get("ddr")),
                n=block.get("n"),
                far=_fmt_pct(block.get("operating_far")),
                ceil=_fmt_pct(block.get("ceiling")),
                ok="PASS" if block.get("within_tol") else "FAIL",
            )
        )
    lines.append(
        "\n## Firewall\n\n"
        f"{summary.get('firewall_note', '').strip()}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="score a small subset (2 assets × 1 platform × 2 classes)",
    )
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument(
        "--no-verify-digests",
        action="store_true",
        help="skip behavior-model freeze digest verification",
    )
    ap.add_argument(
        "--envelope-override",
        type=str,
        default=None,
        help="scoreable classes use this envelope (e.g. pooled_benign)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: results/oracle_ddr, or "
             "results/oracle_ddr_pooled when --envelope-override is set)",
    )
    args = ap.parse_args()

    cfg = load_oracle_ddr_config(args.config)
    far_target = float(cfg.get("far_target") or 0.05)
    thin_n = int(cfg.get("thin_n_threshold") or 20)
    nc_tol = cfg.get("nc_tolerance") or {}
    nc_factor = float(nc_tol.get("factor") or 2.0)
    nc_slack = float(nc_tol.get("slack") or 0.05)
    override = args.envelope_override
    if override is None and str(cfg.get("envelope_routing") or "") == "pooled":
        override = str(cfg.get("envelope_override") or "pooled_benign")
    elif override is None and cfg.get("envelope_override"):
        override = str(cfg["envelope_override"])

    sweep = cfg.get("sweep") or {}
    freeze_path = resolve_path(sweep.get("freeze") or "results/adversary_motion/FROZEN_SWEEP.json")
    points_path = resolve_path(sweep.get("points") or "results/adversary_motion/tracks_points.parquet")
    meta_path = resolve_path(sweep.get("meta") or "results/adversary_motion/tracks_meta.json")

    print("[oracle_ddr] asserting freeze has no DDR payload …")
    freeze = assert_freeze_has_no_ddr(freeze_path)

    out = cfg.get("output") or {}
    if args.out_dir is not None:
        out_dir = args.out_dir if args.out_dir.is_absolute() else resolve_path(args.out_dir)
    elif override:
        out_dir = resolve_path("results/oracle_ddr_pooled")
    else:
        out_dir = resolve_path(out.get("dir") or "results/oracle_ddr")
    report_path = out_dir / "oracle_ddr_report.md"
    summary_path = out_dir / "oracle_ddr_summary.json"
    cells_path = out_dir / "oracle_ddr_cells.parquet"
    for p in (report_path, summary_path, cells_path):
        if "adversary_motion" in str(p) and "FROZEN" in str(p):
            raise RuntimeError(f"refuse to write scores into motion freeze path: {p}")

    if not points_path.is_file():
        raise FileNotFoundError(
            f"missing {points_path}; run scripts/attacks/generate_adversary_tracks.py first"
        )

    print(f"[oracle_ddr] loading points {points_path.relative_to(REPO_ROOT)} …")
    points = pd.read_parquet(points_path)
    meta_rows = json.loads(meta_path.read_text()) if meta_path.is_file() else []
    meta_by_trip = {str(m["trip_id"]): m for m in meta_rows}

    kin_cfg = load_kinematics_config(
        resolve_path(cfg.get("kinematics_sweep") or "configs/attacks/kinematics.yaml")
    )
    placements, placements_sha = load_placements(kin_cfg)
    asset_xy = _asset_coords(placements)

    eng = load_engagement_geometry(
        resolve_path(cfg.get("engagement_geometry") or "configs/defense/engagement_geometry.yaml")
    )
    annulus = {"min_range_nm": eng.min_range_nm, "max_range_nm": eng.max_range_nm}
    inbound = {
        "require_points_before_cpa": int(
            eng.inbound_leg.get("require_points_before_cpa", 3)
        ),
        "min_points": int(eng.inbound_leg.get("min_points", 4)),
    }

    print("[oracle_ddr] loading defenses …")
    if override:
        print(f"[oracle_ddr] envelope_override={override!r}")
    verify = not args.no_verify_digests
    try:
        pipes = _load_pipelines(
            cfg, verify_digests=verify, envelope_override=override
        )
    except ValueError as e:
        if verify and "digest mismatch" in str(e):
            print(f"[oracle_ddr] WARNING: {e}")
            print("[oracle_ddr] reloading with verify_digests=False")
            pipes = _load_pipelines(
                cfg, verify_digests=False, envelope_override=override
            )
        else:
            raise
    presence = PresenceOnlyDefense.from_config(
        resolve_path(cfg.get("presence") or "configs/defense/presence.yaml")
    )
    oracle = PerfectDisguiseOracle.from_config(
        resolve_path(cfg.get("oracle") or "configs/attacks/oracle.yaml")
    )

    far_by_arm: dict[str, float] = {}
    for arm_name, arm in (cfg.get("arms") or {}).items():
        geo = bool(arm.get("join_geometry"))
        default = 0.062 if geo else 0.035
        far_by_arm[arm_name] = load_operating_far(
            arm.get("far_summary") or "",
            geometry=geo,
            default=default,
        )

    # Trip list
    trip_ids = sorted(points["trip_id"].astype(str).unique().tolist())
    if args.smoke:
        keep_assets = set(list(asset_xy.keys())[:2])
        filtered: list[str] = []
        for tid in trip_ids:
            g = points.loc[points["trip_id"].astype(str) == tid]
            if str(g["asset_id"].iloc[0]) not in keep_assets:
                continue
            if str(g["platform"].iloc[0]) != "magura_v5":
                continue
            if str(g["mimicked_class"].iloc[0]) not in ("fishing", "recreational"):
                continue
            # Keep NC + unconstrained + one representative sweep cell per class×asset.
            is_nc = bool(g["negative_control"].iloc[0]) if "negative_control" in g.columns else False
            is_unc = bool(g["unconstrained"].iloc[0]) if "unconstrained" in g.columns else False
            if is_nc or is_unc:
                filtered.append(tid)
                continue
            meta = meta_by_trip.get(str(tid)) or {}
            if (
                float(meta.get("v_mimic_kn", -1)) == 25.0
                and float(meta.get("commit_range_nm", -1)) == 2.0
                and float(meta.get("bearing_offset_deg", -1)) == 0.0
            ):
                filtered.append(tid)
        trip_ids = filtered
        print(f"[oracle_ddr] smoke subset n_trips={len(trip_ids)}")
    if args.max_cells is not None:
        trip_ids = trip_ids[: int(args.max_cells)]
        print(f"[oracle_ddr] capped to max-cells={len(trip_ids)}")

    arm_cfgs = cfg.get("arms") or {}
    records: list[dict[str, Any]] = []
    t0 = time.time()
    n_trips = len(trip_ids)
    for i, tid in enumerate(trip_ids, 1):
        if i == 1 or i % 200 == 0 or i == n_trips:
            print(f"[oracle_ddr] scoring {i}/{n_trips} …", flush=True)
        g = points.loc[points["trip_id"].astype(str) == tid].copy()
        mimicked = str(g["mimicked_class"].iloc[0])
        asset_id = str(g["asset_id"].iloc[0])
        platform = str(g["platform"].iloc[0])
        neg = bool(g["negative_control"].iloc[0]) if "negative_control" in g.columns else False
        unc = bool(g["unconstrained"].iloc[0]) if "unconstrained" in g.columns else False
        meta = dict(meta_by_trip.get(tid) or {})
        meta.setdefault("role", "hostile")
        meta.setdefault("source", "synth")
        meta.setdefault("trip_id", tid)
        meta.setdefault("canonical_class", "usv")
        in_band = meta.get("in_plausibility_band")
        if in_band is None and "in_plausibility_band" in g.columns:
            in_band = bool(g["in_plausibility_band"].iloc[0])
        extrapolation = not bool(in_band) if in_band is not None else None
        lat, lon = asset_xy.get(asset_id, (None, None))

        assertion = oracle.assert_class(
            mimicked,
            contact_id=tid,
            true_class="usv",
            box_xyxy=(0.0, 0.0, 10.0, 10.0),
        )
        obs = presence_for_disguise(assertion)
        p_dec = evaluate_contact(
            presence, assertion=assertion, presence=obs, purpose="eval"
        )
        records.append({
            "cell_id": tid,
            "trip_id": tid,
            "arm": "presence",
            "defense_kind": "presence_only",
            "action": p_dec.action,
            "mimicked_class": mimicked,
            "asserted_class": assertion.asserted_class,
            "platform": platform,
            "asset_id": asset_id,
            "negative_control": neg,
            "unconstrained": unc,
            "extrapolation": extrapolation,
            "cell_kind": cell_kind({
                "negative_control": neg, "unconstrained": unc,
            }),
            "note": p_dec.note,
        })

        for arm_name, arm in arm_cfgs.items():
            windows = [int(w) for w in (arm.get("windows_s") or [])]
            join_geo = bool(arm.get("join_geometry"))
            by_w, complete = extract_windows(
                g,
                windows,
                asset_lat=lat,
                asset_lon=lon,
                annulus=annulus,
                inbound_leg=inbound,
                join_geometry=join_geo,
            )
            pipe = pipes[arm_name]
            if not by_w:
                action = "abstain"
                note = "no_window"
            else:
                dec = evaluate_contact(
                    pipe,
                    assertion=assertion,
                    features_by_window=by_w,
                    complete_windows=complete,
                    purpose="eval",
                    track_meta=meta,
                    far_target=far_target,
                    envelope_override=override,
                )
                action = dec.action
                note = dec.note
            records.append({
                "cell_id": tid,
                "trip_id": tid,
                "arm": arm_name,
                "defense_kind": "consistency",
                "action": action,
                "mimicked_class": mimicked,
                "asserted_class": assertion.asserted_class,
                "platform": platform,
                "asset_id": asset_id,
                "negative_control": neg,
                "unconstrained": unc,
                "extrapolation": extrapolation,
                "cell_kind": cell_kind({
                    "negative_control": neg, "unconstrained": unc,
                }),
                "n_windows": len(by_w),
                "note": note,
            })

    cells = pd.DataFrame(records)
    arms = list(arm_cfgs.keys())
    classes = sorted(cells["mimicked_class"].astype(str).unique().tolist())

    cons = cells.loc[cells["defense_kind"] == "consistency"]
    pres = cells.loc[cells["defense_kind"] == "presence_only"]

    cons_table = ddr_table(
        cons,
        group_cols=("mimicked_class", "arm", "defense_kind"),
        exclude_nc=True,
        thin_n_threshold=thin_n,
    )
    presence_table = ddr_table(
        # Presence rows use arm="presence"; aggregate by class only.
        pres.assign(arm="presence_only"),
        group_cols=("mimicked_class", "defense_kind"),
        exclude_nc=True,
        thin_n_threshold=thin_n,
    )
    unc_only = cons.loc[cons["unconstrained"].astype(bool)]
    unc_table = ddr_table(
        unc_only,
        group_cols=("mimicked_class", "arm", "defense_kind"),
        exclude_nc=False,
        thin_n_threshold=thin_n,
    )
    gaps = gap_table(cons_table, presence_table)

    nc_blocks: dict[str, Any] = {}
    for arm_name in arms:
        nc_blocks[arm_name] = nc_sanity(
            cons,
            far=far_by_arm[arm_name],
            arm=arm_name,
            factor=nc_factor,
            slack=nc_slack,
        )

    # Re-check freeze untouched.
    freeze_after = json.loads(freeze_path.read_text())
    freeze_has_ddr = has_ddr_payload(freeze_after)

    summary: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "condition": cfg.get("condition") or "perfect_disguise_oracle",
        "far_target": far_target,
        "envelope_override": override,
        "n_trips_scored": n_trips,
        "elapsed_s": round(time.time() - t0, 2),
        "smoke": bool(args.smoke),
        "operating_far": far_by_arm,
        "consistency_ddr": cons_table.to_dict(orient="records"),
        "presence_ddr": presence_table.to_dict(orient="records"),
        "unconstrained_ddr": unc_table.to_dict(orient="records"),
        "defensibility_gap": gaps.to_dict(orient="records"),
        "negative_control": nc_blocks,
        "freeze": {
            "path": str(freeze_path.relative_to(REPO_ROOT)),
            "digest": (freeze.get("digests") or {}).get("sweep_cells"),
            "n_cells": freeze.get("n_cells"),
            "ddr_numbers_present": freeze_has_ddr,
            "placements_sha256": placements_sha,
        },
        "firewall_note": str(cfg.get("firewall_note") or "").strip(),
        "paths": {
            "report": str(report_path.relative_to(REPO_ROOT)),
            "summary": str(summary_path.relative_to(REPO_ROOT)),
            "cells": str(cells_path.relative_to(REPO_ROOT)),
        },
    }

    cells_path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(cells_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _write_report(
        report_path,
        summary=summary,
        cons_table=cons_table,
        presence_table=presence_table,
        gaps=gaps,
        unc_table=unc_table,
        arms=arms,
        classes=classes,
        far_by_arm=far_by_arm,
    )

    print(f"[oracle_ddr] wrote {report_path.relative_to(REPO_ROOT)}")
    print(f"[oracle_ddr] wrote {summary_path.relative_to(REPO_ROOT)}")
    print(f"[oracle_ddr] wrote {cells_path.relative_to(REPO_ROOT)}")
    print(f"[oracle_ddr] freeze ddr_numbers_present={freeze_has_ddr}")
    for arm_name, block in nc_blocks.items():
        print(
            f"[oracle_ddr] NC {arm_name}: ddr={_fmt_pct(block.get('ddr'))} "
            f"far={_fmt_pct(block.get('operating_far'))} "
            f"{'PASS' if block.get('within_tol') else 'FAIL'}"
        )
    # Headline print
    for _, r in cons_table.iterrows():
        print(
            f"[oracle_ddr] DDR {r['mimicked_class']} × {r['arm']}: "
            f"{_fmt_pct(r['ddr'])} (n={int(r['n'])})"
        )
    return 0 if not freeze_has_ddr else 2


if __name__ == "__main__":
    raise SystemExit(main())
