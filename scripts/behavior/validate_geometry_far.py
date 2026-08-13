#!/usr/bin/env python3
"""Placement-swept FAR validation for the kinematics_geometry arm.

Scores held-out benign encounter pairs under the frozen extended envelopes,
across the **same digested placements table** used for fit. Reports FAR as a
distribution over ``placement_class`` × ``port_region`` (never a single asset).

Operating-point claim = fit-population placements (berth_approach, anchorage).
``fairway_stress`` / ``offshore_terminal`` are sensitivity strata.

Gate: operating-point overall test FAR@5% must not exceed the kinematics-only
arm's overall test FAR@5% by more than ``far_gate.max_absolute_excess``.

Thresholds are the **val-calibrated fit thresholds** (fit-population); they are
not retuned on stress placements — that would hide placement sensitivity.

Usage
-----
    python scripts/behavior/validate_geometry_far.py
    python scripts/behavior/validate_geometry_far.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "behavior"))

from counterusv.defense import ConsistencyScorer  # noqa: E402
from counterusv.kinematics.behavior_model import (  # noqa: E402
    EnvelopeModel,
    MultiHorizonEnvelope,
    load_envelope,
    select_envelope_rows,
)
from fit_geometry_model import load_joined_window  # noqa: E402
from validate_benign_far import (  # noqa: E402
    FAR_SWEEP,
    empirical_far,
    far_curve,
    far_key,
    score_rows_batch,
)

DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "behavior_model_geometry.yaml"
DEFAULT_FREEZE = REPO_ROOT / "results" / "behavior_model_geometry" / "FROZEN.json"
DEFAULT_OUT = REPO_ROOT / "results" / "behavior_model_geometry"
CONTACT_COLS = ("trip_id", "asset_id")


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


def _pct(x: Any) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) and np.isfinite(x) else "—"


def verify_placements_digest(freeze: dict) -> dict[str, Any]:
    """Confirm fit and FAR score against the same placements table."""
    cfg_block = (freeze.get("configs") or {}).get("placements") or {}
    path = REPO_ROOT / (cfg_block.get("path") or "data/defense/placements.parquet")
    exp = cfg_block.get("sha256")
    got = _sha256_file(path)
    ok = (exp is None) or (got == exp)
    digest_path = REPO_ROOT / "data" / "defense" / "placements_digest.json"
    digest = json.loads(digest_path.read_text()) if digest_path.is_file() else {}
    return {
        "ok": ok,
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": got,
        "expected": exp,
        "n_rows": digest.get("n_rows"),
        "n_valid": digest.get("n_valid"),
        "fit_population": digest.get("fit_population"),
    }


def contact_key(df: pd.DataFrame) -> pd.Series:
    return df["trip_id"].astype(str) + "||" + df["asset_id"].astype(str)


def assign_multihorizon_encounters(
    bundle: MultiHorizonEnvelope,
    tables: dict[int, pd.DataFrame],
    members: dict,
    split: str,
) -> pd.DataFrame:
    """One row per (trip, asset): features from the longest complete horizon."""
    windows = bundle.available_windows()
    by_w: dict[int, pd.DataFrame] = {}
    complete: dict[int, set[str]] = {}
    universe: set[str] = set()
    for w in windows:
        d = tables[w]
        d = d.loc[d["split"] == split] if "split" in d.columns else d
        d = select_envelope_rows(d, members)
        if d.empty:
            by_w[w] = d
            complete[w] = set()
            continue
        d = d.copy()
        d["_ck"] = contact_key(d)
        d = d.drop_duplicates("_ck").set_index("_ck")
        by_w[w] = d
        if "window_complete" in d.columns:
            complete[w] = set(d.index[d["window_complete"].astype(bool)])
        else:
            complete[w] = set(d.index)
        universe |= set(d.index)

    rows: list[dict[str, Any]] = []
    for ck in universe:
        cw = [w for w in windows if ck in complete[w]]
        if not cw:
            continue
        w = max(cw)
        r = by_w[w].loc[ck]
        rec = r.to_dict()
        rec["_ck"] = ck
        rec["assigned_window_s"] = w
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate_envelope_split(
    bundle: MultiHorizonEnvelope,
    tables: dict[int, pd.DataFrame],
    members: dict,
    split: str,
    *,
    far_ops: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score one envelope on one split; return summary + per-row score table."""
    assigned = assign_multihorizon_encounters(bundle, tables, members, split)
    empty_summary = {
        "n_assigned": 0,
        "n_scored": 0,
        "coverage": None,
        "at_calibrated": {far_key(f): None for f in far_ops},
        "self_far_curve": [],
    }
    if assigned.empty:
        return empty_summary, pd.DataFrame()

    score_rows: list[dict[str, Any]] = []
    for w, sub in assigned.groupby("assigned_window_s"):
        env = bundle.horizons[int(w)]
        sc, keep_idx, subs = score_rows_batch(env, sub, model_name="gmm")
        if len(sc) == 0:
            continue
        kept = sub.loc[list(keep_idx)]
        for i, score in enumerate(sc):
            sub_name = subs[i] or "core"
            fit = (env.subspaces.get(sub_name)
                   or env.subspaces.get("core") or {}).get("gmm")
            row: dict[str, Any] = {
                "score": float(score),
                "subspace": sub_name,
                "assigned_window_s": int(w),
                "placement_class": kept.iloc[i].get("placement_class"),
                "port_region": kept.iloc[i].get("port_region"),
                "asset_id": kept.iloc[i].get("asset_id"),
                "trip_id": kept.iloc[i].get("trip_id"),
                "canonical_class": kept.iloc[i].get("canonical_class"),
            }
            for far in far_ops:
                thr = fit.thresholds.get(far_key(far)) if fit else None
                row[f"thr_{far_key(far)}"] = thr
                row[f"flag_{far_key(far)}"] = (
                    bool(score > thr) if thr is not None else None
                )
            score_rows.append(row)

    detail = pd.DataFrame(score_rows)
    n = len(detail)
    scores_arr = detail["score"].to_numpy(dtype="float64") if n else np.array([])
    at_cal = {}
    for far in far_ops:
        k = far_key(far)
        col = f"flag_{k}"
        if n and col in detail.columns and detail[col].notna().any():
            at_cal[k] = float(detail[col].dropna().mean())
        else:
            at_cal[k] = None
    summary = {
        "n_assigned": int(len(assigned)),
        "n_scored": n,
        "coverage": (n / len(assigned)) if len(assigned) else None,
        "score_median": float(np.median(scores_arr)) if n else None,
        "score_p95": float(np.percentile(scores_arr, 95)) if n else None,
        "at_calibrated": at_cal,
        "self_far_curve": far_curve(scores_arr, FAR_SWEEP),
    }
    return summary, detail


def stratum_far(
    detail: pd.DataFrame,
    *,
    group_cols: list[str],
    far_op: float = 0.05,
) -> list[dict[str, Any]]:
    if detail.empty:
        return []
    flag_col = f"flag_{far_key(far_op)}"
    if flag_col not in detail.columns:
        return []
    rows = []
    for keys, sub in detail.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = int(sub[flag_col].notna().sum())
        flagged = int(sub[flag_col].fillna(False).sum()) if n else 0
        rec = {c: keys[i] for i, c in enumerate(group_cols)}
        rec.update({
            "n_scored": n,
            "n_flagged": flagged,
            "far": (flagged / n) if n else None,
        })
        rows.append(rec)
    return rows


def load_kinematics_far_ref(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return (
        ((data.get("by_split") or {}).get("test") or {})
        .get("overall", {})
        .get("at_calibrated", {})
        .get("far_0.05")
    )


def write_report(path: Path, payload: dict) -> None:
    gate = payload.get("gate") or {}
    lines: list[str] = [
        "# Geometry-arm FAR across asset placements",
        "",
        f"Generated: {payload['timestamp']}",
        "",
        "Held-out benign FAR for the `kinematics_geometry` arm, swept over the "
        "**frozen placements table** (same SHA-256 pin as the envelope fit). "
        "FAR is a **distribution over placements**, not a single asset number.",
        "",
        f"Primary horizon: **{payload['primary_window_s']} s**; multi-horizon "
        f"policy: longest complete among {payload['windows_s']} s. "
        f"Default FAR target: **{100 * payload['default_far']:.0f}%**. "
        "Thresholds = val-calibrated fit thresholds (fit-population); "
        "**not** retuned on stress placements.",
        "",
        "## Placements pin",
        "",
        f"- Digest check: **{'PASS' if payload['placements_pin']['ok'] else 'FAIL'}**",
        f"- `placements.parquet` sha256: `{payload['placements_pin']['sha256'][:16]}…`",
        f"- Valid rows: {payload['placements_pin'].get('n_valid')} / "
        f"{payload['placements_pin'].get('n_rows')}",
        f"- Fit population: `{payload['placements_pin'].get('fit_population')}`",
        "",
        "## Gate vs kinematics-only arm",
        "",
        f"- Kinematics-only overall test FAR@5%: "
        f"**{_pct(gate.get('kinematics_far'))}**",
        f"- Geometry operating-point overall test FAR@5% "
        f"(placements `{gate.get('operating_placements')}`): "
        f"**{_pct(gate.get('geometry_operating_far'))}** "
        f"(n={gate.get('geometry_operating_n', 0):,})",
        f"- Max absolute excess allowed: "
        f"**{100 * float(gate.get('max_absolute_excess') or 0):.1f} pp**",
        f"- Ceiling: **{_pct(gate.get('ceiling'))}**",
        f"- Verdict: **{gate.get('verdict', '—')}**"
        + (
            f" — {gate.get('reason')}" if gate.get("reason") else ""
        ),
        "",
        "## Operating-point FAR (fit-population placements, multi-horizon GMM)",
        "",
        "These placements train the envelopes and carry the operating-point claim.",
        "",
    ]

    for split in ("val", "test"):
        block = (payload.get("by_split") or {}).get(split) or {}
        lines += [
            f"### {split}",
            "",
            "| envelope | n_scored | FAR@1% | FAR@5% | FAR@10% |",
            "|---|---:|---:|---:|---:|",
        ]
        for env, row in (block.get("envelopes_operating") or {}).items():
            ops = row.get("at_calibrated") or {}
            lines.append(
                f"| `{env}` | {row.get('n_scored', 0):,} | "
                f"{_pct(ops.get('far_0.01'))} | "
                f"{_pct(ops.get('far_0.05'))} | "
                f"{_pct(ops.get('far_0.1'))} |"
            )
        overall = block.get("overall_operating") or {}
        ops = overall.get("at_calibrated") or {}
        lines.append(
            f"| **overall** | {overall.get('n_scored', 0):,} | "
            f"{_pct(ops.get('far_0.01'))} | "
            f"{_pct(ops.get('far_0.05'))} | "
            f"{_pct(ops.get('far_0.1'))} |"
        )
        lines.append("")

    # Placement-class sensitivity (test)
    lines += [
        "## Placement-class sensitivity (test, FAR@5%)",
        "",
        "| placement_class | role | n_scored | FAR@5% | note |",
        "|---|---|---:|---:|---|",
    ]
    notes = {
        "berth_approach": "operating / fit",
        "anchorage": "operating / fit",
        "offshore_terminal": "sensitivity — sparse traffic",
        "fairway_stress": "sensitivity — pessimistic bound",
    }
    roles = {
        "berth_approach": "fit",
        "anchorage": "fit",
        "offshore_terminal": "far_only",
        "fairway_stress": "far_only",
    }
    for row in payload.get("by_placement_class_test") or []:
        pc = row.get("placement_class")
        lines.append(
            f"| `{pc}` | {roles.get(pc, '—')} | {row.get('n_scored', 0):,} | "
            f"{_pct(row.get('far'))} | {notes.get(pc, '')} |"
        )
    lines.append("")

    # Port region
    lines += [
        "## Port-region sensitivity (test, FAR@5%, operating placements)",
        "",
        "| port_region | n_scored | FAR@5% |",
        "|---|---:|---:|",
    ]
    for row in payload.get("by_port_region_test") or []:
        lines.append(
            f"| `{row.get('port_region')}` | {row.get('n_scored', 0):,} | "
            f"{_pct(row.get('far'))} |"
        )
    lines.append("")

    # Envelope × placement
    lines += [
        "## Envelope × placement_class (test, FAR@5%)",
        "",
        "| envelope | berth_approach | anchorage | offshore_terminal | fairway_stress |",
        "|---|---:|---:|---:|---:|",
    ]
    pcs = ["berth_approach", "anchorage", "offshore_terminal", "fairway_stress"]
    by_ep = payload.get("by_envelope_placement_test") or {}
    for env in by_ep:
        cells = []
        for pc in pcs:
            cell = (by_ep[env] or {}).get(pc) or {}
            n = cell.get("n_scored") or 0
            far = cell.get("far")
            if n:
                cells.append(f"{_pct(far)} (n={n})")
            else:
                cells.append("—")
        lines.append(f"| `{env}` | " + " | ".join(cells) + " |")
    lines.append("")

    # Watch cells
    lines += [
        "## Watch cells",
        "",
        "Class conditioning should absorb routine fairway-like closures "
        "(`cargo_merchant`) and not over-flag craft whose job is approaching "
        "vessels (`working_service`). `passenger_ferry` was elevated at fit.",
        "",
        "| envelope | operating FAR@5% | fairway_stress FAR@5% | note |",
        "|---|---:|---:|---|",
    ]
    for env in (payload.get("watch_envelopes") or []):
        op = ((payload.get("by_split") or {}).get("test") or {}).get(
            "envelopes_operating", {}
        ).get(env, {})
        fw = ((by_ep.get(env) or {}).get("fairway_stress") or {})
        note = ""
        if env == "passenger_ferry":
            note = "known elevated at fit"
        elif env == "cargo_merchant":
            note = "routine-transit absorption check"
        elif env == "working_service":
            note = "approach-as-business check"
        lines.append(
            f"| `{env}` | {_pct((op.get('at_calibrated') or {}).get('far_0.05'))} | "
            f"{_pct(fw.get('far'))} | {note} |"
        )
    lines.append("")

    # All-placements overall (report, not gate)
    all_far = payload.get("all_placements_test_far")
    lines += [
        "## All-placements overall (report-only)",
        "",
        f"Test FAR@5% across every valid placement (incl. stress): "
        f"**{_pct((all_far or {}).get('far'))}** "
        f"(n={(all_far or {}).get('n_scored', 0):,}). "
        "Not the operating-point claim.",
        "",
        "## Notes",
        "",
        "- Contact unit is `(trip_id, asset_id)` — one track may contribute "
        "multiple encounters across placements.",
        "- Thresholds come from the 5.4 fit (val, fit-population); this step "
        "does not retune them. Drift of realized FAR from the 5% target is "
        "expected under vessel-disjoint + placement shift.",
        "- `fairway_stress` is the pessimistic bound: benign traffic has every "
        "reason to pass near a channel point. Elevated FAR there is informative, "
        "not automatically a gate failure.",
        "- `offshore_terminal` is the sparse-traffic case; two of five seed "
        "placements failed the water gate and contribute 0 rows.",
        "- Fallback if the gate fails: kinematics-only remains the headline; "
        "geometry stays a pilot + future work.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _filter_detail(
    detail: pd.DataFrame, placements: list[str] | None,
) -> pd.DataFrame:
    if detail.empty or not placements:
        return detail
    return detail.loc[detail["placement_class"].isin(placements)].copy()


def _summary_from_detail(
    detail: pd.DataFrame, far_ops: list[float], n_assigned: int | None = None,
) -> dict[str, Any]:
    n = len(detail)
    at_cal = {}
    for far in far_ops:
        k = far_key(far)
        col = f"flag_{k}"
        if n and col in detail.columns and detail[col].notna().any():
            at_cal[k] = float(detail[col].dropna().mean())
        else:
            at_cal[k] = None
    assigned = n_assigned if n_assigned is not None else n
    return {
        "n_assigned": assigned,
        "n_scored": n,
        "coverage": (n / assigned) if assigned else None,
        "at_calibrated": at_cal,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="subsample joined rows for a fast wiring check")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    emap = _load_yaml(
        REPO_ROOT / (cfg.get("envelope_map")
                     or "configs/defense/class_envelope_map.yaml")
    )
    gate_cfg = cfg.get("far_gate") or {}
    operating = list(
        gate_cfg.get("operating_placements")
        or cfg.get("fit_placement_classes")
        or ["berth_approach", "anchorage"]
    )
    sensitivity = list(
        gate_cfg.get("sensitivity_placements")
        or ["offshore_terminal", "fairway_stress"]
    )
    max_excess = float(gate_cfg.get("max_absolute_excess") or 0.05)
    watch = list(
        gate_cfg.get("watch_envelopes")
        or ["passenger_ferry", "working_service", "cargo_merchant"]
    )
    kin_far_path = REPO_ROOT / (
        gate_cfg.get("kinematics_far_summary")
        or "results/behavior_model/far_summary.json"
    )

    windows = [int(w) for w in (cfg.get("windows_s") or [600])]
    primary_w = int(cfg.get("window_s") or max(windows))
    default_far = float((cfg.get("calibration") or {}).get("default_far") or 0.05)
    far_ops = list(
        (cfg.get("calibration") or {}).get("far_targets") or [0.01, 0.05, 0.10]
    )

    freeze = json.loads(args.freeze.read_text())
    pin = verify_placements_digest(freeze)
    print(f"[far-geom] placements digest: {'PASS' if pin['ok'] else 'FAIL'}")
    if not pin["ok"]:
        raise SystemExit(
            f"placements digest mismatch: got {pin['sha256'][:16]}… "
            f"expected {(pin.get('expected') or '')[:16]}…"
        )

    print("[far-geom] loading ConsistencyScorer from geometry freeze …")
    scorer = ConsistencyScorer.from_freeze(
        args.freeze, verify_digests=True, far_target=default_far,
    )
    # Load pooled ablation if present on disk but absent from the EO map.
    pooled_name = (
        freeze.get("pooled_ablation")
        or (cfg.get("pooled_ablation") or {}).get("name")
        or "pooled_benign"
    )
    pooled_path = (
        REPO_ROOT / "results" / "behavior_model_geometry"
        / "envelopes" / f"{pooled_name}.joblib"
    )
    extra_envelopes: dict[str, MultiHorizonEnvelope | EnvelopeModel] = {}
    if pooled_path.is_file() and pooled_name not in scorer.envelopes:
        extra_envelopes[pooled_name] = load_envelope(pooled_path)
        print(f"[far-geom] loaded pooled ablation `{pooled_name}`")

    envelope_specs = dict(emap.get("envelopes") or {})
    if pooled_name in extra_envelopes or pooled_name in scorer.envelopes:
        pooled_members = (cfg.get("pooled_ablation") or {}).get("members") or {
            "canonical_class": [
                "fishing", "sailing", "recreational", "passenger_ferry",
                "cargo_merchant", "working_service",
            ]
        }
        envelope_specs[pooled_name] = {"members": pooled_members}

    all_envelopes = {**scorer.envelopes, **extra_envelopes}
    print(f"[far-geom] envelopes: {sorted(all_envelopes)}")

    print("[far-geom] joining kinematics × geometry (all placements) …")
    # Empty placement_classes list → no filter → all classes in the table.
    full_by_w = {
        w: load_joined_window(
            cfg, w, require_complete=False, placement_classes=[],
        )
        for w in windows
    }
    for w in windows:
        d = full_by_w[w]
        print(
            f"  {w:>4} s: n={len(d):,} "
            f"placements={sorted(d['placement_class'].dropna().unique())}"
        )

    if args.smoke:
        for w in windows:
            parts = []
            for s, n in (("val", 400), ("test", 400)):
                sub = full_by_w[w].loc[full_by_w[w]["split"] == s]
                if sub.empty:
                    continue
                parts.append(sub.sample(n=min(n, len(sub)), random_state=0))
            if parts:
                # Keep a little train so filters don't break; not scored for FAR.
                train = full_by_w[w].loc[full_by_w[w]["split"] == "train"]
                if not train.empty:
                    parts.append(train.sample(n=min(200, len(train)), random_state=0))
                full_by_w[w] = pd.concat(parts)
        print("[far-geom] smoke subsample applied")

    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.relative_to(REPO_ROOT)),
        "freeze": str(args.freeze.relative_to(REPO_ROOT)),
        "feature_arm": "kinematics_geometry",
        "primary_window_s": primary_w,
        "windows_s": windows,
        "default_far": default_far,
        "far_operating_points": far_ops,
        "operating_placements": operating,
        "sensitivity_placements": sensitivity,
        "watch_envelopes": watch,
        "placements_pin": pin,
        "by_split": {},
        "smoke": bool(args.smoke),
    }

    t0 = time.perf_counter()
    test_details: dict[str, pd.DataFrame] = {}

    for split in ("val", "test"):
        print(f"\n[far-geom] === {split} ===")
        split_block: dict[str, Any] = {
            "envelopes_all": {},
            "envelopes_operating": {},
            "overall_all": {},
            "overall_operating": {},
        }
        detail_parts: list[pd.DataFrame] = []

        for env_name, espec in envelope_specs.items():
            bundle = all_envelopes.get(env_name)
            if bundle is None or not isinstance(bundle, MultiHorizonEnvelope):
                print(f"  [skip] {env_name}: not a MultiHorizonEnvelope")
                continue
            summary_all, detail = evaluate_envelope_split(
                bundle, full_by_w, espec.get("members") or {}, split,
                far_ops=far_ops,
            )
            detail = detail.copy()
            detail["envelope"] = env_name
            detail_parts.append(detail)

            op_detail = _filter_detail(detail, operating)
            summary_op = _summary_from_detail(
                op_detail, far_ops, n_assigned=len(op_detail),
            )
            split_block["envelopes_all"][env_name] = summary_all
            split_block["envelopes_operating"][env_name] = summary_op
            print(
                f"  {env_name}: op n={summary_op['n_scored']:,}  "
                f"FAR@5%={_pct((summary_op.get('at_calibrated') or {}).get('far_0.05'))}  "
                f"| all n={summary_all['n_scored']:,}  "
                f"FAR@5%={_pct((summary_all.get('at_calibrated') or {}).get('far_0.05'))}"
            )

        if split == "test" and detail_parts:
            test_details["all"] = pd.concat(detail_parts, ignore_index=True)

        # Overall = n-weighted mean over class-conditional envelopes only
        # (exclude pooled ablation from the headline overall).
        for kind, key in (("all", "envelopes_all"), ("operating", "envelopes_operating")):
            overall_at: dict[str, float | None] = {}
            total_n = 0
            for env_name, env_row in split_block[key].items():
                if env_name == pooled_name:
                    continue
                n = env_row.get("n_scored") or 0
                total_n += n
            for far in far_ops:
                k = far_key(far)
                flag_mass = 0.0
                n_mass = 0
                for env_name, env_row in split_block[key].items():
                    if env_name == pooled_name:
                        continue
                    n = env_row.get("n_scored") or 0
                    rate = (env_row.get("at_calibrated") or {}).get(k)
                    if n and rate is not None:
                        n_mass += n
                        flag_mass += rate * n
                overall_at[k] = (flag_mass / n_mass) if n_mass else None
            split_block[f"overall_{kind}"] = {
                "n_scored": total_n,
                "at_calibrated": overall_at,
            }
        print(
            f"  overall operating FAR@5%="
            f"{_pct(split_block['overall_operating']['at_calibrated'].get('far_0.05'))}"
        )
        payload["by_split"][split] = split_block

    # Stratifications on test
    test_all = test_details.get("all", pd.DataFrame())
    # Exclude pooled from strata that feed the gate / watch cells.
    test_cc = (
        test_all.loc[test_all["envelope"] != pooled_name].copy()
        if not test_all.empty else test_all
    )
    test_op = _filter_detail(test_cc, operating)

    payload["by_placement_class_test"] = stratum_far(
        test_cc, group_cols=["placement_class"],
    )
    payload["by_port_region_test"] = stratum_far(
        test_op, group_cols=["port_region"],
    )
    ep_rows = stratum_far(
        test_cc, group_cols=["envelope", "placement_class"],
    )
    by_ep: dict[str, dict[str, dict]] = {}
    for r in ep_rows:
        by_ep.setdefault(str(r["envelope"]), {})[str(r["placement_class"])] = {
            "n_scored": r["n_scored"],
            "n_flagged": r["n_flagged"],
            "far": r["far"],
        }
    payload["by_envelope_placement_test"] = by_ep

    if not test_cc.empty and "flag_far_0.05" in test_cc.columns:
        flags = test_cc["flag_far_0.05"].dropna()
        payload["all_placements_test_far"] = {
            "far": float(flags.mean()) if len(flags) else None,
            "n_scored": int(len(flags)),
        }
    else:
        payload["all_placements_test_far"] = {"far": None, "n_scored": 0}

    # Gate
    kin_far = load_kinematics_far_ref(kin_far_path)
    geom_op = (
        ((payload["by_split"].get("test") or {}).get("overall_operating") or {})
        .get("at_calibrated", {})
        .get("far_0.05")
    )
    geom_op_n = (
        ((payload["by_split"].get("test") or {}).get("overall_operating") or {})
        .get("n_scored") or 0
    )
    ceiling = (kin_far + max_excess) if kin_far is not None else None
    if kin_far is None or geom_op is None:
        verdict, reason = "INCOMPLETE", "missing kinematics or geometry FAR"
    elif geom_op <= ceiling:
        verdict, reason = (
            "PASS",
            f"geometry {_pct(geom_op)} ≤ ceiling {_pct(ceiling)}",
        )
    else:
        verdict, reason = (
            "FAIL",
            f"geometry {_pct(geom_op)} > ceiling {_pct(ceiling)}; "
            "fallback: kinematics-only headline",
        )
    payload["gate"] = {
        "kinematics_far": kin_far,
        "geometry_operating_far": geom_op,
        "geometry_operating_n": geom_op_n,
        "max_absolute_excess": max_excess,
        "ceiling": ceiling,
        "operating_placements": operating,
        "verdict": verdict,
        "reason": reason,
    }
    print(f"\n[far-geom] GATE {verdict}: {reason}")

    payload["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    sum_path = args.out / "far_placement_summary.json"
    sum_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    report = args.out / "far_placement_report.md"
    write_report(report, payload)
    print(f"[far-geom] wrote {sum_path}")
    print(f"[far-geom] wrote {report}")
    print(f"[far-geom] done in {payload['wall_clock_s']}s")
    return 0 if verdict != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
