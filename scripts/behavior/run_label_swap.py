#!/usr/bin/env python3
"""Real-track label-swap control — synthesis-free class discriminability.

Scores attack-run-like AIS windows (p95 SOG ≥ 30 kn, straightness ≥ 0.95)
under a different asserted benign class on both feature arms. Thresholds are
frozen; this script never retunes them.

Usage
-----
    python scripts/behavior/run_label_swap.py
    python scripts/behavior/run_label_swap.py --smoke
"""

from __future__ import annotations

import argparse
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

from fit_geometry_model import load_joined_window  # noqa: E402
from validate_benign_far import far_key, score_rows_batch  # noqa: E402

from counterusv.defense.consistency import ConsistencyScorer  # noqa: E402
from counterusv.eval.label_swap import (  # noqa: E402
    admissible_splits,
    attack_run_mask,
    load_envelope_map,
    load_label_swap_config,
    matrix_from_records,
    resolve_envelope_name,
    scoreable_eo_classes,
    swap_pairs,
    thin_n_label,
)
from counterusv.kinematics.behavior_model import (  # noqa: E402
    EnvelopeModel,
    MultiHorizonEnvelope,
)

DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "label_swap.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _load_kin_window(tmpl: str, window_s: int, *, require_complete: bool) -> pd.DataFrame:
    path = (REPO_ROOT / tmpl.format(w=window_s)).resolve()
    df = pd.read_parquet(path)
    if require_complete and "window_complete" in df.columns:
        df = df.loc[df["window_complete"]].copy()
    if "role" in df.columns:
        df = df.loc[df["role"] == "benign"].copy()
    return df


def assign_attack_run_trips(
    tables: dict[int, pd.DataFrame],
    windows: list[int],
    *,
    sog_p95_min: float,
    straightness_min: float,
    source_class: str | None = None,
    splits: set[str] | None = None,
) -> pd.DataFrame:
    """One row per trip from the longest attack-run window."""
    by_w: dict[int, pd.DataFrame] = {}
    complete: dict[int, set] = {}
    universe: set = set()
    for w in windows:
        d = tables[w]
        mask = attack_run_mask(
            d, sog_p95_min=sog_p95_min, straightness_min=straightness_min,
        )
        d = d.loc[mask].copy()
        if source_class is not None and "canonical_class" in d.columns:
            d = d.loc[d["canonical_class"].astype(str) == source_class]
        if splits is not None and "split" in d.columns:
            d = d.loc[d["split"].astype(str).isin(splits)]
        if d.empty:
            by_w[w] = d
            complete[w] = set()
            continue
        d = d.drop_duplicates("trip_id").set_index("trip_id")
        by_w[w] = d
        complete[w] = set(d.index)
        universe |= set(d.index)

    rows: list[dict[str, Any]] = []
    for trip in universe:
        cw = [w for w in windows if trip in complete[w]]
        if not cw:
            continue
        w = max(cw)
        r = by_w[w].loc[trip]
        rec = r.to_dict() if not isinstance(r, dict) else r
        if hasattr(r, "to_dict"):
            rec = r.to_dict()
        rec["trip_id"] = trip
        rec["assigned_window_s"] = int(w)
        rows.append(rec)
    return pd.DataFrame(rows)


def assign_attack_run_encounters(
    tables: dict[int, pd.DataFrame],
    windows: list[int],
    *,
    sog_p95_min: float,
    straightness_min: float,
    source_class: str | None = None,
    splits: set[str] | None = None,
    placement_classes: list[str] | None = None,
) -> pd.DataFrame:
    """One row per (trip, asset) from the longest attack-run joined window."""
    by_w: dict[int, pd.DataFrame] = {}
    complete: dict[int, set[str]] = {}
    universe: set[str] = set()
    for w in windows:
        d = tables[w]
        # Joined tables may lack role; attack_run_mask still uses sog/straightness.
        mask = attack_run_mask(
            d.assign(role=d["role"] if "role" in d.columns else "benign"),
            sog_p95_min=sog_p95_min,
            straightness_min=straightness_min,
        )
        d = d.loc[mask.to_numpy()].copy()
        if source_class is not None and "canonical_class" in d.columns:
            d = d.loc[d["canonical_class"].astype(str) == source_class]
        if splits is not None and "split" in d.columns:
            d = d.loc[d["split"].astype(str).isin(splits)]
        if placement_classes and "placement_class" in d.columns:
            d = d.loc[d["placement_class"].isin(placement_classes)]
        if d.empty or "asset_id" not in d.columns:
            by_w[w] = d
            complete[w] = set()
            continue
        d = d.copy()
        d["_ck"] = d["trip_id"].astype(str) + "||" + d["asset_id"].astype(str)
        d = d.drop_duplicates("_ck").set_index("_ck")
        by_w[w] = d
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
        rec["assigned_window_s"] = int(w)
        rows.append(rec)
    return pd.DataFrame(rows)


def _bundle_for_asserted(
    scorer: ConsistencyScorer,
    asserted_class: str,
    *,
    envelope_override: str | None = None,
) -> MultiHorizonEnvelope | EnvelopeModel | None:
    policy, env_name, _ = scorer.resolve_envelope(
        asserted_class, envelope_override=envelope_override
    )
    if policy != "score" or not env_name:
        return None
    return scorer.envelopes.get(env_name)


def score_assigned(
    scorer: ConsistencyScorer,
    assigned: pd.DataFrame,
    asserted_class: str,
    *,
    far_target: float,
    envelope_override: str | None = None,
) -> dict[str, Any]:
    """Score assigned rows under ``asserted_class`` at frozen FAR threshold."""
    empty = {
        "n_assigned": int(len(assigned)),
        "n_scored": 0,
        "n_flagged": 0,
        "flag_rate": None,
        "envelope_used": None,
    }
    if assigned.empty:
        return empty
    bundle = _bundle_for_asserted(
        scorer, asserted_class, envelope_override=envelope_override
    )
    policy, env_name, _ = scorer.resolve_envelope(
        asserted_class, envelope_override=envelope_override
    )
    if bundle is None:
        return {
            **empty,
            "envelope_used": env_name,
            "note": "unscoreable_asserted_class" if policy != "score" else "missing_envelope",
        }

    far = float(far_target)
    fk = far_key(far)
    score_rows: list[dict[str, Any]] = []
    for w, sub in assigned.groupby("assigned_window_s"):
        if isinstance(bundle, MultiHorizonEnvelope):
            env = bundle.horizons.get(int(w))
            if env is None:
                continue
        else:
            env = bundle
        sc, keep_idx, subs = score_rows_batch(env, sub, model_name="gmm")
        if len(sc) == 0:
            continue
        kept = sub.loc[list(keep_idx)]
        for i, score in enumerate(sc):
            sub_name = subs[i] or "core"
            fit = (env.subspaces.get(sub_name)
                   or env.subspaces.get("core") or {}).get("gmm")
            thr = fit.thresholds.get(fk) if fit else None
            flagged = bool(score > thr) if thr is not None else None
            if flagged is None:
                continue
            score_rows.append({
                "score": float(score),
                "flagged": flagged,
                "trip_id": kept.iloc[i].get("trip_id"),
                "asset_id": kept.iloc[i].get("asset_id"),
                "canonical_class": kept.iloc[i].get("canonical_class"),
                "split": kept.iloc[i].get("split"),
                "assigned_window_s": int(w),
            })

    if not score_rows:
        return {**empty, "envelope_used": env_name}
    detail = pd.DataFrame(score_rows)
    n_scored = int(len(detail))
    n_flagged = int(detail["flagged"].sum())
    return {
        "n_assigned": int(len(assigned)),
        "n_scored": n_scored,
        "n_flagged": n_flagged,
        "flag_rate": float(n_flagged / n_scored) if n_scored else None,
        "detail": detail,
        "envelope_used": env_name,
    }


def _load_scorer(freeze_path: Path, *, verify: bool) -> ConsistencyScorer:
    try:
        return ConsistencyScorer.from_freeze(freeze_path, verify_digests=verify)
    except ValueError as e:
        if verify and "digest" in str(e).lower():
            print(f"[label-swap] WARNING: {e}; retry without verify")
            return ConsistencyScorer.from_freeze(freeze_path, verify_digests=False)
        raise


def run_arm_kinematics(
    cfg: dict,
    emap: dict,
    *,
    smoke: bool,
    verify: bool,
    thin_n: int,
    envelope_override: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    arm = dict((cfg.get("arms") or {}).get("kinematics_only") or {})
    windows = [int(w) for w in (arm.get("windows_s") or [120, 180, 300])]
    tmpl = arm.get("features_path_template") or "data/behavior/features_window_{w}s.parquet"
    band = dict(cfg.get("attack_run") or {})
    sog_min = float(band.get("sog_p95_min") or 30.0)
    str_min = float(band.get("straightness_min") or 0.95)
    far = float(cfg.get("far_target") or 0.05)

    print("[label-swap] kinematics: loading windows …")
    tables = {
        w: _load_kin_window(tmpl, w, require_complete=True) for w in windows
    }
    # Source classes present in the band (any split).
    band_classes: set[str] = set()
    for w, d in tables.items():
        m = attack_run_mask(d, sog_p95_min=sog_min, straightness_min=str_min)
        if m.any():
            band_classes |= set(d.loc[m, "canonical_class"].astype(str).unique())
    asserted = scoreable_eo_classes(emap)
    if smoke:
        band_classes = {c for c in band_classes if c in ("recreational", "passenger_ferry")}
        asserted = [c for c in asserted if c in ("fishing", "recreational", "sailing", "small_craft")]
    sources = sorted(band_classes)
    print(f"[label-swap] kinematics sources in band: {sources}")
    if envelope_override:
        print(f"[label-swap] kinematics envelope_override={envelope_override!r}")

    freeze = REPO_ROOT / (arm.get("freeze_path") or "results/behavior_model/FROZEN.json")
    scorer = _load_scorer(freeze, verify=verify)
    if envelope_override and envelope_override not in scorer.envelopes:
        extra = Path(arm.get("freeze_path") or "results/behavior_model/FROZEN.json")
        joblib = extra.parent / "envelopes" / f"{envelope_override}.joblib"
        scorer.attach_envelope(envelope_override, joblib)

    swap_records: list[dict] = []
    matched_records: list[dict] = []
    cell_details: list[pd.DataFrame] = []

    for src in sources:
        # Matched-class control (held-out). Hygiene stays conditional.
        splits_m = admissible_splits(src, src, cfg=cfg, emap=emap)
        assigned_m = assign_attack_run_trips(
            tables, windows, sog_p95_min=sog_min, straightness_min=str_min,
            source_class=src, splits=splits_m,
        )
        if smoke and len(assigned_m) > 40:
            assigned_m = assigned_m.head(40)
        res_m = score_assigned(
            scorer, assigned_m, src, far_target=far,
            envelope_override=envelope_override,
        )
        matched_records.append({
            "arm": "kinematics_only",
            "source_class": src,
            "asserted_class": src,
            "envelope": res_m.get("envelope_used")
            or resolve_envelope_name(src, emap),
            "splits": sorted(splits_m),
            "n_scored": res_m["n_scored"],
            "n_flagged": res_m["n_flagged"],
            "flag_rate": res_m["flag_rate"],
            "thin_n": thin_n_label(res_m["n_scored"], thin_n),
            "control": "matched",
            "envelope_override": envelope_override,
        })

        for asserted_cls in asserted:
            if asserted_cls == src:
                continue
            splits = admissible_splits(src, asserted_cls, cfg=cfg, emap=emap)
            if not splits:
                continue
            assigned = assign_attack_run_trips(
                tables, windows, sog_p95_min=sog_min, straightness_min=str_min,
                source_class=src, splits=splits,
            )
            if smoke and len(assigned) > 40:
                assigned = assigned.head(40)
            res = score_assigned(
                scorer, assigned, asserted_cls, far_target=far,
                envelope_override=envelope_override,
            )
            rec = {
                "arm": "kinematics_only",
                "source_class": src,
                "asserted_class": asserted_cls,
                "envelope": res.get("envelope_used")
                or resolve_envelope_name(asserted_cls, emap),
                "splits": sorted(splits),
                "n_assigned": res["n_assigned"],
                "n_scored": res["n_scored"],
                "n_flagged": res["n_flagged"],
                "flag_rate": res["flag_rate"],
                "thin_n": thin_n_label(res["n_scored"], thin_n),
                "control": "swap",
                "envelope_override": envelope_override,
            }
            swap_records.append(rec)
            rate_s = (
                "—" if res["flag_rate"] is None else f"{res['flag_rate']:.1%}"
            )
            print(f"  kin {src}→{asserted_cls}: n={res['n_scored']} rate={rate_s}")
            detail = res.get("detail")
            if isinstance(detail, pd.DataFrame) and not detail.empty:
                detail = detail.copy()
                detail["arm"] = "kinematics_only"
                detail["source_class"] = src
                detail["asserted_class"] = asserted_cls
                cell_details.append(detail)

    meta = {
        "sources": sources,
        "asserted": asserted,
        "envelope_override": envelope_override,
    }
    all_band = assign_attack_run_trips(
        tables, windows, sog_p95_min=sog_min, straightness_min=str_min,
    )
    meta["n_band_trips"] = int(all_band["trip_id"].nunique()) if not all_band.empty else 0
    return swap_records, matched_records, meta


def run_arm_geometry(
    cfg: dict,
    emap: dict,
    *,
    smoke: bool,
    verify: bool,
    thin_n: int,
    envelope_override: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    arm = dict((cfg.get("arms") or {}).get("kinematics_geometry") or {})
    model_cfg = _load_yaml(REPO_ROOT / (arm.get("model_cfg")
                           or "configs/defense/behavior_model_geometry.yaml"))
    windows = [int(w) for w in (arm.get("windows_s") or model_cfg.get("windows_s")
                                or [180, 300, 600])]
    band = dict(cfg.get("attack_run") or {})
    sog_min = float(band.get("sog_p95_min") or 30.0)
    str_min = float(band.get("straightness_min") or 0.95)
    far = float(cfg.get("far_target") or 0.05)
    operating = list(
        cfg.get("operating_placement_classes")
        or arm.get("fit_placement_classes")
        or ["berth_approach", "anchorage"]
    )

    print("[label-swap] geometry: loading joined windows (all placements) …")
    tables_all = {
        w: load_joined_window(
            model_cfg, w, require_complete=True, placement_classes=[],
        )
        for w in windows
    }
    band_classes: set[str] = set()
    for d in tables_all.values():
        if d.empty:
            continue
        role_col = "role" if "role" in d.columns else None
        tmp = d if role_col else d.assign(role="benign")
        m = attack_run_mask(tmp, sog_p95_min=sog_min, straightness_min=str_min)
        if m.any():
            band_classes |= set(d.loc[m.to_numpy(), "canonical_class"].astype(str).unique())

    asserted = scoreable_eo_classes(emap)
    if smoke:
        band_classes = {c for c in band_classes if c in (
            "recreational", "passenger_ferry", "working_service",
        )} or band_classes
        asserted = [c for c in asserted if c in ("fishing", "recreational", "sailing")]
    sources = sorted(band_classes)
    print(f"[label-swap] geometry sources in band: {sources} "
          f"(thin-n expected under attack-run ∩ asset pairing)")
    if envelope_override:
        print(f"[label-swap] geometry envelope_override={envelope_override!r}")

    freeze = REPO_ROOT / (
        arm.get("freeze_path") or "results/behavior_model_geometry/FROZEN.json"
    )
    scorer = _load_scorer(freeze, verify=verify)
    if envelope_override and envelope_override not in scorer.envelopes:
        joblib = freeze.parent / "envelopes" / f"{envelope_override}.joblib"
        scorer.attach_envelope(envelope_override, joblib)

    swap_records: list[dict] = []
    matched_records: list[dict] = []

    def _score_geo_cells(
        placement_classes: list[str] | None,
        placement_tag: str,
    ) -> None:
        for src in sources:
            splits_m = admissible_splits(src, src, cfg=cfg, emap=emap)
            assigned_m = assign_attack_run_encounters(
                tables_all, windows, sog_p95_min=sog_min, straightness_min=str_min,
                source_class=src, splits=splits_m,
                placement_classes=placement_classes,
            )
            res_m = score_assigned(
                scorer, assigned_m, src, far_target=far,
                envelope_override=envelope_override,
            )
            matched_records.append({
                "arm": "kinematics_geometry",
                "placement_scope": placement_tag,
                "source_class": src,
                "asserted_class": src,
                "envelope": res_m.get("envelope_used")
                or resolve_envelope_name(src, emap),
                "splits": sorted(splits_m),
                "n_scored": res_m["n_scored"],
                "n_flagged": res_m["n_flagged"],
                "flag_rate": res_m["flag_rate"],
                "thin_n": thin_n_label(res_m["n_scored"], thin_n),
                "control": "matched",
                "envelope_override": envelope_override,
            })
            for asserted_cls in asserted:
                if asserted_cls == src:
                    continue
                splits = admissible_splits(src, asserted_cls, cfg=cfg, emap=emap)
                if not splits:
                    continue
                assigned = assign_attack_run_encounters(
                    tables_all, windows, sog_p95_min=sog_min, straightness_min=str_min,
                    source_class=src, splits=splits,
                    placement_classes=placement_classes,
                )
                res = score_assigned(
                    scorer, assigned, asserted_cls, far_target=far,
                    envelope_override=envelope_override,
                )
                swap_records.append({
                    "arm": "kinematics_geometry",
                    "placement_scope": placement_tag,
                    "source_class": src,
                    "asserted_class": asserted_cls,
                    "envelope": res.get("envelope_used")
                    or resolve_envelope_name(asserted_cls, emap),
                    "splits": sorted(splits),
                    "n_assigned": res["n_assigned"],
                    "n_scored": res["n_scored"],
                    "n_flagged": res["n_flagged"],
                    "flag_rate": res["flag_rate"],
                    "thin_n": thin_n_label(res["n_scored"], thin_n),
                    "control": "swap",
                    "envelope_override": envelope_override,
                })
                rate_s = (
                    "—" if res["flag_rate"] is None else f"{res['flag_rate']:.1%}"
                )
                print(
                    f"  geo[{placement_tag}] {src}→{asserted_cls}: "
                    f"n={res['n_scored']} rate={rate_s}"
                )

    _score_geo_cells(operating, "operating")
    _score_geo_cells(None, "all_placements")

    all_band = assign_attack_run_encounters(
        tables_all, windows, sog_p95_min=sog_min, straightness_min=str_min,
        placement_classes=operating,
    )
    meta = {
        "sources": sources,
        "asserted": asserted,
        "n_band_encounters_operating": int(len(all_band)),
        "n_band_trips_operating": (
            int(all_band["trip_id"].nunique()) if not all_band.empty else 0
        ),
        "operating_placements": operating,
        "envelope_override": envelope_override,
        "note": (
            "Attack-run ∩ asset-relative geometry is thin on this corpus; "
            "kinematics arm is the headline instrument."
        ),
    }
    return swap_records, matched_records, meta


def _fmt_rate(r: float | None) -> str:
    if r is None:
        return "—"
    return f"{r:.0%}"


def _matrix_md(records: list[dict], *, arm: str, placement_scope: str | None = None) -> str:
    rows = [
        r for r in records
        if r.get("arm") == arm
        and r.get("control") == "swap"
        and (placement_scope is None or r.get("placement_scope") == placement_scope)
    ]
    if not rows:
        return "_No cells._\n"
    rates, ns = matrix_from_records(rows)
    lines = ["| source \\ asserted | " + " | ".join(str(c) for c in rates.columns) + " |",
             "|---|" + "|".join(["---:"] * len(rates.columns)) + "|"]
    for src in rates.index:
        cells = []
        for col in rates.columns:
            rate = rates.loc[src, col]
            n = ns.loc[src, col]
            if pd.isna(rate) or pd.isna(n):
                cells.append("—")
            else:
                tag = "†" if int(n) < 20 else ""
                cells.append(f"{float(rate):.0%} (n={int(n)}){tag}")
        lines.append(f"| {src} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_† thin-n (n < 20)._")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    *,
    summary: dict,
    kin_swap: list[dict],
    kin_matched: list[dict],
    geo_swap: list[dict],
    geo_matched: list[dict],
) -> None:
    ts = summary.get("timestamp", "")
    lines = [
        "# Real-track label-swap control",
        "",
        f"_Generated {ts}"
        + (" (smoke)" if summary.get("smoke") else "")
        + "._",
        "",
        "## Scope",
        "",
        "Synthesis-free discriminability: real attack-run-like AIS windows",
        f"(p95 SOG ≥ {summary['attack_run']['sog_p95_min']}, "
        f"straightness ≥ {summary['attack_run']['straightness_min']}) "
        "scored under a **different** asserted benign class. Thresholds are",
        "frozen FAR@5%; this report does not retune them and does not claim",
        "adversary-motion DDR.",
        "",
        "## Split hygiene",
        "",
        "- Asserted envelope members exclude source class → train+val+test admissible.",
        "- `small_craft` / Class-B proxy pairings → val+test only.",
        "- Matched-class control → val+test only.",
        "",
        "## Kinematics arm (headline)",
        "",
        f"Band trips (any split): **{summary['kinematics']['n_band_trips']}**.",
        "",
        "### Off-diagonal swap matrix (flag rate @ FAR@5%)",
        "",
        _matrix_md(kin_swap, arm="kinematics_only"),
        "",
        "### Matched-class control (same windows, true class)",
        "",
        "| source | n scored | flag rate | thin-n |",
        "|---|---:|---:|:---:|",
    ]
    for r in sorted(kin_matched, key=lambda x: x["source_class"]):
        lines.append(
            f"| {r['source_class']} | {r['n_scored']} | "
            f"{_fmt_rate(r['flag_rate'])} | {r.get('thin_n') or ''} |"
        )
    lines += [
        "",
        "## Geometry arm (thin-n under attack-run ∩ asset pairing)",
        "",
        f"Operating (berth+anchorage) band encounters: "
        f"**{summary['geometry']['n_band_encounters_operating']}** "
        f"({summary['geometry']['n_band_trips_operating']} trips). "
        f"{summary['geometry'].get('note', '')}",
        "",
        "### Operating placements — swap matrix",
        "",
        _matrix_md(geo_swap, arm="kinematics_geometry", placement_scope="operating"),
        "",
        "### All placements — swap matrix (sensitivity)",
        "",
        _matrix_md(geo_swap, arm="kinematics_geometry", placement_scope="all_placements"),
        "",
        "### Matched-class control (operating)",
        "",
        "| source | n scored | flag rate | thin-n |",
        "|---|---:|---:|:---:|",
    ]
    for r in sorted(
        [x for x in geo_matched if x.get("placement_scope") == "operating"],
        key=lambda x: x["source_class"],
    ):
        lines.append(
            f"| {r['source_class']} | {r['n_scored']} | "
            f"{_fmt_rate(r['flag_rate'])} | {r.get('thin_n') or ''} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- High off-diagonal flag rate with low matched-class rate ⇒ class channel",
        "  buys discriminability in the attack-run band.",
        "- High matched-class rate ⇒ the band is already harsh under the true label",
        "  (instrument still useful for swaps, but not a free lunch).",
        "- Geometry cells with n ≪ 20 are documentary; kinematics carries the claim.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-verify-digests", action="store_true")
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument(
        "--envelope-override",
        type=str,
        default=None,
        help="scoreable classes use this envelope (e.g. pooled_benign); "
             "hygiene / admissible splits stay class-conditional",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: config output, or "
             "results/label_swap_pooled when --envelope-override is set)",
    )
    args = ap.parse_args()

    cfg = load_label_swap_config(args.config)
    emap = load_envelope_map(cfg.get("envelope_map"))
    thin_n = int(cfg.get("thin_n_threshold") or 20)
    verify = not args.no_verify_digests
    override = args.envelope_override
    if override is None and str(cfg.get("envelope_routing") or "") == "pooled":
        override = str(cfg.get("envelope_override") or "pooled_benign")
    elif override is None and cfg.get("envelope_override"):
        override = str(cfg["envelope_override"])
    t0 = time.perf_counter()

    kin_swap, kin_matched, kin_meta = run_arm_kinematics(
        cfg, emap, smoke=args.smoke, verify=verify, thin_n=thin_n,
        envelope_override=override,
    )
    if args.skip_geometry:
        geo_swap, geo_matched, geo_meta = [], [], {
            "sources": [], "asserted": [],
            "n_band_encounters_operating": 0,
            "n_band_trips_operating": 0,
            "note": "skipped",
        }
    else:
        geo_swap, geo_matched, geo_meta = run_arm_geometry(
            cfg, emap, smoke=args.smoke, verify=verify, thin_n=thin_n,
            envelope_override=override,
        )

    out = dict(cfg.get("output") or {})
    if args.out_dir is not None:
        out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    elif override:
        out_dir = REPO_ROOT / "results" / "label_swap_pooled"
    else:
        out_dir = REPO_ROOT / Path(out.get("dir") or "results/label_swap")
    report_path = out_dir / "label_swap_report.md"
    summary_path = out_dir / "label_swap_summary.json"
    cells_path = out_dir / "label_swap_cells.parquet"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "config": str(Path(args.config)),
        "far_target": float(cfg.get("far_target") or 0.05),
        "attack_run": dict(cfg.get("attack_run") or {}),
        "envelope_override": override,
        "kinematics": kin_meta,
        "geometry": geo_meta,
        "swap_cells": kin_swap + geo_swap,
        "matched_controls": kin_matched + geo_matched,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "scope": (
            "Synthesis-free label-swap discriminability only. "
            "No adversary-motion DDR / cost curves."
            + (
                f" Envelope override={override!r}; hygiene stays conditional."
                if override else ""
            )
        ),
    }

    write_report(
        report_path,
        summary=summary,
        kin_swap=kin_swap,
        kin_matched=kin_matched,
        geo_swap=geo_swap,
        geo_matched=geo_matched,
    )
    cells = pd.DataFrame(kin_swap + geo_swap + kin_matched + geo_matched)
    cells_path.parent.mkdir(parents=True, exist_ok=True)
    if not cells.empty:
        # splits lists → string for parquet
        cells = cells.copy()
        if "splits" in cells.columns:
            cells["splits"] = cells["splits"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else x
            )
        cells.to_parquet(cells_path, index=False)

    # JSON-safe summary (drop huge nested detail if any)
    safe = dict(summary)
    summary_path.write_text(json.dumps(safe, indent=2, default=str) + "\n")
    print(f"[label-swap] wrote {report_path.relative_to(REPO_ROOT)}")
    print(f"[label-swap] wrote {summary_path.relative_to(REPO_ROOT)}")
    if not cells.empty:
        print(f"[label-swap] wrote {cells_path.relative_to(REPO_ROOT)}")
    print(f"[label-swap] done in {summary['wall_clock_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
