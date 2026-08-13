"""Oracle DDR + defensibility-gap helpers (perfect-disguise condition).

Scores frozen adversary-motion tracks under asserted = mimicked class.
Aggregation / gap / NC sanity live here; orchestration is in
``scripts/defense/run_oracle_ddr.py``. Never writes scores into the motion
freeze.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from counterusv.eval.label_swap import thin_n_label
from counterusv.defense.geometry_features import geometry_features_from_points
from counterusv.kinematics.features import features_from_points, last_window_mask

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "oracle_ddr.yaml"

_DDR_KEYS = frozenset({
    "ddr",
    "detection_rate",
    "cost_curve",
    "ddr_by_cell",
    "adaptive_cost_curve",
})


def load_oracle_ddr_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CFG
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"oracle_ddr config not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def resolve_path(rel: str | Path, *, root: Path | None = None) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (root or REPO_ROOT) / p


def operating_far(summary: Mapping[str, Any], *, geometry: bool) -> float | None:
    """Read measured FAR@5% from a behavior-model FAR summary JSON."""
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


def load_operating_far(
    far_summary_path: Path | str,
    *,
    geometry: bool,
    default: float,
) -> float:
    p = resolve_path(far_summary_path)
    if not p.is_file():
        return float(default)
    summary = json.loads(p.read_text())
    far = operating_far(summary, geometry=geometry)
    return float(far if far is not None else default)


def extract_windows(
    points: pd.DataFrame,
    windows_s: Sequence[int],
    *,
    asset_lat: float | None = None,
    asset_lon: float | None = None,
    annulus: Mapping[str, Any] | None = None,
    inbound_leg: Mapping[str, Any] | None = None,
    join_geometry: bool = False,
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """Extract last-W kinematics (+ optional geometry) features per window."""
    by_w: dict[int, dict[str, Any]] = {}
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
                win,
                float(asset_lat),
                float(asset_lon),
                annulus=dict(annulus) if annulus else None,
                inbound_leg=dict(inbound_leg) if inbound_leg else None,
            )
            if geo is None:
                continue
            row.update(geo)
        by_w[int(w)] = row
        complete.add(int(w))
    return by_w, complete


def has_ddr_payload(obj: object) -> bool:
    """True only if numeric DDR / cost-curve fields are present (not prose)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in _DDR_KEYS:
                return True
            if has_ddr_payload(v):
                return True
    elif isinstance(obj, list):
        return any(has_ddr_payload(x) for x in obj)
    return False


def assert_freeze_has_no_ddr(path: Path | str) -> dict[str, Any]:
    """Load a motion freeze and refuse if it already carries DDR numbers."""
    p = resolve_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"motion freeze not found: {p}")
    freeze = json.loads(p.read_text())
    if has_ddr_payload(freeze):
        raise ValueError(
            f"anti-circularity: {p} already contains DDR / cost-curve fields; "
            "refuse to proceed (scores belong in results/oracle_ddr/ only)"
        )
    return freeze


def rate_from_actions(actions: Iterable[str]) -> dict[str, Any]:
    """DDR = flagged / scored (excluding abstain)."""
    acts = [str(a) for a in actions]
    scored = [a for a in acts if a in ("flag", "pass")]
    n = len(scored)
    n_flag = sum(1 for a in scored if a == "flag")
    n_abstain = sum(1 for a in acts if a == "abstain")
    return {
        "n": n,
        "n_flag": n_flag,
        "n_abstain": n_abstain,
        "n_total": len(acts),
        "ddr": (n_flag / n) if n else None,
    }


def ddr_table(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("mimicked_class", "arm", "defense_kind"),
    exclude_nc: bool = True,
    thin_n_threshold: int = 20,
) -> pd.DataFrame:
    """Aggregate flag rates by group; optionally drop negative-control cells."""
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty:
        return pd.DataFrame(
            columns=[*group_cols, "n", "n_flag", "n_abstain", "ddr", "thin_n"]
        )
    work = df.copy()
    if exclude_nc and "negative_control" in work.columns:
        work = work.loc[~work["negative_control"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for keys, g in work.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = rate_from_actions(g["action"].tolist())
        row = {c: keys[i] for i, c in enumerate(group_cols)}
        row.update(stats)
        row["thin_n"] = thin_n_label(stats["n"], thin_n_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def defensibility_gap(
    consistency_ddr: float | None,
    presence_ddr: float | None,
) -> float | None:
    """Consistency DDR minus presence DDR (disguise → presence ≈ 0)."""
    if consistency_ddr is None or presence_ddr is None:
        return None
    return float(consistency_ddr) - float(presence_ddr)


def gap_table(
    consistency: pd.DataFrame,
    presence: pd.DataFrame,
    *,
    on: Sequence[str] = ("mimicked_class", "arm"),
) -> pd.DataFrame:
    """Join consistency and presence DDR tables and attach the gap."""
    if consistency.empty:
        return pd.DataFrame()
    cons = consistency.copy()
    pres = presence.copy()
    if "defense_kind" in cons.columns:
        cons = cons.loc[cons["defense_kind"] == "consistency"].copy()
    if "defense_kind" in pres.columns:
        pres = pres.loc[pres["defense_kind"] == "presence_only"].copy()
    # Presence has no arm — broadcast by mimicked_class onto each arm row.
    p_cols = ["mimicked_class", "ddr", "n"]
    p_cols = [c for c in p_cols if c in pres.columns]
    p_small = pres[p_cols].rename(
        columns={"ddr": "presence_ddr", "n": "presence_n"}
    )
    if "mimicked_class" in cons.columns and "mimicked_class" in p_small.columns:
        out = cons.merge(p_small, on="mimicked_class", how="left")
    else:
        out = cons.copy()
        out["presence_ddr"] = None
        out["presence_n"] = None
    out["gap"] = [
        defensibility_gap(
            None if pd.isna(cd) else float(cd),
            None if pd.isna(pd_) else float(pd_),
        )
        for cd, pd_ in zip(
            out["ddr"] if "ddr" in out.columns else [],
            out["presence_ddr"] if "presence_ddr" in out.columns else [],
        )
    ]
    # Preserve requested join keys for callers.
    _ = on
    return out


def nc_sanity(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    far: float,
    arm: str,
    factor: float = 2.0,
    slack: float = 0.05,
) -> dict[str, Any]:
    """Negative-control flag rate vs measured FAR (matched asserted class)."""
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    if df.empty:
        return {
            "arm": arm,
            "n": 0,
            "n_flag": 0,
            "ddr": None,
            "operating_far": float(far),
            "ceiling": float(far) * float(factor) + float(slack),
            "within_tol": True,
            "note": "no NC records",
        }
    work = df.copy()
    if "negative_control" in work.columns:
        work = work.loc[work["negative_control"].astype(bool)].copy()
    if "arm" in work.columns:
        work = work.loc[work["arm"].astype(str) == str(arm)].copy()
    if "defense_kind" in work.columns:
        work = work.loc[work["defense_kind"].astype(str) == "consistency"].copy()
    stats = rate_from_actions(work["action"].tolist() if not work.empty else [])
    ceiling = float(far) * float(factor) + float(slack)
    ddr = stats["ddr"]
    within = True if ddr is None else (float(ddr) <= ceiling)
    return {
        "arm": arm,
        **stats,
        "operating_far": float(far),
        "ceiling": ceiling,
        "within_tol": within,
        "tolerance": {"factor": float(factor), "slack": float(slack)},
    }


def cell_kind(row: Mapping[str, Any]) -> str:
    if bool(row.get("negative_control")):
        return "nc"
    if bool(row.get("unconstrained")):
        return "unc"
    return "swp"


def pivot_ddr_markdown(
    table: pd.DataFrame,
    *,
    arms: Sequence[str],
    classes: Sequence[str] | None = None,
) -> str:
    """Render a mimicked_class × arm DDR markdown table."""
    if table.empty:
        return "_No scored cells._\n"
    work = table.copy()
    if classes is None:
        classes = sorted(work["mimicked_class"].astype(str).unique().tolist())
    lines = [
        "| mimicked_class | " + " | ".join(arms) + " |",
        "|---| " + " | ".join(["---"] * len(arms)) + " |",
    ]
    for cls in classes:
        cells: list[str] = []
        for arm in arms:
            sub = work.loc[
                (work["mimicked_class"].astype(str) == cls)
                & (work["arm"].astype(str) == arm)
            ]
            if sub.empty or sub.iloc[0]["ddr"] is None:
                cells.append("—")
                continue
            r = sub.iloc[0]
            pct = 100.0 * float(r["ddr"])
            thin = f" {r['thin_n']}" if r.get("thin_n") else ""
            cells.append(f"**{pct:.1f}%** (n={int(r['n'])}){thin}")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
