"""Adaptive-adversary cost-curve helpers (RQ3).

Join terminal oracle DDR to the frozen motion sweep; derive added approach
time and companion axes; optionally walk tracks causally for first-flag
warning time / standoff. Never writes into the motion freeze.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from counterusv.attacks.kinematics import range_nm as haversine_range_nm
from counterusv.eval.label_swap import thin_n_label
from counterusv.eval.oracle_ddr import (
    assert_freeze_has_no_ddr,
    extract_windows,
    rate_from_actions,
    resolve_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "configs" / "defense" / "adaptive_cost.yaml"

CONSISTENCY_ARMS = ("kinematics_only", "kinematics_geometry")


def load_adaptive_cost_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CFG
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"adaptive_cost config not found: {p}")
    return yaml.safe_load(p.read_text()) or {}


def platform_burst_kn(
    platforms: Mapping[str, Any],
    platform: str,
    *,
    default: float = 42.0,
) -> float:
    block = platforms.get(platform) or {}
    if isinstance(block, Mapping) and "burst_kn" in block:
        return float(block["burst_kn"])
    return float(default)


def added_approach_time_h(
    *,
    v_mimic_kn: float,
    v_max_kn: float,
    commit_range_nm: float,
    start_range_nm: float = 12.0,
    unconstrained: bool = False,
) -> float:
    """Δt_add = R (1/v_mimic − 1/v_max) in hours; unconstrained → 0."""
    if unconstrained:
        return 0.0
    v_m = float(v_mimic_kn)
    v_x = float(v_max_kn)
    if v_m <= 0 or v_x <= 0:
        raise ValueError(f"speeds must be positive (got v_mimic={v_m}, v_max={v_x})")
    if v_m >= v_x - 1e-12:
        return 0.0
    r = max(0.0, float(start_range_nm) - float(commit_range_nm))
    return float(r * (1.0 / v_m - 1.0 / v_x))


def added_approach_time_min(**kwargs: Any) -> float:
    return added_approach_time_h(**kwargs) * 60.0


def join_ddr_sweep(
    ddr_cells: pd.DataFrame,
    sweep_cells: pd.DataFrame,
    platforms: Mapping[str, Any],
    *,
    start_range_nm: float = 12.0,
    arms: Sequence[str] = CONSISTENCY_ARMS,
    exclude_nc: bool = True,
) -> pd.DataFrame:
    """Join consistency DDR rows to sweep knobs; attach Δt_add."""
    cons = ddr_cells.loc[
        ddr_cells["arm"].astype(str).isin(list(arms))
        & (ddr_cells["defense_kind"].astype(str) == "consistency")
    ].copy()
    if exclude_nc and "negative_control" in cons.columns:
        cons = cons.loc[~cons["negative_control"].astype(bool)].copy()

    sweep = sweep_cells.copy()
    sweep["cell_id"] = sweep["cell_id"].astype(str)
    cons["cell_id"] = cons["cell_id"].astype(str)

    keep_sweep = [
        c
        for c in (
            "cell_id",
            "v_mimic_kn",
            "commit_range_nm",
            "bearing_offset_deg",
            "peak_speed_kn",
            "use_burst_on_commit",
            "in_plausibility_band",
        )
        if c in sweep.columns
    ]
    # Prefer sweep flags when present; keep DDR copies for NC/unc if join drops.
    joined = cons.merge(sweep[keep_sweep], on="cell_id", how="inner", suffixes=("", "_sw"))
    if joined.empty:
        raise ValueError("join_ddr_sweep produced 0 rows — cell_id mismatch?")

    v_max = [
        platform_burst_kn(platforms, str(p)) for p in joined["platform"].astype(str)
    ]
    joined["v_max_kn"] = v_max
    joined["r_mimic_nm"] = (
        float(start_range_nm) - joined["commit_range_nm"].astype(float)
    ).clip(lower=0.0)
    joined["delta_t_add_h"] = [
        added_approach_time_h(
            v_mimic_kn=float(r.v_mimic_kn),
            v_max_kn=float(r.v_max_kn),
            commit_range_nm=float(r.commit_range_nm),
            start_range_nm=float(start_range_nm),
            unconstrained=bool(r.unconstrained),
        )
        for r in joined.itertuples(index=False)
    ]
    joined["delta_t_add_min"] = joined["delta_t_add_h"] * 60.0
    joined["flagged"] = joined["action"].astype(str) == "flag"
    return joined.reset_index(drop=True)


def ddr_by_axis(
    df: pd.DataFrame,
    axis: str,
    *,
    group_cols: Sequence[str] = ("mimicked_class", "arm"),
    thin_n_threshold: int = 20,
) -> pd.DataFrame:
    """Aggregate DDR by cost / knob axis × group columns."""
    if df.empty:
        return pd.DataFrame(
            columns=[*group_cols, axis, "n", "n_flag", "n_abstain", "ddr", "thin_n"]
        )
    if axis not in df.columns:
        raise KeyError(f"axis {axis!r} not in dataframe")
    cols = [*group_cols, axis]
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(list(cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        stats = rate_from_actions(g["action"].tolist())
        row = {c: keys[i] for i, c in enumerate(cols)}
        row.update(stats)
        row["thin_n"] = thin_n_label(stats["n"], thin_n_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def range_series(
    points: pd.DataFrame,
    asset_lat: float,
    asset_lon: float,
) -> pd.Series:
    """Haversine range-to-asset (nm) aligned to ``points`` index."""
    lat = points["lat"].to_numpy(dtype=float)
    lon = points["lon"].to_numpy(dtype=float)
    r = np.array(
        [
            haversine_range_nm(float(la), float(lo), float(asset_lat), float(asset_lon))
            for la, lo in zip(lat, lon)
        ],
        dtype=float,
    )
    return pd.Series(r, index=points.index, name="r_nm")


def annulus_entry_time(
    points: pd.DataFrame,
    r_nm: pd.Series,
    *,
    max_range_nm: float,
) -> float | None:
    """Unix time when range first drops to ≤ max_range_nm."""
    work = points.copy()
    work["_r"] = r_nm.reindex(work.index).to_numpy()
    work = work.sort_values("t")
    hit = work.loc[work["_r"] <= float(max_range_nm)]
    if hit.empty:
        return None
    return float(hit["t"].iloc[0])


def checkpoint_times(
    points: pd.DataFrame,
    *,
    min_history_s: float,
    stride_s: float,
) -> list[float]:
    """Sparse end-times for causal scoring once enough history exists."""
    if points.empty:
        return []
    t = np.sort(points["t"].to_numpy(dtype=float))
    t0 = float(t[0])
    t1 = float(t[-1])
    first = t0 + float(min_history_s)
    if first > t1 + 1e-9:
        return []
    stride = max(1.0, float(stride_s))
    out: list[float] = []
    tk = first
    while tk <= t1 + 1e-9:
        out.append(float(tk))
        tk += stride
    if not out or abs(out[-1] - t1) > 1e-6:
        out.append(t1)
    # Deduplicate while preserving order.
    seen: set[float] = set()
    uniq: list[float] = []
    for x in out:
        key = round(x, 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)
    return uniq


def first_flag_along_track(
    points: pd.DataFrame,
    *,
    asset_lat: float,
    asset_lon: float,
    max_range_nm: float,
    windows_s: Sequence[int],
    join_geometry: bool,
    annulus: Mapping[str, Any] | None,
    inbound_leg: Mapping[str, Any] | None,
    evaluate_fn: Callable[..., Any],
    stride_s: float = 60.0,
) -> dict[str, Any]:
    """Walk the track forward; return first-flag time / range (or nulls).

    ``evaluate_fn`` must accept ``features_by_window`` / ``complete_windows``
    kwargs and return an object with ``.action`` (``DefenseDecision``).
    """
    g = points.sort_values("t").reset_index(drop=True)
    if len(g) < 3:
        return {
            "flagged": False,
            "t_flag_unix": None,
            "t_flag_s": None,
            "R_flag_nm": None,
            "t_enter_unix": None,
            "n_checkpoints": 0,
        }

    r = range_series(g, asset_lat, asset_lon)
    t_enter = annulus_entry_time(g, r, max_range_nm=max_range_nm)
    min_hist = float(max(int(w) for w in windows_s)) if windows_s else 300.0
    times = checkpoint_times(g, min_history_s=min_hist, stride_s=stride_s)

    for tk in times:
        prefix = g.loc[g["t"] <= tk + 1e-9]
        if len(prefix) < 3:
            continue
        by_w, complete = extract_windows(
            prefix,
            windows_s,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            annulus=annulus,
            inbound_leg=inbound_leg,
            join_geometry=join_geometry,
        )
        if not by_w:
            continue
        dec = evaluate_fn(
            features_by_window=by_w,
            complete_windows=complete,
        )
        if str(dec.action) != "flag":
            continue
        # Range at last point of prefix.
        r_flag = float(r.loc[prefix.index].iloc[-1])
        t_flag_s = None
        if t_enter is not None:
            t_flag_s = float(tk) - float(t_enter)
        return {
            "flagged": True,
            "t_flag_unix": float(tk),
            "t_flag_s": t_flag_s,
            "R_flag_nm": r_flag,
            "t_enter_unix": t_enter,
            "n_checkpoints": len(times),
        }

    return {
        "flagged": False,
        "t_flag_unix": None,
        "t_flag_s": None,
        "R_flag_nm": None,
        "t_enter_unix": t_enter,
        "n_checkpoints": len(times),
    }


def warning_summary(
    warning_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("mimicked_class", "arm"),
    thin_n_threshold: int = 20,
) -> pd.DataFrame:
    """Summarize standoff / warning time among causally flagged cells."""
    if warning_df.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "n",
                "n_flagged",
                "flag_rate",
                "R_flag_nm_median",
                "t_flag_min_median",
                "thin_n",
            ]
        )
    rows: list[dict[str, Any]] = []
    for keys, g in warning_df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        flagged = g.loc[g["flagged"].astype(bool)]
        n = len(g)
        n_f = len(flagged)
        row = {c: keys[i] for i, c in enumerate(group_cols)}
        row["n"] = n
        row["n_flagged"] = n_f
        row["flag_rate"] = (n_f / n) if n else None
        if n_f:
            row["R_flag_nm_median"] = float(flagged["R_flag_nm"].median())
            t_s = flagged["t_flag_s"].dropna()
            row["t_flag_min_median"] = (
                float(t_s.median()) / 60.0 if len(t_s) else None
            )
        else:
            row["R_flag_nm_median"] = None
            row["t_flag_min_median"] = None
        row["thin_n"] = thin_n_label(n, thin_n_threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_curve_markdown(
    curves: pd.DataFrame,
    *,
    axis: str,
    arms: Sequence[str],
    classes: Sequence[str],
) -> str:
    """Compact markdown table: class × arm DDR at each axis value."""
    if curves.empty or axis not in curves.columns:
        return "_no curve rows_\n"
    lines = [
        f"| {axis} | "
        + " | ".join(f"{c}/{a}" for c in classes for a in arms)
        + " |"
    ]
    lines.append("|---|" + "|".join(["---:" for _ in classes for _ in arms]) + "|")
    vals = sorted(curves[axis].dropna().unique().tolist())
    for v in vals:
        cells: list[str] = []
        for c in classes:
            for a in arms:
                hit = curves.loc[
                    (curves[axis] == v)
                    & (curves["mimicked_class"].astype(str) == str(c))
                    & (curves["arm"].astype(str) == str(a))
                ]
                if hit.empty or hit.iloc[0]["ddr"] is None:
                    cells.append("—")
                else:
                    ddr = float(hit.iloc[0]["ddr"])
                    n = int(hit.iloc[0]["n"])
                    cells.append(f"{100.0 * ddr:.0f}% (n={n})")
        # Format axis value
        if isinstance(v, float):
            v_s = f"{v:g}"
        else:
            v_s = str(v)
        lines.append(f"| {v_s} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def load_freeze_platforms(freeze_path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (freeze_dict, platforms) after anti-circularity check."""
    freeze = assert_freeze_has_no_ddr(freeze_path)
    platforms = dict(freeze.get("platforms") or {})
    return freeze, platforms
