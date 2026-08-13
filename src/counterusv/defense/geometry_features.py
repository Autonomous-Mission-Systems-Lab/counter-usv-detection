"""Asset-relative encounter geometry features (scorer contract v2).

Pure function of (windowed points, asset lat/lon). Callers pass the same
last-*W* point slice the kinematics path already isolates — this module does
not invent a second windowing rule.

Abstain (return ``None``) when no point falls in the engagement annulus or
inbound-leg gates fail. Never impute zeros for missing geometry.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from counterusv.kinematics.features import ang_diff_deg, haversine_km

NM_PER_KM = 1.0 / 1.852
GEOMETRY_FEATURE_KEYS = (
    "range_min_nm",
    "closing_rate_med_kn",
    "closing_rate_p90_kn",
    "bearing_rate_std_dps",
    "dcpa_nm",
    "tcpa_s",
    "closing_frac",
    "inbound_leg_persistence_s",
    "n_points_in_annulus",
    "cpa_range_nm",
    "geometry_usable",
)


def range_bearing_nm(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    asset_lat: float,
    asset_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Range (nm) and bearing (deg CW from N) from asset to each contact point."""
    lat_a = np.asarray(lat, dtype="float64")
    lon_a = np.asarray(lon, dtype="float64")
    range_nm = haversine_km(asset_lat, asset_lon, lat_a, lon_a) * NM_PER_KM

    alat = np.radians(asset_lat)
    plat = np.radians(lat_a)
    dlon = np.radians(lon_a - asset_lon)
    x = np.sin(dlon) * np.cos(plat)
    y = np.cos(alat) * np.sin(plat) - np.sin(alat) * np.cos(plat) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return range_nm, bearing


def _default_annulus(annulus: Mapping[str, Any] | None) -> dict[str, float]:
    a = dict(annulus or {})
    return {
        "min_range_nm": float(a.get("min_range_nm", 0.25)),
        "max_range_nm": float(a.get("max_range_nm", 6.0)),
    }


def _default_inbound(inbound_leg: Mapping[str, Any] | None) -> dict[str, int]:
    leg = dict(inbound_leg or {})
    return {
        "require_points_before_cpa": int(leg.get("require_points_before_cpa", 3)),
        "min_points": int(leg.get("min_points", 4)),
    }


def geometry_features_from_points(
    points: pd.DataFrame,
    asset_lat: float,
    asset_lon: float,
    *,
    annulus: Mapping[str, Any] | None = None,
    inbound_leg: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute asset-relative features for one windowed point stream.

    Closing rate = −Δrange / Δt expressed in knots (nm/h); positive means
    approaching. CPA is the minimum range among points inside the annulus
    (``cpa_definition: min_range_in_annulus``). The inbound leg is the
    contiguous sequence of in-annulus points ending at that CPA index.

    DCPA / TCPA use the great-circle motion of the last segment before CPA
    (or the last two in-annulus points if CPA is the first usable sample):
    constant-velocity extrapolation of range to the closest-point-of-approach
    of that segment's ray. At ~60 s AIS cadence these estimates are coarse;
    callers should report coverage rather than densify.

    Returns
    -------
    dict | None
        Feature block with ``geometry_usable=True``, or ``None`` to abstain.
    """
    required = {"t", "lat", "lon"}
    missing = required - set(points.columns)
    if missing:
        raise KeyError(f"points missing columns: {sorted(missing)}")
    if points.empty:
        return None

    ann = _default_annulus(annulus)
    leg = _default_inbound(inbound_leg)
    min_r = ann["min_range_nm"]
    max_r = ann["max_range_nm"]

    pts = points.sort_values("t", kind="mergesort").reset_index(drop=True)
    t = pts["t"].to_numpy(dtype="float64")
    lat = pts["lat"].to_numpy(dtype="float64")
    lon = pts["lon"].to_numpy(dtype="float64")
    range_nm, bearing = range_bearing_nm(lat, lon, float(asset_lat), float(asset_lon))

    # Scoreable band: r <= max (terminal zone inside min still computes).
    in_max = range_nm <= max_r
    if not np.any(in_max):
        return None

    # Annulus for CPA / inbound-leg construction.
    in_annulus = (range_nm >= min_r) & (range_nm <= max_r)
    if not np.any(in_annulus):
        # All points inside terminal band — still scoreable; use in_max as the
        # working set so pier-side contacts are not forced to abstain.
        in_annulus = in_max

    idx_ann = np.flatnonzero(in_annulus)
    n_ann = int(idx_ann.size)
    if n_ann < leg["min_points"]:
        return None

    # CPA among in-annulus points.
    cpa_local = int(np.argmin(range_nm[idx_ann]))
    cpa_idx = int(idx_ann[cpa_local])
    cpa_range = float(range_nm[cpa_idx])
    points_before = cpa_local  # count of in-annulus points strictly before CPA
    if points_before < leg["require_points_before_cpa"]:
        return None

    # Contiguous inbound leg ending at CPA within the in-annulus index list.
    # Walk backward while indices are consecutive in time order.
    start_local = cpa_local
    while start_local > 0 and idx_ann[start_local - 1] == idx_ann[start_local] - 1:
        start_local -= 1
    leg_idx = idx_ann[start_local : cpa_local + 1]
    if leg_idx.size < leg["min_points"]:
        return None

    leg_t = t[leg_idx]
    leg_r = range_nm[leg_idx]
    leg_b = bearing[leg_idx]
    dt = np.diff(leg_t)
    valid_dt = dt > 0
    if not np.any(valid_dt):
        return None

    # Closing rate (kn): −Δr / Δt * 3600
    dr = np.diff(leg_r)
    closing = np.full(dr.shape, np.nan)
    closing[valid_dt] = -dr[valid_dt] / dt[valid_dt] * 3600.0
    closing_ok = closing[np.isfinite(closing)]
    if closing_ok.size == 0:
        return None

    # Bearing-rate stability: std of (Δbearing_unwrapped / Δt) in deg/s.
    db = ang_diff_deg(leg_b[1:], leg_b[:-1])
    brate = np.full(db.shape, np.nan)
    brate[valid_dt] = db[valid_dt] / dt[valid_dt]
    brate_ok = brate[np.isfinite(brate)]
    bearing_rate_std = float(np.std(brate_ok)) if brate_ok.size else float("nan")

    closing_frac = float(np.mean(closing_ok > 0.0))
    inbound_persistence = float(leg_t[-1] - leg_t[0])

    # DCPA / TCPA from last segment before CPA (constant-velocity ray).
    dcpa_nm, tcpa_s = _dcpa_tcpa_last_segment(
        lat, lon, t, cpa_idx, float(asset_lat), float(asset_lon),
    )

    return {
        "range_min_nm": float(np.min(range_nm[in_annulus])),
        "closing_rate_med_kn": float(np.median(closing_ok)),
        "closing_rate_p90_kn": float(np.quantile(closing_ok, 0.90)),
        "bearing_rate_std_dps": bearing_rate_std,
        "dcpa_nm": dcpa_nm,
        "tcpa_s": tcpa_s,
        "closing_frac": closing_frac,
        "inbound_leg_persistence_s": inbound_persistence,
        "n_points_in_annulus": n_ann,
        "cpa_range_nm": cpa_range,
        "geometry_usable": True,
    }


def _dcpa_tcpa_last_segment(
    lat: np.ndarray,
    lon: np.ndarray,
    t: np.ndarray,
    cpa_idx: int,
    asset_lat: float,
    asset_lon: float,
) -> tuple[float, float]:
    """Closest approach of the last pre-CPA segment, extrapolated at const vel.

    Uses a local ENU (nm) frame centered on the asset. Returns (dcpa_nm, tcpa_s)
    relative to the segment start; NaN if the segment is unusable.
    """
    if cpa_idx < 1:
        return float("nan"), float("nan")
    i0, i1 = cpa_idx - 1, cpa_idx
    dt = float(t[i1] - t[i0])
    if dt <= 0:
        return float("nan"), float("nan")

    # East/north nm relative to asset (small-angle ENU).
    def enu(la: float, lo: float) -> tuple[float, float]:
        north = (la - asset_lat) * 60.0  # 1 deg lat ≈ 60 nm
        east = (lo - asset_lon) * 60.0 * np.cos(np.radians(asset_lat))
        return east, north

    e0, n0 = enu(float(lat[i0]), float(lon[i0]))
    e1, n1 = enu(float(lat[i1]), float(lon[i1]))
    ve, vn = (e1 - e0) / dt, (n1 - n0) / dt
    speed2 = ve * ve + vn * vn
    if speed2 <= 0:
        return float(np.hypot(e0, n0)), float("nan")

    # Time of closest approach of ray p(t) = p0 + v*t to origin.
    tcpa = -(e0 * ve + n0 * vn) / speed2
    east_c = e0 + ve * tcpa
    north_c = n0 + vn * tcpa
    dcpa = float(np.hypot(east_c, north_c))
    return dcpa, float(tcpa)
