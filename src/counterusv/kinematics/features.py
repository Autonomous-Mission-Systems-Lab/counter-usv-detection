"""Kinematic feature extraction (AIS / windowed).

Mirrors the definitions in ``scripts/data/ingest_ais.py`` ``compute_features`` so
whole-track and observation-window features are comparable. Window policy
(one last-W-seconds slice per track per length) is locked in
``configs/defense/scorer_features.yaml``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088
LOITER_SOG_KN = 0.5


def haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ang_diff_deg(a, b):
    """Smallest signed angular difference a−b in degrees, wrapped to [−180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def features_from_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the frozen scorer feature set, one row per ``trip_id``.

    Expects columns: ``trip_id``, ``t``, ``lat``, ``lon``, ``sog``, ``cog``,
    ``heading`` (optional), and optionally ``mmsi``. Points must be sorted by
    ``(trip_id, t)``. Definitions match ``scripts/data/ingest_ais.py``.
    """
    required = {"trip_id", "t", "lat", "lon", "sog", "cog"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"points missing columns: {sorted(missing)}")
    if df.empty:
        return pd.DataFrame()

    n = len(df)
    trip = df["trip_id"].to_numpy()
    t = df["t"].to_numpy(dtype="float64")
    lat = df["lat"].to_numpy(dtype="float64")
    lon = df["lon"].to_numpy(dtype="float64")
    sog = df["sog"].to_numpy(dtype="float64")
    cog = df["cog"].to_numpy(dtype="float64")
    if "heading" in df.columns:
        heading = df["heading"].to_numpy(dtype="float64")
    else:
        heading = np.full(n, np.nan)

    same = np.zeros(n, dtype=bool)
    same[1:] = trip[1:] == trip[:-1]
    dt = np.full(n, np.nan)
    dt[1:] = np.where(same[1:], np.diff(t), np.nan)
    seg_km = np.full(n, np.nan)
    seg_km[1:] = np.where(
        same[1:], haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]), np.nan)
    dcog = np.full(n, np.nan)
    dcog[1:] = np.where(same[1:], np.abs(ang_diff_deg(cog[1:], cog[:-1])), np.nan)
    turn_dps = np.where(dt > 0, dcog / dt, np.nan)
    sog_diff = np.full(n, np.nan)
    sog_diff[1:] = np.where(same[1:], np.diff(sog), np.nan)
    accel_kn_min = np.where(dt > 0, sog_diff / (dt / 60.0), np.nan)

    cog_sin = np.sin(np.radians(cog))
    cog_cos = np.cos(np.radians(cog))
    loiter = (sog < LOITER_SOG_KN).astype("float64")
    heading_ok = (~np.isnan(heading)).astype("float64")

    work = pd.DataFrame({
        "trip_id": trip,
        "t": t,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "seg_km": seg_km,
        "turn_dps": turn_dps,
        "accel_abs": np.abs(accel_kn_min),
        "accel_kn_min": accel_kn_min,
        "cog_sin": cog_sin,
        "cog_cos": cog_cos,
        "loiter": loiter,
        "heading_ok": heading_ok,
    })
    if "mmsi" in df.columns:
        work["mmsi"] = df["mmsi"].to_numpy()

    g = work.groupby("trip_id", sort=True)
    first = g.first()
    last = g.last()
    net_km = haversine_km(
        first["lat"].to_numpy(), first["lon"].to_numpy(),
        last["lat"].to_numpy(), last["lon"].to_numpy(),
    )
    path_km = g["seg_km"].sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        straightness = np.where(path_km > 0, net_km / path_km, np.nan)

    Rc = np.sqrt(g["cog_sin"].mean().to_numpy() ** 2
                 + g["cog_cos"].mean().to_numpy() ** 2)
    Rc = np.clip(Rc, 1e-9, 1.0)
    cog_circ_std_deg = np.degrees(np.sqrt(-2.0 * np.log(Rc)))

    sog_vals = g["sog"]
    turn_vals = g["turn_dps"]

    feat = pd.DataFrame({
        "trip_id": first.index.to_numpy(),
        "n_points": g.size().to_numpy(),
        "t_start": first["t"].to_numpy(),
        "t_end": last["t"].to_numpy(),
        "sog_mean": sog_vals.mean().to_numpy(),
        "sog_med": sog_vals.median().to_numpy(),
        "sog_p95": sog_vals.quantile(0.95).to_numpy(),
        "sog_max": sog_vals.max().to_numpy(),
        "sog_std": sog_vals.std().to_numpy(),
        "loiter_frac": g["loiter"].mean().to_numpy(),
        "path_km": path_km,
        "net_km": net_km,
        "straightness": straightness,
        "turn_rate_mean_dps": turn_vals.mean().to_numpy(),
        "turn_rate_p95_dps": turn_vals.quantile(0.95).to_numpy(),
        "cog_circ_std_deg": cog_circ_std_deg,
        "heading_avail_frac": g["heading_ok"].mean().to_numpy(),
        "accel_mean_abs": g["accel_abs"].mean().to_numpy(),
        "accel_std": g["accel_kn_min"].std().to_numpy(),
    })
    if "mmsi" in work.columns:
        feat["mmsi"] = first["mmsi"].to_numpy()
    feat["duration_s"] = (feat["t_end"] - feat["t_start"]).astype("float64")
    feat["span_s"] = feat["duration_s"]  # alias: observed time span in the slice
    return feat.reset_index(drop=True)


def last_window_mask(
    points: pd.DataFrame, window_s: float,
) -> pd.Series:
    """Boolean mask: points in the last ``window_s`` seconds of each trip."""
    t_end = points.groupby("trip_id", sort=False)["t"].transform("max")
    return points["t"] >= (t_end - float(window_s))


def extract_last_windows(
    points: pd.DataFrame,
    window_lengths_s: Iterable[float],
    *,
    min_points: int = 3,
    min_span_frac: float = 0.5,
) -> pd.DataFrame:
    """One feature row per ``(trip_id, window_s)`` using the last-W-seconds slice.

    Adds ``window_s``, ``window_complete`` (span ≥ min_span_frac · W and
    n_points ≥ min_points). Does **not** tile or slide — one window per track
    per length.
    """
    if points.empty:
        return pd.DataFrame()
    pts = points.sort_values(["trip_id", "t"], kind="mergesort").reset_index(drop=True)
    frames: list[pd.DataFrame] = []
    for w in window_lengths_s:
        w = float(w)
        mask = last_window_mask(pts, w)
        sub = pts.loc[mask]
        if sub.empty:
            continue
        feat = features_from_points(sub)
        if feat.empty:
            continue
        feat["window_s"] = w
        feat["window_complete"] = (
            (feat["n_points"] >= min_points)
            & (feat["span_s"] >= min_span_frac * w)
        )
        frames.append(feat)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
