"""Materialize defended-asset placements from AIS point traffic.

Turns the engagement-geometry *policy* into a concrete placements table
(lat/lon per seed × archetype) with validity-gate outcomes. Coordinates are
derived from the corpus, not hand-typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from counterusv.defense.engagement import EngagementGeometryConfig, PortRegion
from counterusv.defense.geometry_features import range_bearing_nm
from counterusv.kinematics.features import haversine_km

NM_PER_KM = 1.0 / 1.852
CELL_DEG = 0.1  # density cell for fairway / dwell clustering


def offset_nm(
    lat: float, lon: float, bearing_deg: float, distance_nm: float,
) -> tuple[float, float]:
    """Offset a WGS-84 point by ``distance_nm`` along ``bearing_deg`` (CW from N)."""
    br = np.radians(bearing_deg)
    dlat = (distance_nm * np.cos(br)) / 60.0
    dlon = (distance_nm * np.sin(br)) / (60.0 * np.cos(np.radians(lat)))
    return float(lat + dlat), float(lon + dlon)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, b = range_bearing_nm(np.array([lat2]), np.array([lon2]), lat1, lon1)
    return float(b[0])


def _in_radius_mask(
    lat: np.ndarray, lon: np.ndarray, clat: float, clon: float, radius_nm: float,
) -> np.ndarray:
    r = haversine_km(clat, clon, lat, lon) * NM_PER_KM
    return r <= radius_nm


def _densest_cell_median(
    lat: np.ndarray, lon: np.ndarray, cell_deg: float = CELL_DEG,
) -> tuple[float, float] | None:
    if lat.size == 0:
        return None
    i = np.floor(lat / cell_deg).astype(np.int64)
    j = np.floor(lon / cell_deg).astype(np.int64)
    # Encode cell key
    key = i.astype(np.int64) * 10_000_000 + j.astype(np.int64)
    vals, counts = np.unique(key, return_counts=True)
    best = vals[int(np.argmax(counts))]
    mask = key == best
    return float(np.median(lat[mask])), float(np.median(lon[mask]))


def _moored_dwell_points(
    points: pd.DataFrame,
    *,
    moored_sog_kn: float,
    moored_dwell_h: float,
) -> pd.DataFrame:
    """Points belonging to long-dwell nearly-stationary segments."""
    if points.empty:
        return points.iloc[0:0]
    dwell_s = float(moored_dwell_h) * 3600.0
    pts = points.sort_values(["trip_id", "t"], kind="mergesort").reset_index(drop=True)
    sog = pts["sog"].to_numpy(dtype="float64")
    moored = sog < float(moored_sog_kn)
    if not moored.any():
        return pts.iloc[0:0]

    # Contiguous moored runs per trip.
    trip = pts["trip_id"].to_numpy()
    t = pts["t"].to_numpy(dtype="float64")
    keep = np.zeros(len(pts), dtype=bool)
    i = 0
    n = len(pts)
    while i < n:
        if not moored[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and trip[j + 1] == trip[i] and moored[j + 1]:
            j += 1
        if (t[j] - t[i]) >= dwell_s:
            keep[i : j + 1] = True
        i = j + 1
    return pts.loc[keep]


def _cluster_medians(
    lat: np.ndarray, lon: np.ndarray, cell_deg: float = CELL_DEG, top_k: int = 8,
) -> list[tuple[float, float, int]]:
    """Return up to top_k densest cell medians as (lat, lon, count)."""
    if lat.size == 0:
        return []
    i = np.floor(lat / cell_deg).astype(np.int64)
    j = np.floor(lon / cell_deg).astype(np.int64)
    key = i.astype(np.int64) * 10_000_000 + j.astype(np.int64)
    vals, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts)
    out: list[tuple[float, float, int]] = []
    for idx in order[:top_k]:
        mask = key == vals[idx]
        out.append((
            float(np.median(lat[mask])),
            float(np.median(lon[mask])),
            int(counts[idx]),
        ))
    return out


@dataclass
class PlacementCandidate:
    port_region: str
    placement_class: str
    role: str
    lat: float
    lon: float
    asset_id: str = ""
    n_inbound_legs_in_annulus: int = 0
    min_observed_range_nm: float = float("nan")
    water_occupied: bool = False
    valid: bool = False
    reject_reason: str = ""

    def finalize_id(self) -> None:
        self.asset_id = f"{self.port_region}__{self.placement_class}"


def derive_seed_placements(
    points: pd.DataFrame,
    seed: PortRegion,
    cfg: EngagementGeometryConfig,
    *,
    search_nm: float | None = None,
) -> list[PlacementCandidate]:
    """Derive the four archetype placements for one seed region."""
    policy = cfg.placement_policy
    search = float(
        search_nm
        if search_nm is not None
        else policy.get("materialize_search_nm", 40)
    )
    classes = {c.name: c for c in cfg.placement_classes()}

    lat = points["lat"].to_numpy(dtype="float64")
    lon = points["lon"].to_numpy(dtype="float64")
    near = _in_radius_mask(lat, lon, seed.approx_lat, seed.approx_lon, search)
    local = points.loc[near].copy()
    if local.empty:
        # Degenerate seed — still emit rows at approx so the table is complete.
        return [
            _fallback_row(seed, name, classes[name].role, seed.approx_lat, seed.approx_lon,
                          reason="no_points_in_search_radius")
            for name in ("fairway_stress", "berth_approach", "anchorage", "offshore_terminal")
            if name in classes
        ]

    # --- fairway_stress ---
    fw = classes["fairway_stress"]
    moving_sog = float(fw.params.get("moving_sog_kn", 5.0))
    moving = local[local["sog"] >= moving_sog]
    fw_pos = _densest_cell_median(
        moving["lat"].to_numpy(dtype="float64"),
        moving["lon"].to_numpy(dtype="float64"),
    )
    if fw_pos is None:
        fw_pos = (seed.approx_lat, seed.approx_lon)
    fairway_lat, fairway_lon = fw_pos

    # --- berth_approach / anchorage from moored dwell ---
    berth_cls = classes["berth_approach"]
    moored_sog = float(berth_cls.params.get("moored_sog_kn", 0.5))
    moored_dwell_h = float(berth_cls.params.get("moored_dwell_h", 6.0))
    dwell = _moored_dwell_points(
        local, moored_sog_kn=moored_sog, moored_dwell_h=moored_dwell_h,
    )
    clusters = _cluster_medians(
        dwell["lat"].to_numpy(dtype="float64") if not dwell.empty else np.array([]),
        dwell["lon"].to_numpy(dtype="float64") if not dwell.empty else np.array([]),
    )

    # Prefer berth = densest cluster that is *not* seaward of fairway relative
    # to seed approx (harbour-interior heuristic: closer to seed approx than
    # fairway, or densest overall if that fails).
    berth_lat, berth_lon = fairway_lat, fairway_lon
    if clusters:
        # Score: prefer clusters landward of fairway (toward seed approx).
        seed_to_fw = bearing_deg(seed.approx_lat, seed.approx_lon, fairway_lat, fairway_lon)
        best = None
        best_score = -1.0
        for clat, clon, cnt in clusters:
            # Distance from fairway — larger separation preferred for anchorage later.
            d_fw = float(haversine_km(fairway_lat, fairway_lon, clat, clon) * NM_PER_KM)
            # Landward: cluster is on the seed side of fairway if bearing from
            # fairway to cluster is opposite seed→fairway (±90°).
            br_fc = bearing_deg(fairway_lat, fairway_lon, clat, clon)
            delta = abs((br_fc - (seed_to_fw + 180.0)) % 360.0)
            if delta > 180:
                delta = 360 - delta
            landward = 1.0 if delta <= 90.0 else 0.3
            score = cnt * landward
            if score > best_score:
                best_score = score
                best = (clat, clon)
        if best is not None:
            berth_lat, berth_lon = best

    # Nudge berth slightly toward fairway (water side) by 0.15 nm if needed.
    br_bf = bearing_deg(berth_lat, berth_lon, fairway_lat, fairway_lon)
    berth_lat, berth_lon = offset_nm(berth_lat, berth_lon, br_bf, 0.15)

    # Anchorage: densest cluster ≥ anchorage_min_sep_nm from berth, prefer seaward.
    anc_cls = classes["anchorage"]
    min_sep = float(anc_cls.params.get("anchorage_min_sep_nm", 1.0))
    anc_lat, anc_lon = offset_nm(
        fairway_lat, fairway_lon,
        bearing_deg(seed.approx_lat, seed.approx_lon, fairway_lat, fairway_lon),
        1.5,
    )
    if clusters:
        best = None
        best_score = -1.0
        for clat, clon, cnt in clusters:
            d_berth = float(haversine_km(berth_lat, berth_lon, clat, clon) * NM_PER_KM)
            if d_berth < min_sep:
                continue
            # Prefer seaward of fairway (same bearing as seed→fairway).
            br_fc = bearing_deg(fairway_lat, fairway_lon, clat, clon)
            seed_to_fw = bearing_deg(seed.approx_lat, seed.approx_lon, fairway_lat, fairway_lon)
            delta = abs((br_fc - seed_to_fw) % 360.0)
            if delta > 180:
                delta = 360 - delta
            seaward = 1.0 if delta <= 90.0 else 0.4
            score = cnt * seaward
            if score > best_score:
                best_score = score
                best = (clat, clon)
        if best is not None:
            anc_lat, anc_lon = best

    # Offshore terminal: seaward of fairway.
    off_cls = classes["offshore_terminal"]
    seaward_nm = float(off_cls.params.get("seaward_offset_nm", 4.0))
    seaward_br = bearing_deg(seed.approx_lat, seed.approx_lon, fairway_lat, fairway_lon)
    # Prefer opposite of berth if that is more seaward.
    br_away_berth = bearing_deg(berth_lat, berth_lon, fairway_lat, fairway_lon)
    seaward_br = br_away_berth if br_away_berth == br_away_berth else seaward_br
    off_lat, off_lon = offset_nm(fairway_lat, fairway_lon, seaward_br, seaward_nm)

    raw = [
        PlacementCandidate(seed.id, "berth_approach", berth_cls.role, berth_lat, berth_lon),
        PlacementCandidate(seed.id, "anchorage", anc_cls.role, anc_lat, anc_lon),
        PlacementCandidate(seed.id, "offshore_terminal", off_cls.role, off_lat, off_lon),
        PlacementCandidate(seed.id, "fairway_stress", fw.role, fairway_lat, fairway_lon),
    ]
    for row in raw:
        row.finalize_id()
        _apply_validity(row, local, cfg)
    return raw


def _fallback_row(
    seed: PortRegion, name: str, role: str, lat: float, lon: float, *, reason: str,
) -> PlacementCandidate:
    row = PlacementCandidate(seed.id, name, role, lat, lon, reject_reason=reason)
    row.finalize_id()
    row.valid = False
    return row


def _apply_validity(
    row: PlacementCandidate,
    local_points: pd.DataFrame,
    cfg: EngagementGeometryConfig,
) -> None:
    gates = dict(cfg.placement_policy.get("validity_gates") or {})
    min_legs = int(gates.get("min_inbound_legs_in_annulus", 50))
    max_min_range = float(gates.get("reject_if_min_observed_range_nm_gt", 3.5))
    require_water = bool(gates.get("require_over_water", True))
    min_pts = int(cfg.inbound_leg.get("min_points", 4))
    req_before = int(cfg.inbound_leg.get("require_points_before_cpa", 3))

    lat = local_points["lat"].to_numpy(dtype="float64")
    lon = local_points["lon"].to_numpy(dtype="float64")
    if lat.size == 0:
        row.valid = False
        row.reject_reason = "no_local_points"
        return

    ranges, _ = range_bearing_nm(lat, lon, row.lat, row.lon)
    row.min_observed_range_nm = float(np.nanmin(ranges))
    # Water-occupancy proxy (no shoreline product): any AIS fix within 1 nm.
    row.water_occupied = bool(np.any(ranges <= 1.0))

    # Fast inbound-leg proxy: trips with ≥min_points in annulus and ≥req_before
    # points before the CPA index (min-range sample). Avoids per-trip feature calls.
    work = local_points[["trip_id", "t"]].copy()
    work["range_nm"] = ranges
    work = work.loc[work["range_nm"] <= cfg.max_range_nm]
    n_ok = 0
    if not work.empty:
        for _, g in work.groupby("trip_id", sort=False):
            if len(g) < min_pts:
                continue
            cpa_i = int(g["range_nm"].to_numpy().argmin())
            if cpa_i >= req_before:
                n_ok += 1
    row.n_inbound_legs_in_annulus = int(n_ok)

    reasons: list[str] = []
    if require_water and not row.water_occupied:
        reasons.append("not_over_water")
    if row.n_inbound_legs_in_annulus < min_legs:
        reasons.append(f"inbound_legs<{min_legs}")
    if row.min_observed_range_nm > max_min_range:
        reasons.append(f"min_range>{max_min_range}")
    row.reject_reason = ",".join(reasons)
    row.valid = len(reasons) == 0


def materialize_placements(
    points: pd.DataFrame,
    cfg: EngagementGeometryConfig,
    *,
    seeds: Iterable[PortRegion] | None = None,
    search_nm: float | None = None,
) -> pd.DataFrame:
    """Build the full placements table for the given seeds (default: all)."""
    seed_list = list(seeds) if seeds is not None else cfg.port_regions()
    rows: list[dict[str, Any]] = []
    for seed in seed_list:
        for cand in derive_seed_placements(points, seed, cfg, search_nm=search_nm):
            rows.append({
                "asset_id": cand.asset_id,
                "port_region": cand.port_region,
                "placement_class": cand.placement_class,
                "role": cand.role,
                "lat": cand.lat,
                "lon": cand.lon,
                "n_inbound_legs_in_annulus": cand.n_inbound_legs_in_annulus,
                "min_observed_range_nm": cand.min_observed_range_nm,
                "water_occupied": cand.water_occupied,
                "valid": cand.valid,
                "reject_reason": cand.reject_reason,
            })
    return pd.DataFrame(rows)
