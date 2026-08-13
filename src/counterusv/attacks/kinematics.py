"""Eval-only adversary motion model (hostile / adaptive world-frame tracks).

Emits AIS-cadence point streams ``trip_id/t/lat/lon/sog/cog`` relative to a
named defended asset. Features must be derived via the shared kinematics and
geometry extractors — this module never invents a parallel feature path.

Firewall: every track is tagged ``role=hostile`` / ``source=synth`` and is
accepted for scoring only through ``purpose="eval"``. Never write generated
points into benign training corpora.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from counterusv.defense.consistency import FirewallError, assert_benign_train_allowed
from counterusv.kinematics.features import haversine_km

NM_PER_KM = 1.0 / 1.852
NM_PER_DEG_LAT = 60.0

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG = REPO_ROOT / "configs" / "attacks" / "kinematics.yaml"

BLOCKED_CORPUS_PREFIXES = (
    "data/behavior/",
    "data/defense/features_",
    "data/tracks_",
)


class EvalOnlyFirewallError(PermissionError):
    """Raised when synth tracks are steered toward a benign training path."""


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def load_kinematics_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_CFG
    if not p.is_absolute():
        p = REPO_ROOT / p
    return _load_yaml(p)


@dataclass(frozen=True)
class PlatformProfile:
    """Cited platform performance envelope."""

    key: str
    name: str
    cruise_kn: float
    burst_kn: float
    range_nm: float
    citations: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, key: str, block: Mapping[str, Any]) -> "PlatformProfile":
        return cls(
            key=key,
            name=str(block.get("name") or key),
            cruise_kn=float(block["cruise_kn"]),
            burst_kn=float(block["burst_kn"]),
            range_nm=float(block.get("range_nm") or 0.0),
            citations=tuple(block.get("citations") or ()),
        )


@dataclass(frozen=True)
class SweepCell:
    """One point in the frozen mimicry sweep grid."""

    cell_id: str
    platform: str
    mimicked_class: str
    v_mimic_kn: float
    commit_range_nm: float
    bearing_offset_deg: float
    use_burst_on_commit: bool = True
    unconstrained: bool = False
    negative_control: bool = False
    asset_id: str | None = None
    approach_bearing_deg: float = 0.0

    def peak_speed_kn(self, platform: PlatformProfile) -> float:
        if self.use_burst_on_commit:
            return max(float(self.v_mimic_kn), float(platform.burst_kn))
        # Negative-control / no-burst: stay at mimic speed (no cruise step-up).
        return float(self.v_mimic_kn)


@dataclass
class GeneratedTrack:
    """Synth track: points + eval-only metadata."""

    points: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def trip_id(self) -> str:
        return str(self.meta.get("trip_id") or "")


def destination_point(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_nm: float,
) -> tuple[float, float]:
    """Move ``distance_nm`` along ``bearing_deg`` (CW from N) from ``(lat, lon)``."""
    br = np.radians(bearing_deg % 360.0)
    dlat = (distance_nm * np.cos(br)) / NM_PER_DEG_LAT
    cos_lat = max(1e-6, abs(np.cos(np.radians(lat))))
    dlon = (distance_nm * np.sin(br)) / (NM_PER_DEG_LAT * cos_lat)
    return float(lat + dlat), float(lon + dlon)


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (deg CW from N) from point 1 to point 2."""
    alat = np.radians(lat1)
    plat = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(plat)
    y = np.cos(alat) * np.sin(plat) - np.sin(alat) * np.cos(plat) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def range_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return float(haversine_km(lat1, lon1, lat2, lon2) * NM_PER_KM)


def in_plausibility_band(
    peak_speed_kn: float,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> bool:
    """True if peak approach speed is within the corpus-supported band."""
    band = dict((cfg or {}).get("plausibility_band") or {})
    max_in = float(band.get("max_in_band_kn") or 40.0)
    return float(peak_speed_kn) <= max_in + 1e-9


def assert_eval_only(meta: Mapping[str, Any] | None) -> None:
    """Require hostile/synth tags; refuse if the track looks like train material."""
    if meta is None:
        raise EvalOnlyFirewallError("synth track metadata required")
    role = str(meta.get("role") or "").strip().lower()
    source = str(meta.get("source") or "").strip().lower()
    if role != "hostile":
        raise EvalOnlyFirewallError(
            f"adversary motion tracks must have role='hostile' (got {role!r})"
        )
    if source != "synth":
        raise EvalOnlyFirewallError(
            f"adversary motion tracks must have source='synth' (got {source!r})"
        )
    try:
        assert_benign_train_allowed(meta)
    except FirewallError:
        return
    raise EvalOnlyFirewallError(
        "firewall inconsistency: synth/hostile meta was accepted for training"
    )


def refuse_benign_corpus_write(path: Path | str) -> None:
    """Block writes into benign feature / track corpora."""
    rel = str(path).replace("\\", "/")
    for prefix in BLOCKED_CORPUS_PREFIXES:
        if prefix in rel or rel.startswith(prefix):
            raise EvalOnlyFirewallError(
                f"refusing to write synth tracks into benign corpus path: {rel}"
            )


def thin_to_cadence(
    points: pd.DataFrame,
    cadence_s: float = 60.0,
) -> pd.DataFrame:
    """Keep the first point and then points at least ``cadence_s`` apart."""
    if points.empty:
        return points.copy()
    pts = points.sort_values("t", kind="mergesort").reset_index(drop=True)
    keep: list[int] = [0]
    last_t = float(pts.loc[0, "t"])
    for i in range(1, len(pts)):
        t = float(pts.loc[i, "t"])
        if t - last_t >= float(cadence_s) - 1e-9:
            keep.append(i)
            last_t = t
    last_i = len(pts) - 1
    if keep[-1] != last_i and float(pts.loc[last_i, "t"]) - last_t >= 1.0:
        keep.append(last_i)
    return pts.loc[keep].reset_index(drop=True)


def _clamp_speed(v: float, platform: PlatformProfile) -> float:
    return float(min(max(v, 0.0), platform.burst_kn))


def _clamp_turn(
    prev_cog: float,
    desired_cog: float,
    dt_s: float,
    max_turn_rate_dps: float,
) -> float:
    if dt_s <= 0:
        return float(desired_cog % 360.0)
    a = (desired_cog - prev_cog + 540.0) % 360.0 - 180.0
    max_step = float(max_turn_rate_dps) * float(dt_s)
    a = float(np.clip(a, -max_step, max_step))
    return float((prev_cog + a) % 360.0)


def generate_two_phase_track(
    asset_lat: float,
    asset_lon: float,
    cell: SweepCell,
    platform: PlatformProfile,
    *,
    cfg: Mapping[str, Any] | None = None,
    trip_id: str | None = None,
    t0: float = 1_700_000_000.0,
) -> GeneratedTrack:
    """Synthesize a fine-timestep two-phase approach, then thin to AIS cadence.

    Phase 1 (mimic): heading = bearing_to_asset + offset at ``v_mimic``.
    Phase 2 (commit): from ``commit_range``, head toward the asset at burst
    (or cruise for negative controls) until ``terminal_range_nm``.
    """
    cfg = dict(cfg or {})
    dyn = dict(cfg.get("dynamics") or {})
    max_turn = float(dyn.get("max_turn_rate_dps") or 8.0)
    max_accel = float(dyn.get("max_accel_kn_per_s") or 0.5)
    dt = float(cfg.get("synth_dt_s") or 1.0)
    thin_s = float(cfg.get("thin_cadence_s") or 60.0)
    start_r = float(cfg.get("start_range_nm") or 12.0)
    terminal_r = float(cfg.get("terminal_range_nm") or 0.3)

    v_mimic = _clamp_speed(float(cell.v_mimic_kn), platform)
    if cell.use_burst_on_commit:
        v_commit = max(v_mimic, _clamp_speed(float(platform.burst_kn), platform))
    else:
        # Full-fidelity mimic: no aggressive speed-up on the late commit.
        v_commit = v_mimic

    approach = float(cell.approach_bearing_deg) % 360.0
    lat, lon = destination_point(asset_lat, asset_lon, approach, start_r)
    sog = v_mimic
    br_to_asset = bearing_between(lat, lon, asset_lat, asset_lon)
    cog = (br_to_asset + float(cell.bearing_offset_deg)) % 360.0

    tid = trip_id or cell.cell_id
    rows: list[dict[str, Any]] = []
    t = float(t0)
    phase = "mimic"
    max_steps = int(6 * 3600 / max(dt, 1e-3))

    for _ in range(max_steps):
        r = range_nm(lat, lon, asset_lat, asset_lon)
        rows.append({
            "trip_id": tid, "t": t, "lat": lat, "lon": lon,
            "sog": sog, "cog": cog, "phase": phase,
        })
        if r <= terminal_r:
            break

        if phase == "mimic" and r <= float(cell.commit_range_nm):
            phase = "commit"

        br_to_asset = bearing_between(lat, lon, asset_lat, asset_lon)
        if phase == "mimic":
            desired_cog = (br_to_asset + float(cell.bearing_offset_deg)) % 360.0
            desired_sog = v_mimic
        else:
            desired_cog = br_to_asset
            desired_sog = v_commit

        cog = _clamp_turn(cog, desired_cog, dt, max_turn)
        ds = float(np.clip(desired_sog - sog, -max_accel * dt, max_accel * dt))
        sog = _clamp_speed(sog + ds, platform)
        step_nm = sog * dt / 3600.0
        lat, lon = destination_point(lat, lon, cog, step_nm)
        t += dt

        if phase == "commit" and r < terminal_r * 2 and step_nm > r:
            lat, lon = destination_point(asset_lat, asset_lon, approach, terminal_r)
            rows.append({
                "trip_id": tid, "t": t, "lat": lat, "lon": lon,
                "sog": sog, "cog": cog, "phase": phase,
            })
            break

    fine = pd.DataFrame(rows)
    phase_vals = sorted({str(p) for p in fine["phase"].unique()}) if len(fine) else []
    pts = fine.drop(columns=["phase"], errors="ignore")
    thinned = thin_to_cadence(pts, thin_s)

    peak = cell.peak_speed_kn(platform)
    meta = {
        "trip_id": tid,
        "role": "hostile",
        "source": "synth",
        "canonical_class": "usv",
        "asset_id": cell.asset_id,
        "platform": platform.key,
        "mimicked_class": cell.mimicked_class,
        "v_mimic_kn": float(cell.v_mimic_kn),
        "commit_range_nm": float(cell.commit_range_nm),
        "bearing_offset_deg": float(cell.bearing_offset_deg),
        "use_burst_on_commit": bool(cell.use_burst_on_commit),
        "unconstrained": bool(cell.unconstrained),
        "negative_control": bool(cell.negative_control),
        "approach_bearing_deg": approach,
        "peak_speed_kn": float(peak),
        "in_plausibility_band": in_plausibility_band(peak, cfg=cfg),
        "thin_cadence_s": thin_s,
        "n_fine": int(len(fine)),
        "n_thinned": int(len(thinned)),
        "cell_id": cell.cell_id,
        "phases_present": phase_vals,
    }
    assert_eval_only(meta)
    return GeneratedTrack(points=thinned, meta=meta)


def platforms_from_config(cfg: Mapping[str, Any]) -> dict[str, PlatformProfile]:
    out: dict[str, PlatformProfile] = {}
    for key, block in (cfg.get("platforms") or {}).items():
        out[str(key)] = PlatformProfile.from_config(str(key), block)
    return out


def _cell_id(
    *,
    platform: str,
    mimicked_class: str,
    v_mimic: float,
    commit: float,
    offset: float,
    unconstrained: bool,
    negative_control: bool,
    asset_id: str,
) -> str:
    tag = "nc" if negative_control else ("unc" if unconstrained else "swp")
    return (
        f"{tag}__{platform}__{mimicked_class}__"
        f"v{v_mimic:g}_c{commit:g}_b{offset:g}__{asset_id}"
    )


def build_sweep_cells(
    cfg: Mapping[str, Any],
    assets: pd.DataFrame,
    *,
    approach_bearings: Mapping[str, float] | None = None,
) -> list[SweepCell]:
    """Enumerate the frozen sweep x assets (headline fit placements by default)."""
    sweep = dict(cfg.get("sweep") or {})
    nc = dict(cfg.get("negative_control") or {})
    platforms = list(sweep.get("platforms") or list((cfg.get("platforms") or {})))
    classes = list(sweep.get("mimicked_classes") or ["fishing", "recreational"])
    v_list = [float(v) for v in (sweep.get("v_mimic_kn") or [8, 25])]
    c_list = [float(c) for c in (sweep.get("commit_range_nm") or [2.0])]
    b_list = [float(b) for b in (sweep.get("bearing_offset_deg") or [0.0])]
    include_unc = bool(sweep.get("include_unconstrained", True))
    bearings = dict(approach_bearings or {})

    roles = set(cfg.get("headline_placement_roles") or ["fit"])
    include_far = bool(cfg.get("include_far_only", False))
    rows = assets.copy()
    if "valid" in rows.columns:
        rows = rows.loc[rows["valid"]].copy()
    if "role" in rows.columns and not include_far:
        rows = rows.loc[rows["role"].isin(roles)].copy()

    plat_map = platforms_from_config(cfg)
    eng = _load_yaml(
        REPO_ROOT / (cfg.get("engagement_geometry")
                     or "configs/defense/engagement_geometry.yaml")
    )
    max_r = float((eng.get("annulus") or {}).get("max_range_nm") or 6.0)

    cells: list[SweepCell] = []
    rng = np.random.default_rng(int(cfg.get("seed") or 1337))

    for _, asset in rows.iterrows():
        asset_id = str(asset["asset_id"])
        ab = float(bearings[asset_id]) if asset_id in bearings else float(rng.uniform(0.0, 360.0))
        for plat_key in platforms:
            if plat_key not in plat_map:
                raise KeyError(f"unknown platform {plat_key!r}")
            plat = plat_map[plat_key]
            for cls in classes:
                cells.append(SweepCell(
                    cell_id=_cell_id(
                        platform=plat_key, mimicked_class=cls,
                        v_mimic=float(nc.get("v_mimic_kn") or 8.0),
                        commit=float(nc.get("commit_range_nm") or 0.5),
                        offset=float(nc.get("bearing_offset_deg") or 0.0),
                        unconstrained=False, negative_control=True,
                        asset_id=asset_id,
                    ),
                    platform=plat_key,
                    mimicked_class=str(cls),
                    v_mimic_kn=float(nc.get("v_mimic_kn") or 8.0),
                    commit_range_nm=float(nc.get("commit_range_nm") or 0.5),
                    bearing_offset_deg=float(nc.get("bearing_offset_deg") or 0.0),
                    use_burst_on_commit=bool(nc.get("use_burst_on_commit", False)),
                    unconstrained=False,
                    negative_control=True,
                    asset_id=asset_id,
                    approach_bearing_deg=ab,
                ))
                for v in v_list:
                    for c in c_list:
                        for b in b_list:
                            cells.append(SweepCell(
                                cell_id=_cell_id(
                                    platform=plat_key, mimicked_class=cls,
                                    v_mimic=v, commit=c, offset=b,
                                    unconstrained=False, negative_control=False,
                                    asset_id=asset_id,
                                ),
                                platform=plat_key,
                                mimicked_class=str(cls),
                                v_mimic_kn=v,
                                commit_range_nm=c,
                                bearing_offset_deg=b,
                                use_burst_on_commit=True,
                                unconstrained=False,
                                negative_control=False,
                                asset_id=asset_id,
                                approach_bearing_deg=ab,
                            ))
                if include_unc:
                    cells.append(SweepCell(
                        cell_id=_cell_id(
                            platform=plat_key, mimicked_class=cls,
                            v_mimic=plat.burst_kn, commit=max_r, offset=0.0,
                            unconstrained=True, negative_control=False,
                            asset_id=asset_id,
                        ),
                        platform=plat_key,
                        mimicked_class=str(cls),
                        v_mimic_kn=float(plat.burst_kn),
                        commit_range_nm=max_r,
                        bearing_offset_deg=0.0,
                        use_burst_on_commit=True,
                        unconstrained=True,
                        negative_control=False,
                        asset_id=asset_id,
                        approach_bearing_deg=ab,
                    ))
    return cells


def cells_to_frame(cells: Sequence[SweepCell]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in cells])


def digest_cells_frame(df: pd.DataFrame) -> str:
    """Stable SHA-256 over a canonical serialization of the sweep table."""
    cols = sorted(df.columns)
    sort_cols = [c for c in ("cell_id",) if c in cols] or cols
    canon = df.loc[:, cols].sort_values(by=sort_cols).reset_index(drop=True)
    payload = canon.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_sweep_manifest(
    *,
    cfg: Mapping[str, Any],
    cells: pd.DataFrame,
    placements_sha: str | None,
    out_path: Path,
) -> dict[str, Any]:
    """Write FROZEN_SWEEP.json with grid digest and **no DDR numbers**."""
    refuse_benign_corpus_write(out_path)
    cell_sha = digest_cells_frame(cells)
    freeze = {
        "version": str(cfg.get("frozen") or "1"),
        "feature_arm_consumer": "kinematics + kinematics_geometry",
        "thin_cadence_s": cfg.get("thin_cadence_s"),
        "seed": cfg.get("seed"),
        "sweep": cfg.get("sweep"),
        "negative_control": cfg.get("negative_control"),
        "platforms": {
            k: {
                "cruise_kn": v.get("cruise_kn"),
                "burst_kn": v.get("burst_kn"),
                "citations": v.get("citations"),
            }
            for k, v in (cfg.get("platforms") or {}).items()
        },
        "n_cells": int(len(cells)),
        "digests": {
            "sweep_cells": cell_sha,
            "placements": placements_sha,
        },
        "paths": {
            "cells": "results/adversary_motion/sweep_cells.parquet",
            "points": "results/adversary_motion/tracks_points.parquet",
        },
        "note": (
            "Sweep grid frozen BEFORE any DDR / cost-curve scoring. "
            "Validity smoke may score negative controls; those rates are not "
            "DDR claims and must not be written into this freeze."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(freeze, indent=2) + "\n")
    return freeze


def load_placements(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    path = REPO_ROOT / (cfg.get("placements") or "data/defense/placements.parquet")
    df = pd.read_parquet(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    digest_path = REPO_ROOT / (
        cfg.get("placements_digest") or "data/defense/placements_digest.json"
    )
    if digest_path.is_file():
        dig = json.loads(digest_path.read_text())
        exp = ((dig.get("digests") or {}).get("placements.parquet"))
        if exp and exp != h:
            raise ValueError(
                f"placements digest mismatch: got {h[:16]}... expected {exp[:16]}..."
            )
    return df, h
