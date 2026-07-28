#!/usr/bin/env python3
"""Real trajectory corpus: AIS ingestion & cleaning .

Feeds the **benign-behavior model** . AIS is used **offline only** — never
a runtime input (see docs/THREAT_MODEL.md). Hostile/adaptive trajectories are NOT
here; they are synthesized at evaluation at evaluation. This keeps the benign side of
the class-kinematics consistency check entirely real-data.

Pipeline
--------
1. **Ingest** the MarineCadastre daily national CSVs (MMSI, time, lat/lon, SOG, COG,
   heading, numeric VesselType, TransceiverClass A/B). Class labels come from the
   self-reported ``VesselType`` mapped through ``data/taxonomy.yaml``'s
   ``ais_ship_type`` table — the SAME canonical classes the EO detector asserts, so
   the consistency check is well-defined.
2. **Clean**: valid-coordinate / valid-SOG-COG-heading filters, exact dedup,
   implausible-jump (teleport) removal, cadence thinning to a fixed interval, and
   gap-based segmentation into trip-level tracks.
3. **Features**: per-track class-conditional kinematics — speed distribution,
   acceleration, straightness/sinuosity, turn rate, heading stability, loiter
   fraction. ``range_to_shore`` is intentionally **omitted** : it needs
   reliable shoreline data + cross-region normalization and otherwise encodes
   geography rather than vessel behavior.
4. **Report** the Class-B (small-craft) share and the carriage/self-report bias
   honestly.

Outputs
-------
* ``data/tracks/tracks_ais.parquet`` — one row per trip-level track: class +
  kinematic features + provenance (the named AIS deliverable).
* ``data/tracks/tracks_ais_points.parquet`` — the cleaned, thinned point-level tracks
  (for windowed / time-to-flag scoring in behavior modeling and evaluation).
* ``results/ais_ingest/report.md`` + ``figures/*.png`` + ``summary.json``.

Note: DMA (Danish) archives are present on disk as *supplementary* coverage but use
a different (text ship-type) schema; a DMA adapter is a clean future extension —
``read_marinecadastre`` is the only source adapter wired here.

Usage
-----
    python scripts/data/ingest_ais.py # all MarineCadastre days
    python scripts/data/ingest_ais.py --days 2023-06-01 2023-06-02
    python scripts/data/ingest_ais.py --bbox 32 42 -125 -117 # lat_min lat_max lon_min lon_max
    python scripts/data/ingest_ais.py --cadence 60 --gap 1800
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results" / "ais_ingest"
MC_DIR = "raw/ais/marinecadastre"

# --- cleaning / segmentation thresholds (documented; provisional) -----------
MAX_IMPLIED_SPEED_KN = 80.0 # implied point-to-point speed above this = teleport
GAP_S = 1800 # >30 min gap starts a new trip-level track
CADENCE_S = 60 # thin to <=1 point per this interval (native ~1 min)
LOITER_SOG_KN = 0.5 # SOG below this counts as stopped/loitering
SOG_MAX_VALID = 102.2 # 102.3 is the AIS "not available" sentinel
MIN_TRIP_POINTS = 5 # trips shorter than this are dropped from features
MIN_TRIP_DURATION_S = 300 # ... and shorter than 5 min
KNOTS_PER_MS = 1.943844 # m/s -> knots
EARTH_R_KM = 6371.0088


# ---------------------------------------------------------------------------
# Taxonomy: AIS ship-type code -> canonical class
# ---------------------------------------------------------------------------
def load_ais_map(tax_path: Path):
    import yaml
    tax = yaml.safe_load(tax_path.read_text())
    ais = tax["ais_ship_type"]
    by_code = {int(k): v for k, v in ais.get("by_code", {}).items()}
    by_range = []
    for rng, canon in ais.get("by_range", {}).items():
        lo, hi = (int(x) for x in str(rng).split("-"))
        by_range.append((lo, hi, canon))
    roles = {name: meta.get("role", "benign")
             for name, meta in tax["canonical_classes"].items()}
    return by_code, by_range, roles


def code_to_canonical(code, by_code, by_range):
    """Map a numeric VesselType to a canonical class, or None if unmapped."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    c = int(code)
    if c in by_code:
        return by_code[c]
    for lo, hi, canon in by_range:
        if lo <= c <= hi:
            return canon
    return None


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
MC_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading",
           "VesselType", "TransceiverClass"]
MC_DTYPE = {"MMSI": "int64", "LAT": "float32", "LON": "float32", "SOG": "float32",
            "COG": "float32", "Heading": "float32", "VesselType": "float32",
            "TransceiverClass": "category"}


def read_marinecadastre(zip_paths, bbox=None, chunksize=2_000_000, verbose=True):
    """Read + row-filter the MarineCadastre daily CSVs into one point DataFrame.

    Row-level cleaning done here (cheap, shrinks memory before the global sort):
    valid coordinates, valid SOG/COG/heading, drop null MMSI. bbox is
    (lat_min, lat_max, lon_min, lon_max) if given.
    """
    frames = []
    for zp in zip_paths:
        t0 = time.time()
        n_raw = n_keep = 0
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(name) as fh:
                for ch in pd.read_csv(fh, usecols=MC_COLS, dtype=MC_DTYPE,
                                      chunksize=chunksize):
                    n_raw += len(ch)
                    ch = _clean_rows(ch, bbox)
                    n_keep += len(ch)
                    if len(ch):
                        frames.append(ch)
        if verbose:
            print(f"[ingest] {Path(zp).name}: raw={n_raw:,} kept={n_keep:,} "
                  f"({time.time()-t0:.1f}s)")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def _clean_rows(ch, bbox):
    # timestamp -> int64 epoch seconds (fixed format for speed)
    t = pd.to_datetime(ch["BaseDateTime"], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    ch = ch.assign(t=(t.astype("int64") // 1_000_000_000))
    lat, lon = ch["LAT"], ch["LON"]
    ok = (lat.between(-90, 90) & lon.between(-180, 180)
          & ~((lat == 0) & (lon == 0)) & ch["t"].notna() & (ch["t"] > 0))
    if bbox is not None:
        la0, la1, lo0, lo1 = bbox
        ok &= lat.between(la0, la1) & lon.between(lo0, lo1)
    ch = ch[ok].copy()
    # invalid SOG/COG/heading -> NaN (do not drop the row; position is still good)
    ch.loc[~ch["SOG"].between(0, SOG_MAX_VALID), "SOG"] = np.nan
    ch.loc[~ch["COG"].between(0, 360, inclusive="left"), "COG"] = np.nan
    ch.loc[~ch["Heading"].between(0, 360, inclusive="left"), "Heading"] = np.nan
    return ch[["MMSI", "t", "LAT", "LON", "SOG", "COG", "Heading",
               "VesselType", "TransceiverClass"]]


# ---------------------------------------------------------------------------
# Geometry helpers (vectorized)
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def ang_diff_deg(a, b):
    """Smallest signed angular difference a-b in degrees, wrapped to [-180,180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


# ---------------------------------------------------------------------------
# Clean -> segment -> thin
# ---------------------------------------------------------------------------
def assign_static_class(df, by_code, by_range, roles):
    """Per-MMSI modal VesselType -> canonical class + role + transceiver."""
    def modal(s):
        s = s.dropna()
        if s.empty:
            return np.nan
        return s.mode().iloc[0]

    agg = df.groupby("MMSI").agg(
        vessel_type_code=("VesselType", modal),
        transceiver_class=("TransceiverClass",
                           lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "?"),
    )
    agg["canonical_class"] = agg["vessel_type_code"].map(
        lambda c: code_to_canonical(c, by_code, by_range))
    agg["canonical_class"] = agg["canonical_class"].fillna("unknown_other")
    agg["role"] = agg["canonical_class"].map(roles).fillna("non_target")
    return agg


def segment_and_thin(df, cadence_s, gap_s):
    """Sort, drop teleports, thin to cadence, and cut trips on gaps.

    Returns the point-level df with a ``trip_id`` column, sorted by (MMSI, t).
    """
    df = df.sort_values(["MMSI", "t"], kind="stable").reset_index(drop=True)
    # exact (MMSI, t) duplicates -> keep first
    df = df[~df.duplicated(["MMSI", "t"], keep="first")].reset_index(drop=True)

    same = df["MMSI"].values[1:] == df["MMSI"].values[:-1]
    dt = np.diff(df["t"].values).astype("float64")
    seg = haversine_km(df["LAT"].values[:-1], df["LON"].values[:-1],
                       df["LAT"].values[1:], df["LON"].values[1:])
    implied_kn = np.where((dt > 0) & same, seg / (dt / 3600.0), 0.0) # km/h-ish
    implied_kn = implied_kn / 1.852 # km/h -> knots
    teleport = np.zeros(len(df), dtype=bool)
    teleport[1:] = same & (implied_kn > MAX_IMPLIED_SPEED_KN)
    n_teleport = int(teleport.sum())
    df = df[~teleport].reset_index(drop=True)

    # cadence thinning: keep first point per (MMSI, floor(t/cadence)) bin
    if cadence_s and cadence_s > 1:
        binid = (df["t"].values // cadence_s)
        keep = ~pd.Series(list(zip(df["MMSI"].values, binid))).duplicated(keep="first").values
        df = df[keep].reset_index(drop=True)

    # trip segmentation on time gaps / mmsi change
    same = np.zeros(len(df), dtype=bool)
    same[1:] = df["MMSI"].values[1:] == df["MMSI"].values[:-1]
    dt = np.zeros(len(df), dtype="float64")
    dt[1:] = np.diff(df["t"].values)
    new_trip = (~same) | (dt > gap_s) | (dt <= 0)
    df["trip_id"] = np.cumsum(new_trip)
    return df, n_teleport


# ---------------------------------------------------------------------------
# Features (vectorized per trip)
# ---------------------------------------------------------------------------
def compute_features(df):
    """One row per trip_id with class-conditional kinematic features."""
    n = len(df)
    same = np.zeros(n, dtype=bool)
    same[1:] = df["trip_id"].values[1:] == df["trip_id"].values[:-1]
    dt = np.zeros(n, dtype="float64")
    dt[1:] = np.where(same, np.diff(df["t"].values), np.nan)
    seg_km = np.zeros(n, dtype="float64")
    seg_km[1:] = np.where(same, haversine_km(
        df["LAT"].values[:-1], df["LON"].values[:-1],
        df["LAT"].values[1:], df["LON"].values[1:]), np.nan)
    # course change from COG; turn rate deg/s
    cog = df["COG"].values
    dcog = np.full(n, np.nan)
    dcog[1:] = np.where(same, np.abs(ang_diff_deg(cog[1:], cog[:-1])), np.nan)
    turn_dps = np.where(dt > 0, dcog / dt, np.nan)
    # speed from position (knots) as a cross-check / fallback
    pos_speed_kn = np.where(dt > 0, (seg_km / 1.852) / (dt / 3600.0), np.nan)

    work = pd.DataFrame({
        "trip_id": df["trip_id"].values, "MMSI": df["MMSI"].values,
        "t": df["t"].values, "lat": df["LAT"].values, "lon": df["LON"].values,
        "sog": df["SOG"].values, "cog": cog, "heading": df["Heading"].values,
        "seg_km": seg_km, "dt": dt, "turn_dps": turn_dps, "pos_speed_kn": pos_speed_kn,
        "cog_sin": np.sin(np.radians(cog)), "cog_cos": np.cos(np.radians(cog)),
    })
    g = work.groupby("trip_id", sort=True)

    first = g.first()
    last = g.last()
    net_km = haversine_km(first["lat"].values, first["lon"].values,
                          last["lat"].values, last["lon"].values)
    path_km = g["seg_km"].sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        straightness = np.where(path_km > 0, net_km / path_km, np.nan)

    # circular std of course: R = |mean unit vector|; std = sqrt(-2 ln R)
    Rc = np.sqrt(g["cog_sin"].mean().values ** 2 + g["cog_cos"].mean().values ** 2)
    Rc = np.clip(Rc, 1e-9, 1.0)
    cog_circ_std_deg = np.degrees(np.sqrt(-2.0 * np.log(Rc)))

    feat = pd.DataFrame({
        "trip_id": first.index.values,
        "mmsi": first["MMSI"].values.astype("int64"),
        "n_points": g.size().values,
        "t_start": first["t"].values, "t_end": last["t"].values,
        "mean_lat": g["lat"].mean().values, "mean_lon": g["lon"].mean().values,
        "sog_mean": g["sog"].mean().values, "sog_med": g["sog"].median().values,
        "sog_p95": g["sog"].quantile(0.95).values, "sog_max": g["sog"].max().values,
        "sog_std": g["sog"].std().values,
        "loiter_frac": g["sog"].apply(lambda s: float((s < LOITER_SOG_KN).mean())).values,
        "path_km": path_km, "net_km": net_km, "straightness": straightness,
        "turn_rate_mean_dps": g["turn_dps"].mean().values,
        "turn_rate_p95_dps": g["turn_dps"].quantile(0.95).values,
        "cog_circ_std_deg": cog_circ_std_deg,
        "heading_avail_frac": g["heading"].apply(lambda s: float(s.notna().mean())).values,
        "pos_speed_mean_kn": g["pos_speed_kn"].mean().values,
    })
    feat["duration_s"] = (feat["t_end"] - feat["t_start"]).astype("int64")
    # acceleration (knots per minute) from SOG differences
    work["sog_prev"] = g["sog"].shift(1)
    work["accel_kn_min"] = (work["sog"] - work["sog_prev"]) / (work["dt"] / 60.0)
    acc = work.groupby("trip_id")["accel_kn_min"].agg(
        accel_mean_abs=lambda s: s.abs().mean(), accel_std="std")
    feat = feat.merge(acc, on="trip_id", how="left")
    return feat


# ---------------------------------------------------------------------------
# Report + figures
# ---------------------------------------------------------------------------
def make_figures(tracks, fig_dir):
    import os
    fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(fig_dir.parent / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benign = tracks[tracks["role"] == "benign"]
    classes = (benign["canonical_class"].value_counts().index.tolist())

    # 1. tracks per canonical class (colored by transceiver)
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = tracks.pivot_table(index="canonical_class", columns="transceiver_class",
                             values="trip_id", aggfunc="count", fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    piv.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("trip-level tracks"); ax.set_yscale("log")
    ax.set_title("AIS tracks per canonical class (by transceiver A/B)")
    ax.set_xlabel(""); ax.legend(title="class", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "tracks_per_class.png", dpi=120)
    plt.close(fig)

    # 2. speed distribution per benign class (box)
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [benign.loc[benign["canonical_class"] == c, "sog_mean"].dropna().values
            for c in classes]
    ax.boxplot(data, labels=classes, showfliers=False)
    ax.set_ylabel("mean SOG (knots)"); ax.set_title("Speed by benign class")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(fig_dir / "speed_by_class.png", dpi=120)
    plt.close(fig)

    # 3. straightness vs loiter fraction per class (medians)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in classes:
        sub = benign[benign["canonical_class"] == c]
        ax.scatter(sub["loiter_frac"].median(), sub["straightness"].median(),
                   s=60, label=c)
    ax.set_xlabel("median loiter fraction"); ax.set_ylabel("median straightness")
    ax.set_title("Kinematic separation of benign classes"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "straightness_vs_loiter.png", dpi=120)
    plt.close(fig)

    # 4. turn-rate distribution per class
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [benign.loc[benign["canonical_class"] == c, "turn_rate_mean_dps"].dropna().values
            for c in classes]
    ax.boxplot(data, labels=classes, showfliers=False)
    ax.set_ylabel("mean turn rate (deg/s)"); ax.set_title("Turn rate by benign class")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(fig_dir / "turn_rate_by_class.png", dpi=120)
    plt.close(fig)


def build_summary(tracks, points_n, meta):
    by_class = tracks.groupby("canonical_class").agg(
        tracks=("trip_id", "count"), vessels=("mmsi", "nunique"),
        sog_mean_med=("sog_mean", "median"),
        straightness_med=("straightness", "median"),
        loiter_med=("loiter_frac", "median")).round(4)
    by_class = by_class.sort_values("tracks", ascending=False)
    tc = tracks["transceiver_class"].value_counts()
    b_tracks = int(tc.get("B", 0)); a_tracks = int(tc.get("A", 0))
    n_tracks = int(len(tracks))
    classb = tracks[tracks["transceiver_class"] == "B"]
    classb_classes = classb["canonical_class"].value_counts().to_dict()
    unmapped = meta.get("unmapped_codes", {})
    unknown_tracks = int((tracks["canonical_class"] == "unknown_other").sum())
    return {
        "region": meta["region"], "days": meta["days"],
        "params": meta["params"],
        "point_level": {
            "raw_rows": meta["raw_rows"], "kept_rows_after_row_clean": meta["kept_rows"],
            "teleport_points_dropped": meta["n_teleport"],
            "points_after_thinning": int(points_n),
        },
        "tracks": {
            "total": n_tracks, "vessels": int(tracks["mmsi"].nunique()),
            "dropped_short_tracks": meta["dropped_short"],
            "class_b_share_tracks": round(b_tracks / max(n_tracks, 1), 4),
            "class_a_tracks": a_tracks, "class_b_tracks": b_tracks,
            "unknown_class_tracks": unknown_tracks,
        },
        "class_b_breakdown": classb_classes,
        "per_class": json.loads(by_class.reset_index().to_json(orient="records")),
        "self_report_bias": {
            "note": ("AIS ship-type is self-reported and Class-B units under-report; "
                     "AIS carries no small_craft code (the taxonomy), so small benign "
                     "craft appear as recreational(37)/fishing(30)/unknown. VesselType "
                     "assigned per-MMSI by mode; unmappable/absent -> unknown_other and "
                     "excluded from class-conditional envelopes. The video-derived "
                     "non-cooperative set and real-track validation (6.5) bound "
                     "this bias."),
            "unmapped_vessel_type_codes": unmapped,
        },
    }


def write_report(summary, path):
    s = summary
    tr = s["tracks"]; pl = s["point_level"]
    L = ["# AIS trajectory corpus — ingestion & cleaning \n",
         "Auto-generated by `scripts/data/ingest_ais.py`. AIS is used **offline only** "
         "(never a runtime input). Feeds the class-conditional benign-behavior model "
         "; hostile/adaptive trajectories are synthesized at eval, never here.\n",
         f"**Source/region:** {s['region']} ",
         f"**Days:** {', '.join(s['days'])} ",
         f"**Params:** cadence={s['params']['cadence_s']}s, gap={s['params']['gap_s']}s, "
         f"teleport>{s['params']['max_implied_kn']}kn, loiter<{s['params']['loiter_kn']}kn, "
         f"min_trip={s['params']['min_trip_points']}pts/{s['params']['min_trip_dur_s']}s"
         + (f", bbox={s['params']['bbox']}" if s['params'].get('bbox') else "") + "\n",
         "## Volume\n",
         f"- Raw messages read: **{pl['raw_rows']:,}** → after row-clean "
         f"(valid coords/time): **{pl['kept_rows_after_row_clean']:,}**\n",
         f"- Teleport points dropped (implied speed > {s['params']['max_implied_kn']}kn): "
         f"**{pl['teleport_points_dropped']:,}**\n",
         f"- Cleaned point-level rows (after {s['params']['cadence_s']}s thinning): "
         f"**{pl['points_after_thinning']:,}** → `data/tracks/tracks_ais_points.parquet`\n",
         f"- **Trip-level tracks: {tr['total']:,}** over **{tr['vessels']:,}** vessels "
         f"(dropped {tr['dropped_short_tracks']:,} sub-threshold trips) → "
         "`data/tracks/tracks_ais.parquet`\n",
         "## Class-B (small-craft) share — the carriage-bias headline\n",
         f"- **Class-B tracks: {tr['class_b_tracks']:,} / {tr['total']:,} = "
         f"{100*tr['class_b_share_tracks']:.1f}%** (Class-A: {tr['class_a_tracks']:,}). "
         "Class B is the small-craft transceiver; a high share means the benign model "
         "sees real small-craft motion, the regime a hostile USV would mimic.\n",
         "- Class-B tracks by canonical class: "
         + ", ".join(f"{k}={v:,}" for k, v in
                     sorted(s["class_b_breakdown"].items(), key=lambda x: -x[1])) + "\n",
         "## Per canonical class\n",
         "| class | role | tracks | vessels | med SOG (kn) | med straightness | med loiter |",
         "|---|---|---|---|---|---|---|"]
    roles = {r["canonical_class"]: r for r in s["per_class"]}
    # role lookup from tracks not stored per row here; infer benign/hostile/non_target
    for r in s["per_class"]:
        L.append(f"| {r['canonical_class']} | — | {r['tracks']:,} | {r['vessels']:,} | "
                 f"{r.get('sog_mean_med')} | {r.get('straightness_med')} | "
                 f"{r.get('loiter_med')} |")
    sb = s["self_report_bias"]
    L += ["\n## Carriage / self-report bias (reported honestly)\n",
          f"- {sb['note']}\n",
          f"- Tracks with unmappable/absent ship-type → `unknown_other`: "
          f"**{tr['unknown_class_tracks']:,}**.\n",
          f"- Unmapped VesselType codes (not in taxonomy `ais_ship_type`; folded to "
          f"`unknown_other`, flagged for a taxonomy follow-up): "
          f"{sb['unmapped_vessel_type_codes'] or 'none'}\n",
          "- **range_to_shore** is intentionally NOT computed : it needs "
          "reliable shoreline data + cross-region normalization and otherwise encodes "
          "geography rather than vessel behavior.\n",
          "\n## Figures\n",
          "- `figures/tracks_per_class.png` · `figures/speed_by_class.png` · "
          "`figures/straightness_vs_loiter.png` · `figures/turn_rate_by_class.png`\n"]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AIS ingestion & cleaning .")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--days", nargs="+", help="YYYY-MM-DD subset (default: all found)")
    p.add_argument("--bbox", nargs=4, type=float, default=None,
                   metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--cadence", type=int, default=CADENCE_S)
    p.add_argument("--gap", type=int, default=GAP_S)
    p.add_argument("--no-points", action="store_true",
                   help="skip writing the point-level parquet")
    p.add_argument("--limit-rows", type=int, default=None,
                   help="debug: cap total points after ingest")
    args = p.parse_args(argv)

    mc_dir = args.data_dir / MC_DIR
    zips = sorted(mc_dir.glob("AIS_*.zip"))
    if args.days:
        want = {d.replace("-", "_") for d in args.days}
        zips = [z for z in zips if any(w in z.stem for w in want)]
    if not zips:
        print(f"[error] no MarineCadastre zips under {mc_dir}")
        return 1
    days = [z.stem.replace("AIS_", "").replace("_", "-") for z in zips]
    print(f"[ingest] {len(zips)} day(s): {', '.join(days)}")

    by_code, by_range, roles = load_ais_map(args.data_dir / "taxonomy.yaml")

    t0 = time.time()
    df = read_marinecadastre(zips, bbox=tuple(args.bbox) if args.bbox else None)
    raw_kept = len(df)
    if args.limit_rows:
        df = df.head(args.limit_rows)
    print(f"[ingest] kept {raw_kept:,} points; assigning class & segmenting…")

    static = assign_static_class(df, by_code, by_range, roles)
    # record unmapped codes (present but not in taxonomy)
    present_codes = df["VesselType"].dropna().astype(int)
    vc = present_codes.value_counts()
    unmapped = {int(c): int(n) for c, n in vc.items()
                if code_to_canonical(c, by_code, by_range) is None}

    df, n_teleport = segment_and_thin(df, args.cadence, args.gap)
    points_n = len(df)

    feat = compute_features(df)
    # drop sub-threshold trips
    keep = (feat["n_points"] >= MIN_TRIP_POINTS) & (feat["duration_s"] >= MIN_TRIP_DURATION_S)
    dropped_short = int((~keep).sum())
    feat = feat[keep].reset_index(drop=True)
    # attach class/role/transceiver per mmsi
    feat = feat.merge(static.reset_index().rename(columns={"MMSI": "mmsi"}),
                      on="mmsi", how="left")
    feat["source"] = "marinecadastre"

    # write outputs
    tracks_dir = args.data_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(tracks_dir / "tracks_ais.parquet", index=False)
    if not args.no_points:
        # attach class to points for windowed scoring; keep it compact
        cls = static["canonical_class"].reset_index().rename(columns={"MMSI": "mmsi"})
        pts = df.rename(columns={"MMSI": "mmsi", "LAT": "lat", "LON": "lon",
                                 "SOG": "sog", "COG": "cog", "Heading": "heading"})
        pts = pts.merge(cls, on="mmsi", how="left")
        keep_trips = set(feat["trip_id"].values)
        pts = pts[pts["trip_id"].isin(keep_trips)]
        for c in ("lat", "lon", "sog", "cog", "heading"):
            pts[c] = pts[c].astype("float32")
        pts[["mmsi", "trip_id", "t", "lat", "lon", "sog", "cog", "heading",
             "canonical_class"]].to_parquet(
            tracks_dir / "tracks_ais_points.parquet", index=False)

    meta = {
        "region": "US national (MarineCadastre / NOAA OCM)"
                  + (" [bbox-filtered]" if args.bbox else ""),
        "days": days,
        "raw_rows": raw_kept + n_teleport, # approx pre-clean not tracked; report kept
        "kept_rows": raw_kept, "n_teleport": n_teleport,
        "dropped_short": dropped_short, "unmapped_codes": unmapped,
        "params": {"cadence_s": args.cadence, "gap_s": args.gap,
                   "max_implied_kn": MAX_IMPLIED_SPEED_KN, "loiter_kn": LOITER_SOG_KN,
                   "min_trip_points": MIN_TRIP_POINTS, "min_trip_dur_s": MIN_TRIP_DURATION_S,
                   "bbox": list(args.bbox) if args.bbox else None},
    }
    summary = build_summary(feat, points_n, meta)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        make_figures(feat, args.results_dir / "figures")
    except Exception as e:
        print(f"[warn] figures failed: {e}")
    write_report(summary, args.results_dir / "report.md")

    tr = summary["tracks"]
    print(f"\n[ingest] done in {time.time()-t0:.1f}s")
    print(f" points (cleaned/thinned): {points_n:,}")
    print(f" tracks: {tr['total']:,} over {tr['vessels']:,} vessels "
          f"(dropped {dropped_short:,} short)")
    print(f" Class-B share: {100*tr['class_b_share_tracks']:.1f}% "
          f"(A={tr['class_a_tracks']:,} B={tr['class_b_tracks']:,})")
    print(f" unknown-class tracks: {tr['unknown_class_tracks']:,}; "
          f"unmapped codes: {unmapped or 'none'}")
    print(f"\nWrote:\n {tracks_dir}/tracks_ais.parquet"
          + ("" if args.no_points else f"\n {tracks_dir}/tracks_ais_points.parquet")
          + f"\n {args.results_dir}/report.md + summary.json + figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
