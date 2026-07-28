# Vessel trajectory dataset — data card

Derived **vessel trajectory (track) features** for maritime behavior modeling and
class-conditional motion analysis: cleaned AIS voyage tracks with self-reported vessel
type, plus a small set of shore-camera video-derived tracks of non-transmitting craft.
Each track is reduced to per-track kinematic features (with per-point series available).

This repository **does not re-host** bulk AIS feeds or the source video. Obtain each
source from its provider (see `docs/DATA_LICENSES.md`) and regenerate the derived
features with the scripts in `scripts/`. Per-column schema (dtype, units, caveats) is in
`docs/DATA_DICTIONARY.md`. Checksums are in `CHECKSUMS.derived.sha256`; the version stamp
is `RELEASE.json`.

**Release policy.** Redistributed products are **derived track features + code**, never
bulk raw AIS. Source terms govern (see `docs/DATA_LICENSES.md`). Any **Global Fishing
Watch**–derived feature is **CC BY-NC 4.0** (non-commercial, attribution) and must be
separable/omittable from a permissive release.

---

## 1. Summary

| Field | Value |
|---|---|
| **Content** | Per-track kinematic features (+ per-point series) for maritime vessels |
| **Format** | Parquet under `tracks/` (`tracks_ais.parquet`, `tracks_video.parquet`, + `_points`) |
| **Tracks** | AIS **153,811** (30,053 vessels); video **329** (36 clips) |
| **Labels** | Vessel class per track (AIS self-report; video via annotation transfer) |
| **Maintainer** | Academic research (AMSL / Duke) |

---

## 2. Composition

| Corpus | Tracks | Vessels / clips | Frame | Class source |
|---|---:|---:|---|---|
| MarineCadastre AIS (US, 2023-06-01…07) | 153,811 | 30,053 vessels | Metric (WGS-84 / knots) | Self-reported AIS ship-type |
| SMD video-derived (on-shore) | 329 | 36 clips | **Image-plane, scale-normalized** | Annotation transfer (SMD-Plus) |

- **AIS** — one row per voyage segment (split at >30 min transmission gaps), summarizing a
  cleaned, thinned lat/lon time series into kinematic features. **Class-B (small-craft)
  share = 59.1%** of tracks in this window. Class labels come from the self-reported
  numeric AIS ship type mapped via `taxonomy.yaml` (`ais_ship_type`).
- **SMD video-derived** — shore-camera tracks of (generally non-transmitting) craft,
  used as an independent, differently-sampled comparison set. Identity from the
  dataset's track ground truth; class transferred from SMD-Plus boxes by framewise IoU.

Per-column definitions and units: `docs/DATA_DICTIONARY.md`.

---

## 3. Splits

| Partition | Rule | Notes |
|---|---|---|
| AIS tracks (`ais_track_splits.csv`) | **Vessel-disjoint** by MMSI | `mmsi==0` (no identity) is train-only; region/day tags allow region/time holdouts |
| SMD video (`video_eval_pool.csv`) | Separate held-out pool (grouped by clip) | Not pooled with AIS (different time-horizon/sampling) |

Leakage check **PASS** (0 vessels spanning AIS splits). Details: `results/splits/report.md`.

---

## 4. Collection & processing

| Step | Script / artifact |
|---|---|
| Acquire sources | `scripts/data/fetch_data.py` |
| Ingest AIS | `scripts/data/ingest_ais.py` → `tracks/tracks_ais*.parquet` |
| Ingest video tracks | `scripts/data/ingest_smd_tracks.py` → `tracks/tracks_video*.parquet` |
| Splits | `scripts/data/build_splits.py` → `splits/` |
| Package / checksum | `scripts/data/package_derived.py` |

AIS cleaning: coordinate/speed/course/heading validity filters, `(MMSI, t)` dedup,
teleport removal (>80 kn implied), cadence thinning (~1 min → 60 s), and >30 min-gap
trip segmentation, followed by per-track feature extraction.

---

## 5. Uses

| Use | Recommended data |
|---|---|
| Class-conditional behavior modeling | AIS tracks (optionally filter by `role`/class) |
| Vessel-type classification from motion | AIS tracks, vessel-disjoint splits |
| Robustness to sampling / non-transmitting craft | SMD video pool (image-plane features) |
| Region/time-holdout generalization | AIS splits sliced by the region/day tags |

**Frame compatibility.** AIS features are metric (knots, deg/s); SMD video features are
**image-plane, scale-normalized** (e.g. body-lengths/s). Compare the two only at the
feature-distribution level — do not treat SMD as a metric-scale validation of AIS.

---

## 6. Distribution

| Item | Status |
|---|---|
| Raw AIS feeds / source video | **Not redistributed** — obtain from providers |
| Derived track features (parquet) | Released (checksummed); source terms apply |
| GFW-derived features (if added) | **CC BY-NC 4.0** — separable/omittable |
| License summary | `docs/DATA_LICENSES.md` |

**Versioning.** `RELEASE.json` + `CHECKSUMS.derived.sha256`; regenerate after any
pipeline change. Prefer regenerating derived artifacts over hand-editing parquet.

---

## 7. Known limitations

1. **AIS carriage / self-report bias** — AIS covers only vessels that carry and transmit
   a transponder and report a usable ship type. Small/recreational and non-transmitting
   craft are under-represented, and ship type is self-reported (and sometimes wrong).
   Class-B coverage is strong in this window (59.1%) but not universal.
2. **Video tracks are image-plane only** — the source video lacks the camera calibration
   (intrinsics / height / ground control) needed for a defensible world-plane transform,
   so only scale-normalized image-plane features are provided. They do **not** validate
   AIS metric speed/turn-rate.
3. **Small, region-specific video set** — 329 tracks from one dataset's on-shore clips;
   not a general non-cooperative benchmark.
4. **Temporal/geographic scope** — the AIS window is one week of US coastal data; behavior
   envelopes may not transfer to other regions/seasons without re-ingestion.
5. **Class imbalance & unknowns** — some classes are thin and a fraction of AIS tracks
   have unknown/unmapped ship types (folded to an `unknown_other` bucket).
6. **GFW is non-commercial** — if Global Fishing Watch features are added, the release
   inherits CC BY-NC 4.0 for those columns.

---

## 8. Citation

Cite each upstream source per `docs/DATA_LICENSES.md` (MarineCadastre / NOAA OCM;
Danish Maritime Authority and Global Fishing Watch if used; SMD for the video tracks)
and this dataset when the derived track features are used.
