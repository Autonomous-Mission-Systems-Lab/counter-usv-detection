# Vessel trajectory dataset — data card

Derived **vessel trajectory (track) features** for maritime behavior modeling and
class-conditional motion analysis: cleaned AIS voyage tracks with self-reported vessel
type from **MarineCadastre** (US).

Point-level AIS time series (`tracks_ais_points.parquet`) are kept **local only** for
re-derivation and are not part of the public derived release.

This repository **does not re-host** bulk AIS feeds. Obtain MarineCadastre from NOAA OCM
(see `docs/DATA_LICENSES.md`) and regenerate derived features with
`scripts/data/ingest_ais.py`. Per-column schema (dtype, units, caveats) is in
`docs/DATA_DICTIONARY.md`. Checksums are in `CHECKSUMS.derived.sha256`; the version stamp
is `RELEASE.json`.

**Release policy.** Redistributed products are **derived trip-level track features +
code**, never bulk raw AIS or point-level AIS feeds. The Zenodo data deposit additionally
**drops the `mmsi` column** from released trip/split/manifest tables (keeps `trip_id`).

---

## 1. Summary

| Field | Value |
|---|---|
| **Content** | Per-track kinematic features for maritime vessels |
| **Format** | Parquet under `tracks/` (`tracks_ais.parquet`) |
| **Tracks** | AIS **153,811** (30,053 vessels) |
| **Labels** | Vessel class per track (AIS self-report) |
| **Maintainer** | Academic research (AMSL / Duke) |

---

## 2. Composition

| Corpus | Tracks | Vessels | Frame | Class source |
|---|---:|---:|---|---|
| MarineCadastre AIS (US, 2023-06-01…07) | 153,811 | 30,053 vessels | Metric (WGS-84 / knots) | Self-reported AIS ship-type |

- **AIS** — one row per voyage segment (split at >30 min transmission gaps), summarizing a
  cleaned, thinned lat/lon time series into kinematic features. **Class-B (small-craft)
  share = 59.1%** of tracks in this window. Class labels come from the self-reported
  numeric AIS ship type mapped via `taxonomy.yaml` (`ais_ship_type`).

Per-column definitions and units: `docs/DATA_DICTIONARY.md`.

---

## 3. Splits

| Partition | Rule | Notes |
|---|---|---|
| AIS tracks (`ais_track_splits.csv`) | **Vessel-disjoint** by MMSI (local); public deposit drops `mmsi` | `mmsi==0` (no identity) is train-only; region/day tags allow region/time holdouts |

Leakage check **PASS** (0 vessels spanning AIS splits). Details: `results/splits/report.md`.

---

## 4. Collection & processing

| Step | Script / artifact |
|---|---|
| Acquire MarineCadastre | `scripts/data/fetch_data.py --source marinecadastre_ais` |
| Ingest AIS | `scripts/data/ingest_ais.py` → `tracks/tracks_ais*.parquet` |
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
| Region/time-holdout generalization | AIS splits sliced by the region/day tags |

---

## 6. Distribution

| Item | Status |
|---|---|
| Raw AIS feeds | **Not redistributed** — obtain from NOAA OCM |
| Point-level AIS (`tracks_ais_points.parquet`) | **Local only** — re-derive via `ingest_ais.py` |
| Derived trip-level track features | Released (checksummed); public deposit drops `mmsi` |
| License summary | `docs/DATA_LICENSES.md` |

**Versioning.** `RELEASE.json` + `CHECKSUMS.derived.sha256`; regenerate after any
pipeline change. Prefer regenerating derived artifacts over hand-editing parquet.

---

## 7. Known limitations

1. **AIS carriage / self-report bias** — AIS covers only vessels that carry and transmit
   a transponder and report a usable ship type. Small/recreational and non-transmitting
   craft are under-represented, and ship type is self-reported (and sometimes wrong).
   Class-B coverage is strong in this window (59.1%) but not universal.
2. **Temporal/geographic scope** — the AIS window is one week of US coastal data; behavior
   envelopes may not transfer to other regions/seasons without re-ingestion.
3. **Class imbalance & unknowns** — some classes are thin and a fraction of AIS tracks
   have unknown/unmapped ship types (folded to an `unknown_other` bucket).

---

## 8. Citation

Cite MarineCadastre / NOAA OCM per `docs/DATA_LICENSES.md` and this dataset when the
derived track features are used.
