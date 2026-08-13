# Track corpora — data dictionary

Schema + column definitions for the MarineCadastre AIS trajectory Parquet files
(the sole trajectory corpus used in this project).

- **AIS** (cooperative transmitters): `data/tracks/tracks_ais.parquet` (per-track) and
  `data/tracks/tracks_ais_points.parquet` (per-point, local-only). Produced by
  `scripts/data/ingest_ais.py` from MarineCadastre US national dailies
  (2023-06-01 … 2023-06-07).

Licensing / redistribution terms are in [`DATA_LICENSES.md`](DATA_LICENSES.md).
Canonical class taxonomy and per-source label maps are in
[`../data/taxonomy.yaml`](../data/taxonomy.yaml). Track-corpus composition, splits, and
known limitations are in [`../data/DATACARD_TRACKS.md`](../data/DATACARD_TRACKS.md).

---

## Conventions

- **Time (`t`, `t_start`, `t_end`):** int64 **Unix epoch seconds, UTC**.
  `duration_s = t_end − t_start`.
- **Coordinates (`lat`, `lon`, `mean_lat`, `mean_lon`):** decimal degrees, **WGS-84**.
  Longitude in [−180, 180].
- **Speed:** **knots** (AIS) unless a column name says otherwise (`*_px_s` = pixels/s,
  `*_bl_s` = body-lengths/s for video).
- **Angles / course / heading:** **degrees** in [0, 360). Turn rates in **degrees/second**.
  Circular statistics are used for course (so 359° and 1° are treated as adjacent).
- **Nulls (`NaN`):** mean "not computable / not reported," **not** zero. They arise from
  single-point-per-bin trips (no differences to compute), AIS "not available" sentinels
  (e.g. SOG 102.3), or missing Class-B heading. Consumers must mask, not fill with 0.
- **Identity keys:** AIS = (`mmsi`, `trip_id`). The per-track file joins to the per-point
  file 1-to-many on `trip_id`.

---

## 1. `tracks/tracks_ais.parquet` — one row per AIS trip-level track

153,811 rows × 29 cols. A "track" is one vessel's voyage segment, cut wherever the
transmission gap exceeds 30 min. The lat/lon time series is **collapsed** into the
summary features below (use the points file for the full path).

### Identity & provenance
| column | dtype | units | definition |
|---|---|---|---|
| `trip_id` | int64 | — | Unique track id; join key to the points file. |
| `mmsi` | int64 | — | Maritime Mobile Service Identity (vessel id). **`0` = no valid MMSI reported** — not a usable identity (see caveats). |
| `vessel_type_code` | float32 | AIS code | Per-MMSI **modal** self-reported numeric AIS ship-type (0–255). Null if never reported. |
| `transceiver_class` | category | — | `A` (SOLAS/large) or `B` (small-craft). |
| `canonical_class` | string | — | Project class, mapped from `vessel_type_code` via `taxonomy.yaml`. See enum below. |
| `role` | string | — | `benign` / `hostile` / `non_target` (from taxonomy; `military`→hostile, unknown→non_target). |
| `source` | string | — | Dataset origin, `marinecadastre`. |

### Extent
| column | dtype | units | definition |
|---|---|---|---|
| `n_points` | int64 | count | Cleaned/thinned positions in the track (min 5). |
| `t_start`, `t_end` | int64 | epoch s, UTC | First / last position time. |
| `duration_s` | int64 | seconds | `t_end − t_start` (min 300 s = 5 min). |
| `mean_lat`, `mean_lon` | float32 | deg (WGS-84) | Track **centroid only** (not the path). Coarse by design — no range-to-shore is computed (geography ≠ behavior). |

### Speed
| column | dtype | units | definition |
|---|---|---|---|
| `sog_mean`, `sog_med` | float32 | knots | Mean / median reported Speed Over Ground. |
| `sog_p95` | float64 | knots | 95th-percentile SOG (typical "fast" speed, robust to outliers). |
| `sog_max` | float32 | knots | Peak SOG (≤ 102.2; 102.3 is the AIS "not available" sentinel and is dropped). |
| `sog_std` | float32 | knots | SOG variability (steady transit ≈ low). |
| `pos_speed_mean_kn` | float64 | knots | Speed **recomputed from position/time** (great-circle). Independent cross-check vs `sog_mean`; **noisy** — can be hugely inflated by GPS jitter over small `dt` (see caveats). |

### Loiter
| column | dtype | units | definition |
|---|---|---|---|
| `loiter_frac` | float64 | [0,1] | Fraction of points with SOG < 0.5 kn (stopped/station-keeping). ~1 = drifting/anchored; ~0 = always underway. |

### Path shape
| column | dtype | units | definition |
|---|---|---|---|
| `path_km` | float64 | km | Total distance travelled (sum of great-circle steps; the "odometer"). |
| `net_km` | float32 | km | Straight-line start→end distance. |
| `straightness` | float64 | [0,1] | `net_km / path_km`. **1 = straight-line transit**; ~0 = looping/loitering (went nowhere net). Null when `path_km`≈0. |

### Turning / course stability
| column | dtype | units | definition |
|---|---|---|---|
| `turn_rate_mean_dps` | float64 | deg/s | Mean absolute rate of change of Course Over Ground. Null for tracks with no course changes to measure (~26.9k). |
| `turn_rate_p95_dps` | float64 | deg/s | 95th-percentile turn rate (sharpness of harder turns). |
| `cog_circ_std_deg` | float32 | deg | **Circular** standard deviation of COG. Low = holds one course; high = course scattered. Unbounded-ish as heading randomises. |

### Acceleration
| column | dtype | units | definition |
|---|---|---|---|
| `accel_mean_abs` | float64 | kn/min | Mean absolute acceleration between consecutive points. |
| `accel_std` | float64 | kn/min | Variability of that acceleration. |

### Data-quality context
| column | dtype | units | definition |
|---|---|---|---|
| `heading_avail_frac` | float64 | [0,1] | Fraction of points with a valid **heading** (bow direction). Often low on Class-B → heading-derived features unreliable for those tracks. |

---

## 2. `tracks/tracks_ais_points.parquet` — one row per cleaned AIS position

60,345,228 rows × 9 cols. The raw (cleaned + thinned) path. To reconstruct a vessel's
trajectory, filter by `trip_id` and sort by `t`.

| column | dtype | units | definition |
|---|---|---|---|
| `mmsi` | int64 | — | Vessel id (`0` = missing; see caveats). |
| `trip_id` | int64 | — | Track id; join key to `tracks/tracks_ais.parquet`. |
| `t` | int64 | epoch s, UTC | Position timestamp. Thinned to ~60 s cadence (one point per (MMSI, 60 s) bin). |
| `lat`, `lon` | float32 | deg (WGS-84) | Position. |
| `sog` | float32 | knots | Reported Speed Over Ground. Null = "not available" sentinel dropped (~160k). |
| `cog` | float32 | deg [0,360) | Reported Course Over Ground. **Null ~10.2M (17%)**. |
| `heading` | float32 | deg [0,360) | Reported bow heading. **Null ~33.4M (55%)** — mostly Class-B. |
| `canonical_class` | string | — | Track's class (denormalised from the per-track file for convenience). |

### AIS caveats (for release)
- **`mmsi = 0`** means the position messages carried no valid MMSI; those trips are still
  segmented but are **not identity-resolved**. Filter them out if you need vessel identity.
- **Self-reported bias:** `vessel_type_code`/`canonical_class` are what the operator
  typed into the transponder — not verified. `unknown_other` (15,723 tracks) is dominated
  by unmapped/spare codes (esp. code 57). AIS has **no `small_craft` code** — small craft
  self-report as recreational/fishing/unknown.
- **Carriage bias:** AIS only sees **cooperative transmitters**. Non-transmitting craft
  are absent by construction — an explicit limitation of an AIS-only trajectory corpus.
- `pos_speed_mean_kn` is a diagnostic cross-check, **not** a clean speed feature (max
  observed ~11,000 kn from position jitter over tiny `dt`). Prefer `sog_*`.
- Cleaning parameters (baked in): teleport removal at implied speed > 80 kn; 60 s cadence
  thinning; 30 min gap → new trip; loiter threshold 0.5 kn; SOG valid ≤ 102.2 kn.

---

## Enumerations

**`canonical_class`** (full set in `taxonomy.yaml`). Observed in AIS —
`recreational`, `working_service`, `unknown_other`, `sailing`, `cargo_merchant`,
`passenger_ferry`, `fishing`, `military`.

**`role`:** `benign`, `hostile` (AIS `military` only), `non_target` (buoys/unknown).

**`transceiver_class`** (AIS): `A`, `B`.
