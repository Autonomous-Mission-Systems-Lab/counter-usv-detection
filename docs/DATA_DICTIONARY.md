# Track corpora — data dictionary

Schema + column definitions for the four trajectory Parquet files produced for this project
of the counter-USV project. Written for a data consumer (including a future public
release): every column is listed with dtype, units, definition, and release caveats.

- **AIS** (cooperative transmitters): `data/tracks/tracks_ais.parquet` (per-track) and
  `data/tracks/tracks_ais_points.parquet` (per-point). Produced by `scripts/data/ingest_ais.py`
  from MarineCadastre US national dailies (2023-06-01 … 2023-06-07).
- **SMD video** (non-cooperative, shore-camera): `data/tracks/tracks_video.parquet`
  (per-track) and `data/tracks/tracks_video_points.parquet` (per-frame). Produced by
  `scripts/data/ingest_smd_tracks.py` from the Singapore Maritime Dataset on-shore clips.

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
- **Identity keys:** AIS = (`mmsi`, `trip_id`); video = `track_id` (= `clip#obj_index`).
  In both, the per-track file joins to the per-point file 1-to-many on `trip_id` /
  `track_id`.

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
  are absent by construction — this is exactly what the SMD video corpus complements.
- `pos_speed_mean_kn` is a diagnostic cross-check, **not** a clean speed feature (max
  observed ~11,000 kn from position jitter over tiny `dt`). Prefer `sog_*`.
- Cleaning parameters (baked in): teleport removal at implied speed > 80 kn; 60 s cadence
  thinning; 30 min gap → new trip; loiter threshold 0.5 kn; SOG valid ≤ 102.2 kn.

---

## 3. `tracks/tracks_video.parquet` — one row per SMD video track

329 rows × 27 cols. Shore-camera tracks of **non-cooperative** craft. **All motion
features are image-plane and scale-normalized — NON-metric** (the world-frame
calibration gate failed: no intrinsics / camera height / GCPs). Do **not** compare these
numbers to the AIS metric speed/turn-rate envelopes.

### Identity & class transfer
| column | dtype | units | definition |
|---|---|---|---|
| `track_id` | string | — | `clip#obj_index`; join key to the points file. |
| `clip` | string | — | Source SMD video (e.g. `MVI_1469_VIS`). |
| `obj_index` | int64 | — | Object index within the clip's `TrackGT`. |
| `canonical_class` | string | — | Class transferred from SMD-Plus `ObjectGT` by framewise IoU (else fallback). `cargo_merchant` here = SMD-Plus coarse generic "Vessel/ship" (large vessels, **not verified cargo**). |
| `role` | string | — | `benign` / `non_target` (from taxonomy). |
| `label_source` | string | — | `smdplus_iou` (95%) or `trackgt_fallback` (unmatched → track's own original-SMD label). |
| `label_match_frac` | float64 | [0,1] | Fraction of the track's frames that IoU-matched an SMD-Plus box. |

### Extent & sampling
| column | dtype | units | definition |
|---|---|---|---|
| `n_frames` | int64 | count | Frames the object is present (min 15). |
| `duration_s` | float64 | seconds | Track span (~7–20 s; SMD clips are short). |
| `fps` | float64 | Hz | Source video frame rate (30). |
| `feature_hz` | float64 | Hz | Cadence kinematics were computed at (**3 Hz** resample; raw 30 fps gives sub-pixel noise for distant objects). |

### Image-plane motion (scale-normalized, NON-metric)
| column | dtype | units | definition |
|---|---|---|---|
| `norm_speed_bl_s_mean` | float64 | body-lengths/s | Centroid speed ÷ bbox diagonal — removes distance/zoom scale. Mean over the track. |
| `norm_speed_bl_s_p95`, `norm_speed_bl_s_max` | float64 | body-lengths/s | 95th-pct / peak normalized speed (large values = birds/planes/fast small craft). |
| `speed_px_s_mean` | float64 | pixels/s | Raw (un-normalized) centroid speed; scale-dependent, provided for reference. |
| `loiter_frac` | float64 | [0,1] | Fraction of steps with normalized speed < 0.1 body-lengths/s. |
| `turn_rate_mean_dps`, `turn_rate_p95_dps` | float64 | deg/s | Rate of change of the image-plane **velocity direction**. Biased by perspective; qualitative across classes. |
| `straightness` | float64 | [0,1] | `net / path` of the pixel centroid trajectory. Buoys ≈ 0 (bob in place); transit ≈ 1. Null when path≈0. |

### Shape & appearance
| column | dtype | units | definition |
|---|---|---|---|
| `diag_px_mean` | float64 | pixels | Mean bbox diagonal (apparent size; conflates true size with range). |
| `aspect_mean`, `aspect_std` | float64 | ratio | Mean / std of bbox width÷height over the track. |
| `size_trend_per_s` | float64 | 1/s | Relative slope of bbox diagonal over time; >0 approaching, <0 receding (image-apparent). |

### Provided SMD annotations & ordinal geometry
| column | dtype | units | definition |
|---|---|---|---|
| `moving_frac_annotated` | float64 | [0,1] | Fraction of frames SMD annotated `MotionType == Moving`. |
| `distance_modal` | string | — | Modal SMD `DistanceType` (`Near`/`Far`). **Raw annotation** — also contains sentinels (`-1`, `0`, `Other`, `[]`) for unlabeled. |
| `horizon_rel_y_px` | float64 | pixels | Mean vertical offset of centroid from the `HorizonGT` line. **ORDINAL pseudo-range only** (farther objects sit nearer the horizon) — **non-metric**; null for the 3 clips without HorizonGT. |
| `source` | string | — | `smd_video`. |

---

## 4. `tracks/tracks_video_points.parquet` — one row per frame per video track

163,319 rows × 9 cols. Raw image-plane centroids at full 30 fps (no derived kinematics;
resample yourself if needed).

| column | dtype | units | definition |
|---|---|---|---|
| `track_id` | string | — | Join key to `tracks/tracks_video.parquet`. |
| `clip` | string | — | Source video. |
| `frame` | int64 | frame idx | 0-based frame number. |
| `t_s` | float64 | seconds | `frame / fps`. |
| `cx`, `cy` | float64 | pixels | Bbox center. **Can be negative / off-frame** (annotations extend past the image edge). |
| `w`, `h` | float64 | pixels | Bbox width / height. Sentinel `−1` = degenerate/edge box. |
| `canonical_class` | string | — | Track class (denormalised). |

### Video caveats (for release)
- **Not metric and not pooled with AIS.** Time-horizon/sampling mismatch (SMD ~7–20 s
  @30 fps vs AIS minutes–hours @60 s) → no shared observation window. SMD is a held-out
  non-cooperative sensitivity pool, compared to AIS only at the short-horizon feature-
  **distribution** level.
- Perspective foreshortening biases image-plane straightness/turn-rate — treat
  cross-class comparisons qualitatively.
- Small corpus (329 tracks, ~10 min of footage): a sensitivity check + PercepGuard-style
  baseline input, not a second training corpus.

---

## Enumerations

**`canonical_class`** (full set in `taxonomy.yaml`). Observed —
- AIS: `recreational`, `working_service`, `unknown_other`, `sailing`, `cargo_merchant`,
  `passenger_ferry`, `fishing`, `military`.
- Video: `cargo_merchant`, `unknown_other`, `small_craft`, `passenger_ferry`,
  `static_aid`, `sailing`.

**`role`:** `benign`, `hostile` (AIS `military` only), `non_target` (buoys/unknown).

**`transceiver_class`** (AIS): `A`, `B`.
