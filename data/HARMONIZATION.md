# Data-harmonization spec (the train-time contract)

The single source of truth for how the curated data (COCO master + track corpora) is
turned into model inputs. Everything here is a **load-time / train-time** contract:
nothing in this document rewrites pixels or annotations on disk. The COCO master
(`data/annotations/coco_master.json`) and the track Parquets stay in their native,
provenance-preserving form; loaders apply the transforms below.

Companion docs: `data/INVENTORY.md` (sources/counts), `data/taxonomy.yaml` (classes),
`docs/DATA_DICTIONARY.md` (track columns), `results/eo_audit/report.md` (EO audit),
`configs/base.yaml` (the machine-readable knobs that mirror this contract).

Status: **frozen for detector training** (EO) and the benign-behavior model (tracks). Changes require a version bump
here + a note in the affected run config.

---

## Part A — EO imagery

### A.1 Resolution policy (native-on-disk, letterbox-at-load)

- **Retain native resolution on disk.** No source is pre-resized to a common native
  resolution, and nothing is downscaled to the smallest source. The already-downscaled
  SeaShips Roboflow export was explicitly replaced by the native 1920×1080 VOC release;
  do not reintroduce pre-scaled sources.
- **At load, letterbox to a fixed network input size**, preserving aspect ratio with
  padding. Primary input **640**; **1280** is a first-class alternate evaluated for
  small-target recall (do not treat 1280 as an afterthought — the small-target sources
  below only clear the detector floor at the larger input).
- Native sizes span a wide range, so letterbox (not stretch, not crop) is mandatory to
  avoid aspect distortion:

  | source | native size(s) | notes |
  |---|---|---|
  | seaships | 1920×1080 (all) | shore surveillance; large targets (min-side p50 124px) |
  | smd | 1920×1080 (all) | shore video frames; small targets (p50 49px) |
  | aboships | 1280×720 (all) | onboard/moving-vessel; **smallest** targets (p50 27px, p05 6px) |
  | mcships | ~500×(≈332–500), variable | web chips; frame-filling (area-frac median 0.26) |
  | usv | mixed (mostly 1280×720; also 1000×667, 1024×576, 1920×1080, …) | curated usv set |

### A.2 Letterbox transform (exact)

For an image of native size `(W, H)` to a square input `S ∈ {640, 1280}`:

```
s = S / max(W, H) # single isotropic scale (long side fits)
new_w = round(W * s)
new_h = round(H * s)
pad_x = (S - new_w) / 2 # centered padding (symmetric)
pad_y = (S - new_h) / 2
pad value = 114 on each RGB channel # neutral grey (YOLO convention)
```

Resize by `s` (bilinear), then pad to `S×S` with 114. Centered padding is the contract;
if a detector framework defaults to top-left padding, that is acceptable **only if the
same offsets are used for the boxes** — but centered is preferred for consistency across
families.

### A.3 Box remapping (native coords → input coords)

Boxes live in **native pixel coordinates** in the COCO master and are **never rewritten on
disk**. Every box is remapped with the *same* transform as its image, at load:

```
[x, y, w, h]_input = [x*s + pad_x, y*s + pad_y, w*s, h*s]
```

Multi-scale and mosaic (A.6) apply their own further remaps on top, at train time only.
Clip remapped boxes to `[0, S]`; drop any box whose remapped shortest side falls below the
eligibility floor (A.4) for the current input size (this is why the floor is reported per
input size in the audit).

### A.4 Pixels-on-target eligibility floors (LOCKED here)

Non-destructive flags from the EO audit (`scripts/data/audit_eo.py`). Nothing is deleted from
the master; these define which annotations count as *eligible* for a given purpose. Floors
are on the **native shortest bbox side** (pre-letterbox); the audit also reports the
post-letterbox shortest side per input size.

- **Detector-eligible — shortest side ≥ 8 px (LOCKED).** Below 8px the box is
  noise-dominated and detection is unreliable. Retains **67,397 / 71,501 = 94.3%** of the
  master. (ABOShips-PLUS used a stricter 16px filter; we keep more and report the sweep so
  the choice is auditable.)
- **Patch-eligible — shortest side ≥ 32 px (LOCKED).** A target may be detectable yet too
  small to carry a bounded, marine-EOT-surviving adversarial patch. Stricter floor, used
  **only** for adversarial-patch evaluation. Retains **43,761 / 71,501 = 61.2%**.
- **Report the sweep, don't hard-code silently.** Sensitivity is reported at det
  {4,8,16,24,32} and patch {16,24,32,48,64} so reviewers see the floor's effect. The
  headline uses 8 / 32.

Consequence by source (detector floor @8px, native): the small-target sources
(**ABOShips** p05=6px, **SMD** p05=14.5px) lose the most at the floor and lose more again
after letterbox to 640 — this is the quantitative reason 1280 is evaluated. Large-target
sources (SeaShips, McShips, USV) are essentially unaffected.

### A.5 Source-specific burned-in overlay (SeaShips)

SeaShips frames carry a **burned-in surveillance overlay** (timestamp band along the top,
camera-ID band along the bottom) — a source-specific cue a detector could exploit as a
shortcut. The EO audit shows GT boxes intersect the text bands in only **~0.02%** of
cases, so:

- **Mask** the two constant text bands with a neutral fill (114, matching the letterbox
  pad) as a **non-destructive, fixed-region** load-time op. **No crop, no box remap** — the
  image geometry and all boxes are unchanged; only the text pixels are overwritten.
- Applied to SeaShips **only**, to every SeaShips image, in both train and eval, so the
  masking is not itself a train/eval shortcut.
- **Locked band coordinates** (native 1920×1080, half-open row intervals; scaled
  proportionally at other sizes), measured 2026-07-16 from a 30-image aggregate of
  text-edge / extreme-luminance score and recorded in `configs/base.yaml` +
  `src/counterusv/data/overlay.py`:

  | band | rows `[y0, y1)` | content |
  |---|---|---|
  | top | `[0, 110)` | timestamp |
  | bottom | `[980, 1080)` | camera ID |

  Fill value = 114 (matches the letterbox pad). Because <0.02% of boxes touch the
  bands, any box overlap is left as-is (masking wins; the rare clipped target is
  accepted and noted).
- Other sources have no equivalent constant overlay; do not mask them.

### A.6 Augmentation policy

- **Train:** multi-scale + mosaic enabled (`train.multiscale: true`, `train.mosaic: true`),
  plus the detector family's standard photometric/geometric augments. Mosaic composes 4
  letterboxed tiles; boxes are remapped per tile (A.3) and re-clipped.
- **Per-channel normalization:** scale to `[0,1]`; if the backbone is ImageNet-pretrained,
  apply mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`, otherwise `[0,1]`. Record
  which was used per run.
- **Augmentation OFF for:**
  - **Clean evaluation** (letterbox only — the mAP number must reflect the data, not the
    augmentation pipeline).
  - **Adversarial-patch evaluation** — the patch is optimized/applied against the
    letterboxed image with augmentation disabled, so marine-EOT robustness (`docs/METRICS.md`)
    is measured by the *explicit* EOT transform sweep, not confounded by training augments.

### A.7 Channel firewall + viewpoint (carried from usv curation / splits)

- **USV set is EO/appearance-only.** `source == "usv"` / `channel == "eo_only"` images
  train and test the **detector only**; they never enter the kinematic scorer, and hostile
  trajectories stay synthesized at eval. Enforced by the `usv` source tag; loaders for the
  scorer hard-exclude it.
- **Viewpoint policy (decided 2026-07-10, enforced in the split):** ABOShips (onboard/moving-
  vessel viewpoint) is **train-eligible** for small-craft appearance diversity, but the
  **headline operational eval is shore-only** (`operational_viewpoint=True`: seaships, smd).
  ABOShips is a separate cross-viewpoint transfer stratum, firewalled from every eval split,
  with a with/without-ABOShips training ablation. See `configs/base.yaml`
  (`eval_operational_sources` / `train_auxiliary_sources`). Split leakage control (sequence
  ids + dedup groups) is specified in the split.

---

## Part B — Track kinematics

The benign-behavior model is trained on **real benign tracks only**; hostile /
adaptive trajectories are synthesized at eval and never enter scorer training
(`docs/METRICS.md`). This part fixes the feature contract so AIS and video features are
defined identically where comparable and kept apart where not.

### B.1 Unit & representation conventions (from `docs/DATA_DICTIONARY.md`)

- **Time:** int64 Unix epoch seconds, UTC. `duration_s = t_end − t_start`.
- **Coordinates:** decimal degrees, WGS-84; longitude in [−180, 180].
- **Speed:** **knots** for AIS (metric, world-frame). Video speeds are **body-lengths/s**
  (`*_bl_s`, scale-normalized) or pixels/s (`*_px_s`) — **non-metric**.
- **Angles / course / heading:** degrees in [0, 360); **circular statistics** for course
  (359° and 1° are adjacent). Turn rates in degrees/second.
- **Nulls (`NaN`) mean "not computable / not reported," never zero.** They arise from
  single-point bins, AIS "not available" sentinels (e.g. SOG 102.3), or missing Class-B
  heading. **Consumers must mask, not fill with 0.**
- **Identity keys:** AIS = (`mmsi`, `trip_id`); video = `track_id` (`clip#obj_index`).

### B.2 Resampling cadence & observation window

- **AIS:** native ~1 min; cleaned to a 60 s cadence (one point per (MMSI, 60 s) bin);
  trips cut at >30 min gaps. The scorer resamples to `kinematics.resample_seconds = 30`.
- **SMD video:** raw 30 fps → **3 Hz** feature resample (raw fps gives sub-pixel jitter for
  distant objects). Reported as `feature_hz`.
- **Observation window:** `kinematics.observation_window_s = 300` s of history is the
  default for a consistency decision, **swept** in eval for the time-to-flag vs. false-alarm
  curve. NB: 300 s exceeds every SMD clip (~7–20 s), so SMD informs only the **short end** of
  that curve.

### B.3 World-frame calibration gate (metric vs. image-plane)

The gate separates world-frame, AIS-comparable features from image-plane-only features.

- **AIS features are metric / world-frame** (knots, km, deg/s) → directly usable by the
  scorer and for FAR on held-out AIS tracks.
- **SMD video FAILED the world-frame calibration gate** (`world_frame_calibratable=False`:
  no intrinsics / camera height / ground-control points → no defensible water-plane
  homography). Therefore SMD emits **image-plane, scale-normalized (non-metric)** features
  only; `horizon_rel_y_px` is an **ordinal pseudo-range**, flagged non-metric.
- **Do not pool SMD with AIS**, and do not apply the AIS metric scorer to SMD coordinates.
  SMD is a **held-out non-cooperative sensitivity pool** (and PercepGuard-style baseline
  input), compared to the AIS envelope only at the **short-horizon feature-distribution**
  level — never as a metric speed/turn-rate validation. FAR on video is reported separately
  and only for features that pass the gate; since none do metrically, AIS carriage bias is
  retained as an explicit unresolved limitation (`docs/METRICS.md`).

### B.4 Scorer feature set

Canonical benign-behavior features are frozen in
`configs/defense/scorer_features.yaml` (`configs/base.yaml → kinematics.feature_spec`
points here; no mirrored feature list). The model fit uses a deduplicated
core/course subset in `configs/defense/behavior_model.yaml → features`. All
contract features are derivable from `tracks/tracks_ais.parquet` (see
`docs/DATA_DICTIONARY.md` for exact columns):

**Core (always used; NaN → mask, never fill 0):**
- `speed_mean/med/p95/max/std` (← `sog_*`, kn)
- `loiter_frac` (fraction SOG < 0.5 kn, [0,1])
- `straightness` (`net_km / path_km`, [0,1])
- `accel_mean_abs` / `accel_std` (kn/min)

**Course (COG-derived; gated on non-null — ~15–20% null when no course change):**
- `turn_rate` / `turn_rate_p95` (← `turn_rate_*_dps`, deg/s)
- `cog_circ_std` (← `cog_circ_std_deg`, deg)

**Excluded:**
- `range_to_shore` — **not computed** (geography ≠ behavior; no shoreline +
  cross-region normalization). Do not silently substitute the track centroid.
- `pos_speed_mean_kn` — GPS-jitter inflated over small `dt`.
- Heading-derived features — Class-B heading largely unavailable
  (`heading_avail_frac` retained as a quality flag only).

Class-conditional modeling excludes `benign_unspecified` (coarse-only, no
kinematics), `military`/`usv` (hostile — synthesized at eval), and
`static_aid`/`unknown_other` (non-target). The one-class scorer trains on
benign classes with real envelopes only. The training corpus is the AIS train
split ∩ `role==benign` (`data/behavior/benign_train_manifest.parquet`, built by
`scripts/behavior/build_benign_corpus.py`).

### B.5 Cross-source feature alignment

Where a feature is compared across AIS and video, it must be computed with the **same
definition and resample cadence** so the scorer is not overfit to AIS sampling. Because the
metric axes are incomparable (B.3), cross-source comparison is at the **distribution /
shape** level (straightness, loiter fraction, normalized speed rank) and over the **short
horizon** only. Any comparison longer than the shortest SMD clip is not defensible and must
not be reported.

---

## Change log
- 2026-07-22 — Dropped mirrored `kinematics.features` list from `configs/base.yaml`;
  contract lives only in `scorer_features.yaml` (`feature_spec` pointer retained).
- 2026-07-19 — Scorer feature contract frozen in
  `configs/defense/scorer_features.yaml` (expanded core speed/accel suite +
  COG-gated course features; benign train corpus =
  `data/behavior/benign_train_manifest.parquet`).
- 2026-07-15 — Initial spec. Locks detector floor = 8px, patch floor = 32px;
  documents letterbox/remap, SeaShips overlay mask, augmentation on/off, and the track
  metric/image-plane gate. Master at this date: 25,262 imgs / 71,501 boxes (incl. usv
  261/300).
