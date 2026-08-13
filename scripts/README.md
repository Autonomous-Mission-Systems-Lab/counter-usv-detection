# scripts

CLI entrypoints for data curation, detector baselines, the benign-behavior model,
and attacks. Library code lives under `src/counterusv/`; these scripts are the
thin runnable layer on top.

## Layout

| Folder | Purpose |
|---|---|
| `data/` | Acquire / annotate / audit / split EO + tracks; package derived release |
| `detector/` | Train / eval / freeze EO detector baselines (+ RunPod helpers) |
| `behavior/` | Benign corpus, windowed features, envelope fit / freeze / FAR |
| `defense/` | Wired defense pipeline, presence, oracle DDR, adaptive cost, freeze |
| `attacks/` | Marine-EOT + patch core + evasion/ESR + disguise/TMSR + transfer + oracle + freeze + adversary motion |
| `report/` | Regenerable paper figures from digest-verified freezes → `results/paper/` |
| `release/` | Weight-free data/results bundle for Zenodo (`build_bundle.py`) |

## Quick index

**data/** — `fetch_data` · `collect_usv` · `build_coco_master` · `audit_eo` · `build_eval_slice` · `build_splits` · `ingest_ais` · `export_eo_views` / `qa_eo_loader` · `package_derived`

**detector/** — `train_detector` · `eval_detector_clean` · `usv_capability` · `freeze_baselines` · `sync_eo_train_bundle.sh` · `setup_runpod_eo.sh`

**behavior/** — `build_benign_corpus` · `build_class_envelope_map` · `extract_windowed_features` · `fit_behavior_model` · `fit_geometry_model` · `freeze_behavior_model` · `validate_benign_far` · `validate_geometry_far` · `run_label_swap`

**defense/** — `smoke_pipeline` · `smoke_presence` · `run_oracle_ddr` · `run_adaptive_cost` · `freeze_defense`

**attacks/** — `render_marine_eot_grid` · `smoke_patch_core` · `run_evasion` · `run_disguise` · `run_transfer` · `run_oracle` · `freeze_attacks` · `generate_adversary_tracks` · `validate_adversary_motion` · `sync_*_runpod.sh` · `setup_runpod_*.sh`

**report/** — `build_all` · `fig_rq1_feasibility` · `fig_rq2_ddr_gap` · `fig_label_swap` · `fig_rq3_cost_warning` · `fig_far` · `fig_supp_severity` · `fig_supp_pooled_gap` · `captions` (Fig. 1 is hand-authored under `figures/`)

**release/** — `build_bundle` (weight-free data/results archive → `dist/release/`)

---

## `data/fetch_data.py` — dataset acquisition

Downloads the counter-USV datasets from their **original providers** into
`data/raw/` (no re-hosting; consistent with `docs/DATA_LICENSES.md`). Automates the
sources with stable direct URLs; prints exact manual steps + target paths for the
access-gated ones.

Recommended sequence:

```bash
# 1. See what exists and which sources are automated vs. gated
python scripts/data/fetch_data.py --list

# 2. Pull the automatable EO sources (SeaShips tries the WHU link;
# ABOShips is a ~8.2 GB Zenodo archive)
python scripts/data/fetch_data.py --source seaships aboships

# 3. Pull an AIS window (LARGE — ~hundreds of MB/day; keep it short)
python scripts/data/fetch_data.py --source marinecadastre_ais \
    --ais-start 2023-06-01 --ais-end 2023-06-07

# 4. Print manual instructions for the gated sources
python scripts/data/fetch_data.py --source mcships smd
```

| Source | Auto? | Notes |
|---|---|---|
| `seaships` | yes* | *WHU direct link is often down / serves HTML — the script detects that and prints the Kaggle/Roboflow mirror fallback. |
| `aboships` | yes | Zenodo rec. 4736931, single ~8.2 GB zip. CC BY 4.0. |
| `marinecadastre_ais` | yes | Needs `--ais-start/--ais-end`. NOAA OCM daily zips. |
| `mcships` | manual | Baidu / Google Drive (`pip install gdown` to auto-pull the GDrive id). No stated license from the authors; citation requested; no re-host. Train-only — omitted from the permissive release slice. |
| `smd` | manual | Original videos on Google Drive; SMD-Plus labels on GitHub. On-shore subset for EO detection frames. |

Extras:
- `--all-auto` runs every non-gated source (AIS skipped unless dates given).
- `--force` re-downloads existing files.
- Every landed file is sha256'd into `data/raw/CHECKSUMS.sha256` for the data card (data-card packaging).
- Optional deps: `tqdm` (progress bar; already in `requirements.txt`), `gdown` (Google-Drive sources).

After pulling, record exact image/track counts and the **Class-B share** in
`data/INVENTORY.md` (the source inventory open action items).

## `data/build_coco_master.py` — EO annotation harmonization

Taxonomy-driven converter: reads every EO source's native annotations, maps them
to the canonical classes in `data/taxonomy.yaml`, and writes per-source
`data/annotations/<source>.coco.json` + the merged `data/annotations/coco_master.json`
(with provenance) + `convert_summary.json`. See its `--help`.

## `data/export_eo_views.py` / `data/qa_eo_loader.py` — harmonized EO loader views + QA

Turn the COCO master + leakage-controlled splits into framework-ready views that honor
`data/HARMONIZATION.md` (letterbox contract, SeaShips overlay mask, ≥8px detector floor).
**No pixel copies** — YOLO images are symlinks into `data/raw/`.

```bash
# Emit per-split COCO JSON + Ultralytics YOLO layout (symlinks + labels)
python scripts/data/export_eo_views.py

# Visual QA: remapped boxes on letterboxed images, one/two samples per source
python scripts/data/qa_eo_loader.py --split train --input-size 640
# -> results/eo_loader_qa/report.md + sheet_*.png
```

The load-time Dataset lives in `src/counterusv/data/` (`EODetectionDataset`,
`letterbox_image`, `mask_seaships_overlay`). SeaShips overlay bands are locked at
top `[0,110)` / bottom `[980,1080)` on the native 1920×1080 release.

**Class space & label policy** — the exporter derives the trained class set from
`configs/base.yaml` `detector`: `exclude_classes` (e.g. `working_service`, 0 EO
boxes) get no head slot, and `non_target_policy` (`keep`|`drop`) controls whether
`static_aid`/`unknown_other` are trained as explicit classes (default `keep`;
`drop` gives the vessel-only ablation, or pass `--drop-non-target`). Emitted
`data.yaml` carries both `names` and `roles` (0-indexed ids mapping 1:1 to
canonical taxonomy) so the downstream class–kinematics check needs no re-map.
`data.yaml` uses an **absolute** `path` to the view dir (Ultralytics resolves
relative paths against CWD) — **re-run the exporter on the GPU box** after
syncing raw imagery so symlinks and `path` resolve locally (see `docs/RUNPOD.md`).

## `detector/train_detector.py` / RunPod helpers — detector baseline training

Train the three families in `configs/detector/families.yaml` (YOLO11-s/l surrogates
+ RT-DETR-l held-out target) on the harmonized EO view. Full runs belong on a
**CUDA GPU (RunPod)**; use `--dry-run` / `--smoke` on a laptop first. Details:
[`docs/RUNPOD.md`](../docs/RUNPOD.md).

```bash
# Laptop wiring check
python scripts/detector/train_detector.py --all --dry-run

# Sync ~10 GB EO bundle to the pod (from laptop; set RSYNC_RSH for RunPod SSH)
./scripts/detector/sync_eo_train_bundle.sh root@<HOST>:/workspace/counterUSV

# On the pod — always under tmux so SSH disconnects do not kill training
bash scripts/detector/setup_runpod_eo.sh
tmux new -s train
python scripts/detector/train_detector.py --all   # Ctrl-b d to detach
# later: tmux attach -t train
```

Flags: `--family yolo11s|yolo11l|rtdetr_l`, `--smoke` (1 epoch / 2% data),
`--imgsz 1280`, `--resume`, `--device auto|0|cpu|mps`. Each run writes
`run_meta.json` (wall-clock, GPU, transfer role, config snapshot).

## `detector/eval_detector_clean.py` — clean-mAP report (baselines)

After pulling `results/detector_baselines/<family>/weights/best.pt` from the pod,
score each family with augmentation **OFF** via `pycocotools`:

```bash
# Laptop (or pod) — once weights are local
python scripts/detector/eval_detector_clean.py --all                  # 640 (training resolution)
python scripts/detector/eval_detector_clean.py --family yolo11s
python scripts/detector/eval_detector_clean.py --all --dry-run        # slice counts / missing weights
```

Reports:

- **shore operational** (headline): test ∩ `{seaships, smd, usv}` (all shore-viewpoint
  eval sources; same as the full test split)
- **per-source**, **per-class AP** (highlights `usv` / small benign), **size bins**
  (COCO area + shortest-side)

Writes `results/detector_baselines/report.md` +
`results/detector_baselines/clean_map/<family>_s{imgsz}.json`.

Evaluated at the native **640** training resolution. Off-resolution eval
(e.g. `--imgsz 1280`) is not run by default — it penalises resolution-sensitive
architectures (RT-DETR degrades sharply off its training size) without a
matching 1280-trained model to compare, so it is left for a deliberate
sensitivity study only.

## `detector/usv_capability.py` — undisguised-USV recognition check

Confirms the baselines are **USV-recognition-capable** (the `docs/METRICS.md`
EO-baseline requirement), reusing the cached clean-eval detections (no
re-inference). On the undisguised-`usv` test slice it brackets each family's
capability between a **presence floor** (any-class localization) and the
**perfect-EO oracle** (=1.0), reporting `usv` AP, presence recall, recognition
(class-correct) recall, and precision at an operating point (conf ≥ 0.25,
IoU ≥ 0.5).

```bash
python scripts/detector/eval_detector_clean.py --all   # first, to cache detections
python scripts/detector/usv_capability.py --all
```

Writes `results/detector_baselines/usv_capability.md` +
`results/detector_baselines/clean_map/usv_capability.json`. Feeds the
EO-baseline note in `docs/THREAT_MODEL.md`.

## `detector/freeze_baselines.py` — pin the baseline set for downstream

Freezes the ≥3 trained families after clean-mAP + USV-capability are in:
SHA-256 of configs and `best.pt`, headline metrics, transfer roles, and the
grey-box / white-box / black-box access assumptions. Weights stay gitignored;
the freeze records digests so integrity can be checked later.

```bash
python scripts/detector/freeze_baselines.py            # writes FROZEN.json + smoke
python scripts/detector/freeze_baselines.py --skip-smoke
```

Human detail stays in `report.md` / `usv_capability.md` (not a second freeze markdown).
Canonical inference for attacks / the class–kinematics defense:

```python
from counterusv.models import DetectorBaseline

det = DetectorBaseline.from_freeze("yolo11s")
dets = det.predict_one("path/to/image.jpg")
# Detection: box_xyxy, score, class_name, class_id, role

DetectorBaseline.from_freeze("rtdetr_l").assert_attack_crafting_allowed()
# → PermissionError (held-out transfer target; surrogates only for crafting)
```

## `behavior/build_benign_corpus.py` — benign training corpus + feature contract

Assembles the one-class scorer's training pool: AIS train split ∩
`role==benign` only (hostile / non_target / `usv` hard-excluded). Freezes the
feature contract in `configs/defense/scorer_features.yaml`.

```bash
python scripts/behavior/build_benign_corpus.py
python scripts/behavior/build_benign_corpus.py --dry-run   # counts only
```

Writes:
- `data/behavior/benign_train_manifest.parquet` — training corpus
- `data/behavior/benign_corpus_summary.json` — retained / excluded counts
- `results/behavior_model/corpus_report.md`

## `behavior/build_class_envelope_map.py` — EO class → benign envelope map

Resolves which EO-asserted class is scored against which kinematic envelope
(or abstains). `small_craft` uses a Class-B ∩ {recreational, fishing, sailing}
proxy; hostile / non_target / `benign_unspecified` abstain.

```bash
python scripts/behavior/build_class_envelope_map.py
```

Config: `configs/defense/class_envelope_map.yaml`. Writes
`data/behavior/envelope_coverage.json` +
`results/behavior_model/envelope_map_report.md`.

## `behavior/extract_windowed_features.py` — observation-window kinematics

Recomputes the frozen scorer features over the **last W seconds** of each AIS
track (one window per track per length — no sliding). Sweep default
`[120, 180, 300, 600]` s; also writes whole-track features.

```bash
python scripts/behavior/extract_windowed_features.py
python scripts/behavior/extract_windowed_features.py --smoke   # 2k trips wiring check
```

Writes `data/behavior/features_window_{W}s.parquet`,
`features_whole_track.parquet`, and
`results/behavior_model/windows_report.md`.

## `behavior/fit_behavior_model.py` — class-conditional one-class envelopes

Fits per-envelope benign models on complete-window benign train tracks, at
several observation horizons (**120 / 180 / 300 s**). Primary: GMM (`k` swept to
12, chosen by a val-loglik knee rule); baselines: Mahalanobis (Ledoit–Wolf) and
IsolationForest. Dual subspace (`core` / `core_course`, course gated on non-null
COG). Thresholds = val benign FAR percentiles. Hostile data never used.

Each envelope is saved as a `MultiHorizonEnvelope` bundle; at score time the
longest horizon whose window is complete for the contact is used, so short
tracks are scored (coverage) instead of dropped.

```bash
python scripts/behavior/fit_behavior_model.py
python scripts/behavior/fit_behavior_model.py --smoke --envelope fishing
```

Writes `results/behavior_model/envelopes/<name>.joblib`,
`fit_summary.json`, and `fit_report.md` (primary-horizon FAR, per-horizon `k`,
and multi-horizon coverage/FAR). Recipe:
`configs/defense/behavior_model.yaml`.

## `behavior/fit_geometry_model.py` — kinematics + geometry envelopes

Fits the extended arm: kinematics features joined with asset-relative geometry
on **berth_approach / anchorage** encounter pairs (fit population). Horizons
**180 / 300 / 600 s** (primary **600 s**; 120 s omitted — AIS cadence yields no
usable inbound legs). Same GMM / Mahalanobis / IsolationForest dual-subspace
recipe; geometry columns appended to both subspaces. Also fits a
`pooled_benign` ablation (no class conditioning).

Kinematics-only envelopes under `results/behavior_model/` are left untouched.

```bash
python scripts/behavior/fit_geometry_model.py
python scripts/behavior/fit_geometry_model.py --smoke --envelope recreational
```

Writes `results/behavior_model_geometry/envelopes/<name>.joblib`,
`fit_summary.json`, `fit_report.md`, and `FROZEN.json` (scorer-compatible).
Recipe: `configs/defense/behavior_model_geometry.yaml`. Pipeline selects this
arm via `feature_arm: kinematics_geometry` in `configs/defense/pipeline.yaml`.

## `behavior/validate_geometry_far.py` — placement-swept FAR gate

Scores held-out benign `(trip, asset)` encounters under the frozen
`kinematics_geometry` envelopes across the **same digested placements table**
as the fit. Reports FAR as a distribution over `placement_class` /
`port_region`. Operating-point claim = berth + anchorage; fairway /
offshore_terminal are sensitivity strata.

**Gate:** operating-point overall test FAR@5% ≤ kinematics-only FAR@5% +
`far_gate.max_absolute_excess` (default 5 pp). Exit code 2 on FAIL.

```bash
python scripts/behavior/validate_geometry_far.py
python scripts/behavior/validate_geometry_far.py --smoke
```

Writes `results/behavior_model_geometry/far_placement_report.md` +
`far_placement_summary.json`. Unit tests: `tests/test_geometry_far.py`.

## `behavior/run_label_swap.py` — real-track class-swap discriminability

Scores attack-run-like AIS windows (p95 SOG ≥ 30 kn, straightness ≥ 0.95)
under a **different** asserted benign class on both feature arms. Synthesis-free
anchor for class-conditional discriminability; thresholds frozen (never
retuned). `small_craft` / Class-B proxy pairings are held-out only.

```bash
python scripts/behavior/run_label_swap.py
python scripts/behavior/run_label_swap.py --smoke
```

Writes `results/label_swap/label_swap_report.md`, `label_swap_summary.json`,
and `label_swap_cells.parquet`. Library helpers:
`counterusv.eval.label_swap`. Unit tests: `tests/test_label_swap.py`.

## ConsistencyScorer — class–kinematics defense interface

`counterusv.defense.ConsistencyScorer` scores an EO-asserted class against the
matching benign envelope (or abstains). FAR target is a knob; multi-horizon
bundles pick the longest complete window. Hostile / `usv` / `non_target` tracks
are refused as training material (`purpose="train"` / firewall helpers); the
eval path accepts them explicitly.

```python
from counterusv.defense import ConsistencyScorer
from counterusv.models import DetectorBaseline, Detection

scorer = ConsistencyScorer.from_artifacts()  # envelopes + class_envelope_map
# or, after freeze:
# scorer = ConsistencyScorer.from_freeze()
result = scorer.score("recreational", features)          # Mapping / Series
result = scorer.score_detection(det, features)           # Detection.class_name
# Multi-horizon (short-track coverage):
result = scorer.score(
    "fishing",
    features_by_window={120: f120, 180: f180, 300: f300},
    complete_windows={120, 180},   # longest complete → 180 s
    far_target=0.05,
)
# Eval-only hostile / adaptive kinematics:
result = scorer.score("fishing", attack_feats, purpose="eval",
                      track_meta={"role": "hostile", "source": "synth"})
```

`result.score` is the envelope **anomaly** score (higher = more inconsistent);
`is_inconsistent` is `score > threshold` at the requested FAR. Abstain classes
(`usv`, `military`, `benign_unspecified`, …) return `status="abstain"`.

Prefer the freeze for downstream defense wiring:

```python
scorer = ConsistencyScorer.from_freeze()  # verifies SHA-256 digests
```

## `defense/freeze_defense.py` — pin the defense bundle after evaluation

Re-attests both arm freezes (strips YAML config digests; envelope/data-pin
drift aborts), re-checks the benign-only training firewall, records shared
defense config *paths* plus digests of adversary-motion sweep + RQ2/RQ3
summary artifacts, and smoke-loads both consistency arms plus presence-only.

```bash
python scripts/defense/freeze_defense.py
python scripts/defense/freeze_defense.py --skip-smoke
```

Writes `results/defense/FROZEN.json` and `MODEL_CARD.md`. Pins include
oracle DDR, adaptive cost, real-track label-swap summaries, and the
pooled-vs-conditional ablation summaries (`label_swap_pooled/`,
`oracle_ddr_pooled/`).

## `report/` — regenerable paper figures

Builds the security-lite figure set into `results/paper/` from digest-verified
attack and defense freezes. Fig. 1 is hand-authored (`figures/fig1_system.svg`)
and copied as a static asset; Figs 2–6 and S1–S2 are regenerated from pinned
JSON/parquet. Digests are checked at run time against
`results/attacks/FROZEN.json` and `results/defense/FROZEN.json`.

```bash
python scripts/report/build_all.py
python scripts/report/build_all.py --no-verify   # draft only
python scripts/report/build_all.py --smoke       # temp dir + skip verify
python scripts/report/fig_rq1_feasibility.py
python scripts/report/fig_rq2_ddr_gap.py
python scripts/report/fig_label_swap.py
python scripts/report/fig_rq3_cost_warning.py
python scripts/report/fig_far.py
python scripts/report/fig_supp_severity.py
python scripts/report/fig_supp_pooled_gap.py
python scripts/report/captions.py
```

Outputs: `results/paper/fig{1..6,_S1,_S2}_*.{png,pdf,svg}` + `CAPTIONS.{json,md}` +
`PROVENANCE.json`. See `figures/README.md` for editing/export notes on Fig. 1.

The figures carry no titles and no footnote text — panels are labelled `(a)`,
`(b)`, … and everything descriptive lives in the captions. `captions.py` emits
one record per figure to `CAPTIONS.json`:

| Field | Use |
|---|---|
| `short_title` | bold caption lead-in / list-of-figures entry |
| `caption` | full caption body, panels described inline |
| `panels` | one line per panel, for slides and alt text |
| `takeaway` | the single claim the figure supports |
| `caveats` | limits a reviewer would otherwise raise |
| `sources` | pinned artifacts the panel was drawn from |
| `key_values` | every number the caption quotes |
| `latex_label` | stable `\ref` target |

Numbers are read from the same pinned summaries the figures plot, so a caption
cannot drift from its panel; `CAPTIONS.md` is a rendering of the same records.
Where a caption asserts a relation rather than a value (Fig. S1: black-box
severity never reaches a white/grey L0 baseline) the build fails if new data
breaks it.

## `defense/smoke_pipeline.py` — detector / oracle → decision

Wires `Detection` or `OracleAssertion` + world-frame track features through
`counterusv.defense.DefensePipeline` at the FAR operating point
(`configs/defense/pipeline.yaml`). Contact↔track association is assumed
*given* (radar/EO fusion) — not inferred.

```bash
python scripts/defense/smoke_pipeline.py
python scripts/defense/smoke_pipeline.py --no-verify-digests
```

```python
from counterusv.defense import DefensePipeline
from counterusv.attacks.oracle import PerfectDisguiseOracle

pipe = DefensePipeline.from_freeze()
decision = pipe.evaluate(detection, features)          # Detection
decision = pipe.evaluate(oracle_assertion, features, purpose="eval")
# decision.action ∈ {flag, pass, abstain}
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_pipeline.py`.

## `defense/smoke_presence.py` — presence-only cross-check + harness swap

The RQ2 comparator: an independent world-frame track exists but EO missed
or mislabeled the contact. Catches **evasion**; by construction cannot
reach **disguise** (EO detected the contact). Shares the
`DefenseBackend.evaluate → DefenseDecision` surface with the consistency
pipeline so later detection-rate work can swap defenses in one harness.

```bash
python scripts/defense/smoke_presence.py
python scripts/defense/smoke_presence.py --no-verify-digests
```

```python
from counterusv.defense import (
    PresenceOnlyDefense,
    presence_for_evasion,
    presence_for_disguise,
    evaluate_contact,
    load_defense,
)

presence = PresenceOnlyDefense.from_config()
presence.evaluate(presence=presence_for_evasion())          # → flag
presence.evaluate(oracle_assertion, purpose="eval")         # disguise → pass

# Same contact, two defenses:
evaluate_contact(presence, assertion=oracle, presence=presence_for_disguise(oracle))
evaluate_contact(load_defense("consistency", verify_digests=False),
                 assertion=oracle, features=feats, purpose="eval")
```

Config: `configs/defense/presence.yaml`. Library:
`counterusv.defense.presence`, `counterusv.defense.harness`.

Unit tests: `PYTHONPATH=src python -m pytest tests/test_presence.py`.

## `defense/run_oracle_ddr.py` — oracle DDR + defensibility gap

Scores the frozen adversary-motion sweep under the perfect-disguise oracle
(`asserted_class` = mimicked class, no patch) through both consistency arms
and the presence-only comparator. Detection rates are paired with each arm's
measured FAR. Writes only under `results/oracle_ddr/` — never mutates
`FROZEN_SWEEP.json`.

```bash
python scripts/defense/run_oracle_ddr.py --smoke
python scripts/defense/run_oracle_ddr.py
python scripts/defense/run_oracle_ddr.py --no-verify-digests
```

Config: `configs/defense/oracle_ddr.yaml`. Library: `counterusv.eval.oracle_ddr`.

Outputs: `results/oracle_ddr/oracle_ddr_report.md`, `oracle_ddr_summary.json`,
`oracle_ddr_cells.parquet`.

Unit tests: `PYTHONPATH=src python -m pytest tests/test_oracle_ddr.py`.

## `defense/run_adaptive_cost.py` — RQ3 cost curve

Joins terminal oracle DDR to the frozen motion sweep (no re-score for
curves). Primary cost axis: added approach time
`Δt_add = R (1/v_mimic − 1/v_max)` with `R = start − commit` (nm, kn → hours,
reported in minutes). Companion axes: `v_mimic`, commit range, bearing offset.
Optional causal checkpoint pass records first-flag warning time / standoff.
Never mutates `FROZEN_SWEEP.json`. Oracle-only (no patch slice).

```bash
python scripts/defense/run_adaptive_cost.py --skip-warning-time
python scripts/defense/run_adaptive_cost.py --smoke
python scripts/defense/run_adaptive_cost.py --no-verify-digests
```

Config: `configs/defense/adaptive_cost.yaml`. Library:
`counterusv.eval.adaptive_cost`.

Outputs: `results/adaptive_cost/adaptive_cost_{joined,curves,warning}.parquet`,
`adaptive_cost_report.md`, `adaptive_cost_summary.json`.

Unit tests: `PYTHONPATH=src python -m pytest tests/test_adaptive_cost.py`.

## Engagement geometry — defended-asset contract

`configs/defense/engagement_geometry.yaml` locks the asset point, engagement
annulus, inbound-leg definition, and multi-port placement policy used by the
geometry feature arm and the adversary motion model. Rationale (why this is
not `range_to_shore`): [`docs/ENGAGEMENT_GEOMETRY.md`](../docs/ENGAGEMENT_GEOMETRY.md).

Placement classes are defended-asset archetypes (ship at a berth, ship at
anchor, offshore terminal, plus a deliberately adversarial fairway placement
that is scored but never fit). `fit_population()` returns the archetypes whose
encounters train the geometry envelopes.

```python
from counterusv.defense import load_engagement_geometry
cfg = load_engagement_geometry()
cfg.max_range_nm, cfg.port_regions(), cfg.placement_classes()
cfg.fit_population()   # ['berth_approach', 'anchorage']
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_engagement_geometry.py`.

### Materialize placements + extract geometry features

```bash
# Digested 5×4 placements table (SHA-256 pin before any fit/FAR)
python scripts/defense/materialize_placements.py
python scripts/defense/materialize_placements.py --smoke   # miami_approach only

# Encounter-paired asset-relative windows (leaves kinematics parquets untouched)
python scripts/defense/extract_geometry_features.py
python scripts/defense/extract_geometry_features.py --smoke
python scripts/defense/extract_geometry_features.py --windows 300 600
```

Writes under `data/defense/`:

| Artifact | Path |
|---|---|
| Placements | `placements.parquet` + `placements_digest.json` + `placements_report.md` |
| Geometry windows | `features_geometry_window_{W}s.parquet` |
| Coverage | `geometry_coverage_report.md` (+ `.json`) |

Pure-function API: `counterusv.defense.geometry_features_from_points`.
Unit tests: `tests/test_geometry_features.py`, `tests/test_placements_materialize.py`.

## `behavior/freeze_behavior_model.py` — pin the scorer for downstream

Attests the benign-only firewall on the train manifest, records config paths
(git owns configs), digests envelope bundles, and writes the freeze + model card.

```bash
python scripts/behavior/freeze_behavior_model.py
python scripts/behavior/freeze_behavior_model.py --skip-smoke
```

Writes `results/behavior_model/FROZEN.json` and `MODEL_CARD.md`
(fit/FAR prose stays in `fit_report.md` / `far_report.md`).

## `behavior/validate_benign_far.py` — held-out FAR floor

Scores vessel-disjoint val/test AIS tracks with the frozen ConsistencyScorer
(multi-horizon; val-calibrated thresholds — no retuning). Reports FAR @ 1/5/10%,
region holdout, and an eval-only straight-line high-SOG separability preview.

```bash
python scripts/behavior/validate_benign_far.py
python scripts/behavior/validate_benign_far.py --smoke
```

Writes `results/behavior_model/far_report.md`, `far_summary.json`, and
`far_curves.png`. Split roles: train = fit; val = calibrate; test = FAR floor.

## `data/collect_usv.py` — provenance-first `usv` imagery curation kit

Curates the small hostile-platform `usv` EO set (EO/appearance baseline only). It is a
**manifest-first curation kit, not a scraper**: there is no existing dataset of hostile
USVs-*as-targets* to bootstrap (the literature's "USV datasets" — SeePerSea, PoLaRIS,
MODS, USVInland, MASS-LSVD, MV2 — are all ego-centric obstacle detection *by* an
autonomous vessel, the viewpoint this project excludes), and a keyword scrape cannot meet
the curated set's per-image provenance, link-only redistribution, or EO-channel-firewall rules.

Workflow: find each image/clip on an **authoritative, license-legible host** (DVIDS /
navy.mil public-domain, Wikimedia Commons per-file license, manufacturer/OSINT press —
link only), obtain it per its terms, then register it here with full provenance.

### Fast path — scrape + QC loop

Bulk-fetch candidate images by platform (records a source URL per image), QC them by
deleting the bad files, then re-run so the manifest auto-updates to the survivors:

```bash
# scrape candidates (DuckDuckGo image search; keyless). --all or --platform <names>
python scripts/data/collect_usv.py scrape --all --max 60
# -> images land in data/raw/usv/scraped/<platform>/ ; URLs logged in scrape_index.csv

# ... now browse those folders and DELETE the images that aren't useful ...

# reconcile: drop the deleted ones, rebuild the manifest (rejected URLs are remembered)
python scripts/data/collect_usv.py sync # (re-running `scrape` also reconciles first)
python scripts/data/collect_usv.py status # coverage by platform/role
```

Notes: the engine is DuckDuckGo image search (keyless, returns a source page per image);
Google Images has no stable keyless scrape (JS-rendered, obfuscated URLs, ToS). Scraped
rows carry `platform`, `role`, `image`+`source_url`, `sha256` — `license` stays blank
(scraped/unverified) and the repo still redistributes links+annotations only, never pixels.

### Volume booster — YouTube video → frames loop

When stills run thin, pull video and split it into frames. Two QC stages (videos, then
frames), and both "stick" — deleted videos aren't re-downloaded, deleted frames aren't
regenerated:

```bash
# 1. search + download per platform (yt-dlp; single-stream mp4, no ffmpeg needed)
python scripts/data/collect_usv.py videos --all --max 8 # or --platform magura_v5 sea_baby
# -> data/raw/usv/videos/<platform>/*.mp4 ; ledger in video_index.csv (url/title/uploader/date)

# ... QC: delete the clips that aren't useful ...

# 2. split the SURVIVING videos into deduped frames (once per video)
python scripts/data/collect_usv.py video-frames --fps 1 --max-frames 60
# -> data/raw/usv/frames/<platform>_yt_<id>/*.jpg ; frames inherit the video's URL/date

# ... QC: delete the bad frames ...

# 3. reconcile: manifest auto-updates to surviving frames (+ tombstones deleted videos)
python scripts/data/collect_usv.py sync
```

`videos` flags: `--per-query` (results/query to consider), `--max` (new videos/platform),
`--height` (max res), `--max-duration` (skip long clips), `--max-filesize-mb`.
`video-frames` flags: `--fps`/`--stride`, `--max-frames`, `--dedup-hamming`, `--force`
(re-extract). Frames carry `role` (from platform) and the video `source_url`; `license`
stays blank (backfill later). Same firewall/COCO passthrough as every other `usv` row.

Note (same as the image scrape): downloading video for research + extracting frames, with
link-only redistribution (URLs logged, pixels never re-hosted), is the basis here — verify
per-source terms before any release.

### Manual / provenance-first path

```bash
# 1. Write the per-platform authoritative seed source catalog -> data/raw/usv/sources.yaml
python scripts/data/collect_usv.py seeds

# 2. Register a still — SIMPLIFIED capture: only --platform + --source-url required
# (role auto-derived from the platform; add date/license/viewpoint later)
python scripts/data/collect_usv.py add-image --file ~/dl/mantas.jpg \
    --platform mantas_t12 --source-url https://www.dvidshub.net/image/XXXX
# ...full form when you have it:
# ... --viewpoint near_waterline --date 2026-07-10 --license PD-USGov --attribution "US Navy / TF59"

# 3. Extract frames from a LOCAL clip (obtain per its terms first); near-dup frames skipped
python scripts/data/collect_usv.py frames --file ~/dl/magura.mp4 \
    --platform magura_v5 --source-url https://en.wikipedia.org/wiki/MAGURA_V5 --fps 1

# 4. (optional) register a clearly-flagged generated frame (synthetic=true, excludable)
python scripts/data/collect_usv.py synth --file ~/gen/usv_render.png --platform magura_v5

# 5. Backfill provenance recorded later (by platform / clip / image-id / --all)
python scripts/data/collect_usv.py backfill --platform mantas_t12 \
    --license PD-USGov --date 2026-07-10 --attribution "US Navy / TF59"

# 6. Integrity + firewall + dedup check, and coverage/limitations note
python scripts/data/collect_usv.py verify # missing date/license = warning, not failure
python scripts/data/collect_usv.py status --write # -> results/usv/coverage.md
```

Produces under `data/raw/usv/` (gitignored; released via the data card as links+annotations):
- `manifest.csv` — **append-only per-image provenance** (image_id, file_name, platform,
  `role` (hostile/platform, for hostile-only vs all-in slicing), viewpoint, `source_url`,
  `date_accessed`, `license`, `attribution`, `sha256`, `synthetic`, and `channel=eo_only`).
  This is the redistributable deliverable.
- `sources.yaml` — per-platform authoritative seed catalog + host license posture.
- `images/`, `frames/<clip_id>/`, `synthetic/` — the registered pixels (not re-hosted).

Annotate the images (LabelImg/CVAT) into either `data/raw/usv/annotations.coco.json`
(CVAT COCO export) or `data/raw/usv/Annotations/*.xml` (LabelImg VOC). Then
`build_coco_master.py`'s **`usv` adapter** merges the set into the COCO master with the
canonical `usv` class, propagating `channel=eo_only` + `synthetic` + provenance onto each
image record — so the kinematic scorer can hard-exclude it (`source=="usv"`) and the
EO pixels-on-target audit (`audit_eo.py`) picks it up unchanged (same pixels-on-target
floor / dedup / QA).

```bash
python scripts/data/build_coco_master.py --sources usv # merge just the usv set
python scripts/data/audit_eo.py # re-run the EO audit over the master
```

Frame extraction uses OpenCV (already a dep); no `yt-dlp`/`ffmpeg` needed. Obtain video
clips yourself per each source's terms (e.g. `yt-dlp` if you have rights), then point
`--file` at the local file.

## `data/audit_eo.py` — EO context, quality & pixels-on-target audit

Source-agnostic audit over `data/annotations/coco_master.json`; re-runs cheaply as
new sources land (the curated `usv` set from usv curation passes through unchanged).

```bash
python scripts/data/audit_eo.py # full audit (all sources in the master)
python scripts/data/audit_eo.py --sources smd # subset
python scripts/data/audit_eo.py --no-hash # skip md5/dHash dedup (size/context only)
python scripts/data/audit_eo.py --jobs 8
```

Outputs:
- `data/audit/eo_annotations.csv` — per-annotation: bbox pixels, area fraction,
  centering, expected post-letterbox size (640/1280), and the non-destructive
  **detector-eligible** (≥8px) and **patch-eligible** (≥32px) flags. Nothing is
  deleted; retained counts are reported across a threshold sweep (final floor set
  in the harmonization spec).
- `data/audit/eo_images.csv` — per-image: object count, largest-object area
  fraction, chip-like context tag, sequence/video id, file QA, and exact
  (md5) + perceptual (256-bit dHash) duplicate groups (leakage inputs for leakage-controlled splits).
- `data/audit/eo_audit_summary.json` — aggregates, per-source stats, threshold
  sweeps, QA counts.
- `results/eo_audit/report.md` + `results/eo_audit/figures/*.png` — human report.

Image hashes are cached in `data/audit/_hash_cache.json` (size+mtime keyed), so
re-runs over an unchanged master take seconds. Process pools fall back to threads
under sandboxes that block SysV semaphores.

## `data/build_eval_slice.py` — operational small-craft eval slice

A **versioned manifest over the COCO master (via the EO audit)** — it *selects and
tags* existing annotations/images, copying nothing. Source-agnostic; re-run after the
`usv` set + a re-audit land. Selects the benign classes a hostile USV plausibly
mimics (`small_craft`, `recreational`, `fishing`, `sailing`) for **false-alarm /
transfer reporting**.

```bash
python scripts/data/build_eval_slice.py
python scripts/data/build_eval_slice.py --classes small_craft recreational fishing sailing
```

Every row is **viewpoint-tagged** (`operational_viewpoint`): shore near-waterline
(SeaShips, SMD) is the operational "detection *of* a USV" viewpoint; ABOShips is
onboard/moving-vessel ("detection *by* a vessel") and must be reported separately.
Outputs:
- `data/eval_slices/small_craft_eval_annotations.csv` — per-instance (class, size bin,
  eligibility, viewpoint, `sequence_id`/dedup leakage keys for leakage-controlled splits).
- `data/eval_slices/small_craft_eval_images.csv` — per-image (per-class counts, viewpoint).
- `data/eval_slices/small_craft_eval_summary.json` + `results/eval_slice/report.md` +
  figures. Version = date + input-md5 + git SHA (once the repo is committed).

**Benign-only**: no real hostile small USVs in these sources; the hostile `usv` platform
comes from `collect_usv.py` and is EO-baseline-only.

## `data/build_splits.py` — train/val/test splits with leakage control

Writes **split manifests** over the curated data (nothing copied/modified), each with its
own leakage rule, and a self-checking leakage report (non-zero exit on any violation).

```bash
python scripts/data/build_splits.py # 70/15/15, seed 1337
python scripts/data/build_splits.py --ratios 0.8 0.1 0.1 --seed 7
python scripts/data/build_splits.py --eval-sources seaships smd # usv always added
```

- **EO** (`data/splits/eo_image_splits.csv`): per-image split over the COCO master (via the
  EO audit). Dedup groups (exact + perceptual), sequence sources (SMD clip / ABOShips
  recording-day), and curated **USV source-video clips** are unioned into components that
  stay intact in one split. Operational eval = shore `seaships`+`smd` (from
  `configs/base.yaml → eval_operational_sources`) **plus hostile `usv`**; non-operational
  `aboships`+`mcships` are **train-only** (firewalled from val/test); provider `orig_split`
  is not trusted. Tags a `transfer_seed` (operational shore test imgs) for RQ5.
- **AIS** (`data/splits/ais_track_splits.csv`): **vessel-disjoint** by MMSI (`mmsi==0`
  train-only); `region`/`geo_cell`/`start_day` tags support a region/time-holdout eval. The
  one-class benign scorer consumes `role==benign` only.
- `data/splits/splits_summary.json` + `results/splits/report.md` — counts, per-source/class
  × split tables, and the leakage-check result. Version = date + input-md5 + git SHA.

## `data/ingest_ais.py` — AIS trajectory corpus

Ingests the **MarineCadastre** US national daily CSVs into the real benign-behavior
corpus. AIS is used **offline only** (never a runtime input). Self-reported numeric
`VesselType` is mapped through `data/taxonomy.yaml`'s `ais_ship_type` table to the
same canonical classes the EO detector asserts, so the class–kinematics consistency
check is well-defined.

```bash
python scripts/data/ingest_ais.py # all MarineCadastre days on disk
python scripts/data/ingest_ais.py --days 2023-06-01 2023-06-02
python scripts/data/ingest_ais.py --bbox 32 42 -125 -117 # lat_min lat_max lon_min lon_max
python scripts/data/ingest_ais.py --cadence 60 --gap 1800 # thinning cadence / trip-gap (s)
```

Pipeline: valid-coord/SOG/COG/heading filters → `(MMSI,t)` dedup → teleport removal
(>80 kn implied) → cadence thinning → >30 min-gap trip segmentation → per-track
class-conditional kinematic features. Outputs:
- `data/tracks/tracks_ais.parquet` — one row per trip-level track (class + features +
  provenance); the named deliverable, feeds the benign-behavior model.
- `data/tracks/tracks_ais_points.parquet` — cleaned, thinned point-level tracks for
  windowed / time-to-flag scoring.
- `results/ais_ingest/report.md` + `summary.json` + `figures/*.png` — Class-B share
  and carriage/self-report bias reported honestly.

Per-column definitions (dtype, units, caveats) for both parquets are in
[`../docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md).

`range_to_shore` is intentionally not computed (AIS ingestion: encodes geography, not
behavior). Only the MarineCadastre adapter is wired.

## `data/package_derived.py` — version + checksum the derived release

Checksums every redistributable derived artifact (annotations **except McShips**,
audit/eval/split manifests, trip-level trajectory parquets, behavior/defense
features, taxonomy, harmonization contract, data cards, USV provenance manifest)
and writes a release stamp. **Does not** package raw EO imagery, bulk AIS feeds,
point-level AIS (`tracks_ais_points.parquet`), McShips annotations, or model
weights.

```bash
python scripts/data/package_derived.py           # write CHECKSUMS.derived.sha256 + RELEASE.json
python scripts/data/package_derived.py --strict  # fail if any expected artifact is missing
```

Outputs:
- `data/CHECKSUMS.derived.sha256` — sha256sum-compatible digests
- `data/RELEASE.json` — version stamp (`YYYY-MM-DD+<gitsha>`), counts, file inventory

See `data/DATACARD_EO.md` and `data/DATACARD_TRACKS.md` for the human-readable data
cards (`data/DATACARD.md` is a thin index over both).

## `release/build_bundle.py` — weight-free Zenodo data/results bundle

Stages the derived inventory (with release transforms), evaluation freezes,
summaries, and `results/paper/` under `dist/release/stage/`, then writes
`RELEASE.json`, `CHECKSUMS.sha256`, `ARTIFACT_README.md`, and a `.tar.gz`.

Transforms (staging only — local `data/` unchanged): drop `mmsi` columns; strip
McShips rows from `coco_master` / audit CSVs. Redaction gate fails if any weight
binary or patch bank lands in the staged tree.

```bash
python scripts/release/build_bundle.py
python scripts/release/build_bundle.py --verify   # checksums + no-weights
```

Companion to the GitHub code DOI: this archive is the data/results Zenodo deposit.
Model weights are withheld (see `docs/DUAL_USE.md`).

## `attacks/render_marine_eot_grid.py` — marine-EOT sample grid

Renders the frozen marine expectation-over-transformation library
(`configs/attacks/marine_eot.yaml` → `counterusv.attacks.marine_eot`) as a
per-axis × severity contact sheet, plus a joint Monte-Carlo sample strip.
Used both as a visual QA of the transform distribution and as the shared
severity ladder for patch-optimization expectation and attack-robustness
eval sweeps (scale, rotation, motion blur, glare, spray, grazing angle,
sea state; L0=identity → L4=extreme).

```bash
python scripts/attacks/render_marine_eot_grid.py --synthetic   # no EO imagery needed
python scripts/attacks/render_marine_eot_grid.py --image path/to.jpg
```

Writes `results/attacks/marine_eot/severity_grid.png`, `joint_samples.png`,
per-cell PNGs, `grid_meta.json`, and `report.md`.

Library API (NumPy eval path + differentiable Torch path for EOT expectation):

```python
from counterusv.attacks import MarineEOT
eot = MarineEOT.from_config()
img2 = eot.apply_axis(img, "glare", "L3")   # severity sweep
img3 = eot.sample(img)                      # joint Monte-Carlo draw
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_marine_eot.py`.

## `attacks/smoke_patch_core.py` — physically-realizable patch core QA

Exercises `counterusv.attacks.patch` / `configs/attacks/patch.yaml`: hull/superstructure
placement on a target box, TV + NPS regularizers, marine-EOT expectation inside the
optimize loop, and the ≥32px patch-eligibility floor (`data/HARMONIZATION.md`). The
smoke uses a **dummy** brightness objective only — evasion (ESR) and disguise (TMSR)
losses are wired in later attack steps.

```bash
python scripts/attacks/smoke_patch_core.py --synthetic --no-eot --steps 15
python scripts/attacks/smoke_patch_core.py --synthetic --steps 10   # with marine-EOT
```

Writes `results/attacks/patch_core/{00_clean,01_init,02_optimized}*.png`,
`patch_{init,optimized}.png`, `smoke_meta.json`, and `report.md`.

```python
from counterusv.attacks import PatchCore
core = PatchCore.from_config()
patch = core.init_patch(device="cpu")
for m in core.optimize(image_chw, box_xyxy, patch, attack_loss_fn, steps=50):
    ...
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_patch.py`.

## `attacks/run_evasion.py` — evasion attack (ESR)

White-box evasion patches on the `usv` test slice: optimize a physically-realizable
patch (`counterusv.attacks.evasion` + `configs/attacks/evasion.yaml`, layered on
`patch.yaml` / `marine_eot.yaml`) to suppress the target-class confidence below the
detection threshold, then score **ESR** (`docs/METRICS.md`) with the hard detector —
clean vs. attacked, swept per marine-EOT axis × severity (L0–L4). Reports raw ESR
**and** a patch-attributable ESR (excludes targets the transform suppresses on its own).

Optimization is the heavy part (steps × EOT samples forwards per target); full runs
belong on a GPU host. Use `--dry-run` / `--max-images` / `--steps` for laptop
wiring. **RunPod workflow:** see [`docs/RUNPOD.md`](../docs/RUNPOD.md) § Evasion
(`sync_evasion_runpod.sh` → `setup_runpod_evasion.sh` → `run_evasion.py --device 0`
inside tmux).

```bash
python scripts/attacks/run_evasion.py --dry-run                 # slice + wiring only
python scripts/attacks/run_evasion.py --max-images 3 --steps 30 # quick smoke (CPU/MPS)
# On RunPod (GPU; --save-patches exports patch_bank/ for transfer):
python scripts/attacks/run_evasion.py --family yolo11s --device 0 --save-patches
```

Writes `results/attacks/evasion/<family>/report.md`, `esr_by_severity.json`,
`instances.json`, and a `gallery/`.

```bash
# Laptop → pod (EO bundle must already be there):
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
./scripts/attacks/sync_evasion_runpod.sh root@<IP>:/workspace/counterUSV
# On pod:
bash scripts/attacks/setup_runpod_evasion.sh
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_evasion.py`.

## `attacks/run_disguise.py` — disguise attack (TMSR)

White-box hostile→benign patches on the `usv` test slice (`counterusv.attacks.disguise`
+ `configs/attacks/disguise.yaml`). Optimizes toward each target benign class
(**fishing**, **recreational**), then scores **TMSR** across marine-EOT severity
(raw + patch-attributable). Full runs on GPU (~2× evasion: two benign classes).

```bash
python scripts/attacks/run_disguise.py --dry-run
python scripts/attacks/run_disguise.py --max-images 2 --steps 20 --benign-class fishing
# On RunPod:
python scripts/attacks/run_disguise.py --family yolo11s --device 0 --save-patches
```

Writes `results/attacks/disguise/<family>/<benign>/` and `<family>/summary.md`.

```bash
./scripts/attacks/sync_disguise_runpod.sh root@<IP>:/workspace/counterUSV
# On pod: bash scripts/attacks/setup_runpod_disguise.sh
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_disguise.py`.

## `attacks/run_transfer.py` — access-level transfer (grey / black-box)

Hard-eval saved patch banks on other detectors (`configs/attacks/access_levels.yaml`).
**Grey-box** = yolo11s ↔ yolo11l; **black-box** = YOLO → `rtdetr_l`. No re-optimize.
Requires craft with `--save-patches` first.

```bash
python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --dry-run
python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --device 0
python scripts/attacks/run_transfer.py --attack disguise --surrogate yolo11s \
  --benign-class fishing --device 0
```

Writes `results/attacks/transfer/<attack>/<surrogate>_to_<target>/`.

```bash
./scripts/attacks/sync_transfer_runpod.sh root@<IP>:/workspace/counterUSV
# On pod: bash scripts/attacks/setup_runpod_transfer.sh
```

Unit tests: `PYTHONPATH=src python -m pytest tests/test_transfer.py`.

## `attacks/run_oracle.py` — perfect-disguise oracle (no-patch assertion)

Emits benign class assertions for hostile `usv` contacts without modifying
pixels (`configs/attacks/oracle.yaml` → `counterusv.attacks.oracle`). Models an
ideal patch / zero-tech visual disguise for the oracle DDR condition.

```bash
python scripts/attacks/run_oracle.py --dry-run
python scripts/attacks/run_oracle.py --benign-class fishing
python scripts/attacks/run_oracle.py --all-benigns
```

Writes `results/attacks/oracle/<benign>/assertions.json` + `report.md`.

Unit tests: `PYTHONPATH=src python -m pytest tests/test_oracle.py`.

## `attacks/freeze_attacks.py` — EO attack artifact v1 freeze

Records attack config paths (git owns configs), pins SHA-256 digests of
headline ESR/TMSR/transfer/oracle results, builds an illustrative sample
gallery, and records dual-use redaction of raw patch tensors (`docs/DUAL_USE.md`).
Scope is EO-only (no motion model).

```bash
python scripts/attacks/freeze_attacks.py
python scripts/attacks/freeze_attacks.py --skip-gallery   # digests + notes only
```

Writes `results/attacks/FROZEN.json`, `RELEASE_NOTES.md`, `REDACTION.md`, and
`artifact_v1/gallery/` (composited scenes — not printable patch templates).

## `attacks/generate_adversary_tracks.py` — hostile / adaptive motion sweep

Materializes the eval-only two-phase adversary motion model relative to frozen
fit-population placements. Emits AIS-cadence world-frame points
(`trip_id/t/lat/lon/sog/cog`) under `results/adversary_motion/` and
**SHA-256 freezes the sweep cell table before any DDR scoring**. Never writes
into `data/behavior/` or defense train tables.

```bash
python scripts/attacks/generate_adversary_tracks.py          # full freeze + points
python scripts/attacks/generate_adversary_tracks.py --smoke  # tiny grid (wiring)
python scripts/attacks/generate_adversary_tracks.py --skip-points
```

Writes `sweep_cells.parquet`, `tracks_points.parquet`, `tracks_meta.json`, and
`FROZEN_SWEEP.json` (smoke writes `FROZEN_SWEEP_smoke.json` so it cannot
overwrite the headline freeze). Library: `counterusv.attacks.kinematics`.

## `attacks/validate_adversary_motion.py` — generator validity (no DDR)

Dynamics caps, post-thin cadence (~60 s), kinematics + geometry extractability
(geometry primary window **600 s**), negative-control ≈ FAR smoke on both
arms, and class-contrast smoke. **Does not claim DDR or cost curves.**

```bash
python scripts/attacks/validate_adversary_motion.py
python scripts/attacks/validate_adversary_motion.py --smoke
```

Writes `results/adversary_motion/validity_report.md` + `validity_summary.json`.
Unit tests: `PYTHONPATH=src python -m pytest tests/test_adversary_kinematics.py`.
