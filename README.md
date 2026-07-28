# Counter-USV Adversarial Robustness

Testbed and class–kinematics consistency defense for shore-based counter-USV EO detection under adversarial attack.

## What this project does

A shore-based EO detector classifies incoming small craft as hostile or benign. We study two attack families against that detector — **evasion** (present → absent) and **targeted misclassification** (hostile → benign) — and build a **class–kinematics consistency defense** that flags a contact when the vessel class asserted by the EO detector is inconsistent with the track's observed motion, scored against a **class-conditional benign-behavior model learned from large-scale real trajectory data** (AIS archives + video-derived tracks). Because the benign model is learned from real tracks and no hostile data is used in training, the defense makes the modest, defensible claim that a track is inconsistent with the real benign class model rather than recognizing attacks it was trained on. All work is digital and simulated-physical only.

## Documentation

- `docs/THREAT_MODEL.md` — asset, adversary goal/knowledge/capability, out-of-scope.
- `docs/METRICS.md` — precise definitions of every reported metric.
- `docs/TRANSFER_PROTOCOL.md` — the black-box transfer protocol.
- `docs/RUNPOD.md` — RunPod sync / setup / train workflow for detector baselines.
- `docs/DATA_LICENSES.md` — data sources, licenses, and attribution.
- `docs/DATA_DICTIONARY.md` — schema for derived trajectory parquet files.
- `docs/DUAL_USE.md` — responsible-use statement.
- `data/DATACARD_EO.md` — EO detection dataset card (composition, floors, splits, limitations).
- `data/DATACARD_TRACKS.md` — vessel trajectory dataset card.
- `data/HARMONIZATION.md` — train-time EO / track contract.
- `data/taxonomy.yaml` — canonical classes and source label maps.

## Repository layout

```
src/counterusv/
  data/         # dataset curation, harmonization, loaders
  models/       # detector families (baselines)
  attacks/      # evasion + targeted-misclassification, marine-EOT, adaptive attack
  defense/      # class-kinematics consistency + baseline defenses
  kinematics/   # trajectory ingestion + benign-behavior model
  eval/         # metrics, attack x defense matrix, cost curve, real-track FAR
configs/     # experiment configs (base.yaml + overrides)
scripts/     # entry-point scripts
docs/        # frozen scoping documents
data/        # raw datasets & tracks (not tracked; see docs/DATA_LICENSES.md)
results/     # experiment outputs (not tracked)
tests/       # tests
```

## Environment

Single-GPU project. **Author on a laptop; train detector baselines on RunPod
(CUDA).** Target Python 3.10+ with PyTorch. Install:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependency versions are pinned after the first working install
(`requirements.lock.txt`). Full RunPod sync → setup → train workflow:
[`docs/RUNPOD.md`](docs/RUNPOD.md).

```bash
# Laptop: wiring check (no training)
python scripts/detector/train_detector.py --all --dry-run

# RunPod: after bash scripts/detector/setup_runpod_eo.sh
python scripts/detector/train_detector.py --all
```

## Compute & phasing

Designed for a single GPU over ≈ two semesters. Detector baseline training
targets a rented **24 GB** CUDA GPU (RunPod); the laptop is for data prep, QA,
and `--smoke` / `--dry-run` checks. First-to-cut scope if time is short: the
public leaderboard and multi-architecture adversarial-training baselines.

## Data

Raw datasets and trajectory corpora live under `data/` and are **not** tracked in git.
Obtain sources from their providers (`scripts/data/fetch_data.py`; see `docs/DATA_LICENSES.md`),
then regenerate derived products with the pipeline in `scripts/`. The public contracts
(`DATACARD.md`, `HARMONIZATION.md`, `taxonomy.yaml`) and the derived-release checksums
(`CHECKSUMS.derived.sha256`, `RELEASE.json`) *are* tracked. Redistribution defaults to
annotations + derived features + code over the original public data — never raw imagery
or bulk AIS feeds. Start with `data/DATACARD_EO.md` / `data/DATACARD_TRACKS.md`.
