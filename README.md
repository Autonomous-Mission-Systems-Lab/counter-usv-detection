# Counter-USV Adversarial Robustness

Testbed and class–kinematics consistency defense for shore-based counter-USV EO detection under adversarial attack.

## Overview

### Why this matters

Detecting small unmanned surface vessels (USVs) before they close range is a growing coastal-defense problem. Shore radar is the usual backbone, but small, slow, or low-observable craft can fall below its thresholds, and cooperative AIS does nothing against a craft that never transmits. Electro-optical (EO) detectors are being fielded to close that gap.

Seeing an *undisguised* USV is not the hard case. An adversary can instead:

- **Evade** — suppress EO detection entirely (present → absent), or
- **Disguise** — remain detected, but as benign traffic such as fishing or recreational craft (hostile → benign).

These threats are not equal in a layered system. Evasion must beat every sensor: an EO miss can still appear on radar. A **benign label** explains the contact away, so presence-only fusion does not catch it. That residual **disguise** threat is what this project targets.

A further scoping point: most maritime “USV detection” benchmarks study perception *by* a USV (collision avoidance). This work studies the opposite — shore-based detection *of* an incoming hostile craft.

### Research gap

Existing adversarial work on ships is mostly overhead satellite imagery, onboard “detection by” platforms, or fusion that only checks whether a track *exists*. Class–track consistency defenses appear in other domains (notably driving), but they usually (i) score image-plane bounding-box tracks that a pixel attack can manipulate, (ii) train on labeled tracks that already include the defended classes, and (iii) never require the adversary to change the object’s **physical motion**. AIS-based vessel-behavior models are mature, yet have not been linked to vision-detector robustness. And unlike many aerial threats, a hostile USV is often *built* to look like ordinary small craft — so hostile→benign disguise has no clean counter-UAV analog.

This repository targets that gap: a shore-based EO detector defended against a craft that obtains a benign label (via adversarial patch or visual disguise), using **world-frame kinematics** scored against a **benign-only behavior model** learned from real trajectory data — and measuring the cost when a defense-aware adversary must actually move like that class.

Three research questions structure the work:

1. **Attack feasibility.** With a physically realizable, marine-surviving perturbation, how hard is full EO non-detection versus a hostile→benign flip — across imaging conditions and white / grey / black-box access (including transfer to a held-out detector)?
2. **Defense against disguise.** Does class–kinematics consistency catch hostile→benign cases that presence-only fusion misses, and at what false-alarm rate on real benign traffic?
3. **Adaptive-adversary cost.** What operational cost (approach delay, restricted speed or geometry) must an adversary pay to mimic the spoofed class’s motion and defeat the check?

### What this repository provides

- **Attacks** — evasion and targeted misclassification (disguise), marine imaging transforms, access-level transfer, and a perfect-disguise oracle (no-patch class assertion)
- **Defense** — class–kinematics consistency: flag contacts whose EO-asserted class disagrees with observed motion (`DefensePipeline` wires detections / oracle assertions to a decision)
- **Benign-behavior model** — class-conditional, trained only on real AIS trajectories (MarineCadastre); hostile tracks never enter training
- **Evaluation harness** — metrics, configs, and scripts for attack × defense comparison and kinematic cost curves

Misclassification is treated as means-agnostic (patch or genuine visual disguise). All work is digital and simulated-physical only — see [`docs/DUAL_USE.md`](docs/DUAL_USE.md).

## Repository layout

```
src/counterusv/
  data/         # dataset curation, harmonization, loaders
  models/       # detector families (baselines)
  attacks/      # evasion + targeted-misclassification, marine-EOT, adaptive attack
  defense/      # class-kinematics consistency + baseline defenses
  kinematics/   # trajectory ingestion + benign-behavior model
  eval/         # metrics, attack × defense matrix, cost curve, real-track FAR
configs/        # experiment configs (base.yaml + overrides)
scripts/        # entry-point scripts (see scripts/README.md)
docs/           # threat model, metrics, transfer protocol, data licenses
data/           # raw datasets & tracks (not tracked; see docs/DATA_LICENSES.md)
results/        # experiment outputs (not tracked)
tests/          # tests
```

## Setup

Python 3.10+ with PyTorch. A CUDA GPU is recommended for detector training and full attack sweeps; CPU/MPS are fine for wiring checks and most data-prep steps.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pinned versions used for reported runs live in `requirements.lock.txt`. Optional cloud-GPU sync / train workflow: [`docs/RUNPOD.md`](docs/RUNPOD.md).

```bash
# Wiring check (no training)
python scripts/detector/train_detector.py --all --dry-run

# Train detector baselines (CUDA)
python scripts/detector/train_detector.py --all
```

Script-level usage for data prep, attacks, and the behavior model is documented in [`scripts/README.md`](scripts/README.md).

## Reproduction

Paper figures and headline numbers are regenerable from **frozen digests** (not from re-running training). With local `results/**/FROZEN.json` artifacts (or the weight-free data/results bundle once published):

```bash
python scripts/report/build_all.py          # verify freeze digests, write results/paper/
```

- **Weights** are withheld from the public deposit ([`docs/DUAL_USE.md`](docs/DUAL_USE.md)); request them for bona fide defensive research if you need detector/envelope reload.
- **Source imagery / AIS** come from providers ([`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md)); derived tables are checksummed in `data/CHECKSUMS.derived.sha256`.
- Per-stage CLI, `--smoke` wiring checks, and GPU notes: [`scripts/README.md`](scripts/README.md) · [`docs/RUNPOD.md`](docs/RUNPOD.md).

## Data

Raw datasets and trajectory corpora live under `data/` and are **not** tracked in git. Obtain sources from their providers (`scripts/data/fetch_data.py`; see [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md)), then regenerate derived products with the pipeline in `scripts/`.

Tracked with the repo:

- Public contracts: `data/DATACARD.md`, `data/DATACARD_EO.md`, `data/DATACARD_TRACKS.md`, `data/HARMONIZATION.md`, `data/taxonomy.yaml`
- Derived-release checksums: `data/CHECKSUMS.derived.sha256`, `data/RELEASE.json`

Redistribution defaults to annotations + derived features + code over the original public data — never raw imagery or bulk AIS feeds. Start with the EO and tracks data cards above.

## Documentation

- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) — asset, adversary goal / knowledge / capability, out-of-scope
- [`docs/METRICS.md`](docs/METRICS.md) — definitions of reported metrics
- [`docs/ENGAGEMENT_GEOMETRY.md`](docs/ENGAGEMENT_GEOMETRY.md) — defended-asset / placement policy (geometry arm)
- [`docs/TRANSFER_PROTOCOL.md`](docs/TRANSFER_PROTOCOL.md) — black-box transfer protocol
- [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) — data sources, licenses, and attribution
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — schema for derived trajectory parquet files
- [`docs/DUAL_USE.md`](docs/DUAL_USE.md) — responsible-use / dual-use review (weights and patch banks withheld; motion model is eval-only)
- [`LICENSE`](LICENSE) (MIT, code) · [`LICENSE-DATA`](LICENSE-DATA) (CC BY 4.0, derived data) · [`CITATION.cff`](CITATION.cff)
- [`docs/RUNPOD.md`](docs/RUNPOD.md) — optional cloud-GPU sync / setup / train workflow
- [`data/HARMONIZATION.md`](data/HARMONIZATION.md) — train-time EO / track contract
- [`data/taxonomy.yaml`](data/taxonomy.yaml) — canonical classes and source label maps
