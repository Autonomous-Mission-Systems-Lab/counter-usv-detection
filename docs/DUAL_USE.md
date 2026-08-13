# Responsible Use

This project studies adversarial attacks on counter-USV detectors in order to build and evaluate a **defense**. Attack code exists for red-teaming — to measure and harden the defense — not to provide a turnkey capability against fielded systems.

## Scope
All work is **digital and simulated-physical only**: no adversarial patches are applied to real vessels, and there is no field deployment or RF transmission.

## What is released publicly
Released artifacts are centered on the **evaluation harness, metrics, regenerable figures/freezes, and derived features/annotations**, together with configuration and documentation. This is enough to audit reported numbers and to retrain from source data obtained by the user.

The adversary motion code ships as an **eval-only summary-feature / trajectory generator** driven by a **frozen sweep specification** (`configs/attacks/kinematics.yaml`, `results/adversary_motion/FROZEN_SWEEP.json`). It is not a mission planner, route optimizer, or fielded engagement tool.

Engagement annulus and warning/standoff readouts in configs and figures are **shared evaluation geometry** for scoring and cost curves (see `docs/ENGAGEMENT_GEOMETRY.md`). They do not introduce operational standoff or engagement parameters beyond constructs already used in published COLREGS / VTS-style encounter analysis and in this project's public metrics.

## What is withheld
- **Highest-fidelity physically optimized patch templates** are described rather than distributed. For the EO attack library freeze, see `results/attacks/REDACTION.md` (raw `patch_bank` tensors held locally for transfer reproducibility; excluded from public release).
- **Trained model weights** (detector checkpoints and defense envelope files) are **not** part of the public deposit. Shipping a ready-to-run defended detector is dual-use. Weights are available upon reasonable request for bona fide defensive research (institutional contact).

Defense-side release has no additional redaction surface beyond withheld weights: envelopes ship as model cards and freeze digests only.

## For users
Use this code in accordance with the licenses of the underlying datasets and applicable law and institutional policy. It is intended for defensive research and evaluation.

## Review note
Confirmed 2026-08-04 against the public release surface: patches described not shipped; motion model is a frozen-spec trajectory generator; engagement parameters are eval geometry only; weights withheld; attack redaction record at `results/attacks/REDACTION.md`.
