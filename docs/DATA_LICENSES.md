# Data Sources, Licenses & Attribution

This project builds on existing public maritime imagery datasets and a public AIS
trajectory archive. Each source remains under its original license and must be
obtained from its original provider; this repository does not re-host imagery or
bulk AIS feeds. Users must comply with, and cite, the original sources.

## Redistribution policy
This repository releases **annotations, derived trajectory features, configuration, and code layered over the original public data** rather than re-hosting images or raw AIS feeds, unless a source's license clearly permits re-hosting. To reproduce results, obtain each source from its provider and place it under `data/`.

## EO detection sources

| Source | Modality | Format | License / status | Role |
|---|---|---|---|---|
| **McShips** | EO | Detection scenes | **No stated license** from the authors' distribution (Zheng & Zhang, ICME 2020; GitHub `ZhengYitong2333/Mcships` README requests citation only). Third-party Roboflow mirrors that tag CC0 are **not** authoritative for the copy used here | Train-only auxiliary: military vs. civilian label signal. Never enters val/test or operational eval (`seaships` / `smd` / `usv` only) |
| **SeaShips** | EO | Detection scenes | See original dataset terms | Benign vessel diversity; shore-based surveillance viewpoint. |
| **Singapore Maritime Dataset (SMD)** | EO (+ reflective near-IR) | Video frames (on-shore subset sampled for detection) | See original dataset terms | Detection scenes at shore viewpoint (SMD-Plus labels). Near-IR is reflective, not thermal — unused. |
| **ABOShips (original)** | EO | Detection scenes | CC BY 4.0 (Zenodo rec. 4736931) | Supplementary vessel diversity (onboard viewpoint). Original 11-class version used (retains `militaryship`); **not** ABOShips-PLUS, which collapses to 4 superclasses. |

## Trajectory (kinematics) sources — the benign-behavior model

| Source | Type | License / status | Role |
|---|---|---|---|
| **MarineCadastre AIS (US)** | AIS archive (incl. Class B) | US public domain / CC0 (NOAA Office for Coastal Management); underlying USCG NAIS "Level C" historical data. Raw AIS carries a USCG no-retransmit / no-fee condition | Sole class-conditional benign-behavior corpus; small-craft Class-B coverage. Cite NOAA OCM. |

**AIS is used offline only** — to learn the benign-behavior model from historical archives. It is never a runtime input to the defense (see `THREAT_MODEL.md`). Redistribute **derived behavior features**, not bulk re-hosted AIS feeds — this also satisfies the USCG no-retransmit condition on MarineCadastre raw AIS.

**McShips — no stated license; train-only.** The authors' distribution carries no LICENSE file and states no redistribution grant beyond requesting citation of Zheng & Zhang, ICME 2020. Practical consequences for this repo:
- Do **not** re-host McShips imagery.
- Local academic training on the 9k lite subset is in scope; McShips is **train-only** in our splits and is excluded from operational clean-mAP sources.
- Public release of McShips-derived annotation products (e.g. remapped COCO exports) is **omitted** from the permissive derived-data slice. Obtain imagery via `scripts/data/fetch_data.py --source mcships` and regenerate annotations locally.
- Detector baselines were trained with McShips in the training mix; attribute McShips whenever results depend on those weights (weights themselves are withheld from the public deposit — see `DUAL_USE.md`).

## Notes
Licenses for third-party sources marked "See original dataset terms" are governed
entirely by their respective providers; consult each source's official distribution
for current terms before use or redistribution. ABOShips and MarineCadastre terms
above were recorded from provider documentation (Jul 2026). The McShips row was
corrected 2026-08-03 after confirming the authors' distribution states no license
(a prior CC BY-NC-ND reading was unsupported).

## Citations
Please cite McShips, SeaShips, SMD, and ABOShips per their respective providers when
using those EO sources; cite MarineCadastre / NOAA OCM when using the trajectory corpus.
