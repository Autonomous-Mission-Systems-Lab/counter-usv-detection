# Data Sources, Licenses & Attribution

This project builds on existing public maritime imagery datasets and public AIS trajectory archives. Each source remains under its original license and must be obtained from its original provider; this repository does not re-host imagery or bulk AIS feeds. Users must comply with, and cite, the original sources.

## Redistribution policy
This repository releases **annotations, derived trajectory features, configuration, and code layered over the original public data** rather than re-hosting images or raw AIS feeds, unless a source's license clearly permits re-hosting. To reproduce results, obtain each source from its provider and place it under `data/`.

## EO detection sources

| Source | Modality | Format | License / status | Role |
|---|---|---|---|---|
| **McShips** | EO | Detection scenes | **CC BY-NC-ND** (or equivalent academic research license); non-commercial, no derivatives of the dataset | Military vs. civilian categories — hostile/benign label source. Train-only auxiliary in our splits. |
| **SeaShips** | EO | Detection scenes | See original dataset terms | Benign vessel diversity; shore-based surveillance viewpoint. |
| **Singapore Maritime Dataset (SMD)** | EO (+ reflective near-IR) | Video frames | See original dataset terms | Detection scenes; on-shore video also supplies **video-derived non-cooperative tracks** (SMD video-track ingestion). Its near-IR is reflective, not thermal. |
| **ABOShips (original)** | EO | Detection scenes | CC BY 4.0 (Zenodo rec. 4736931) | Supplementary vessel diversity (onboard viewpoint). Original 11-class version used (retains `militaryship`); **not** ABOShips-PLUS, which collapses to 4 superclasses. |

## Trajectory (kinematics) sources — the benign-behavior model

| Source | Type | License / status | Role |
|---|---|---|---|
| **MarineCadastre AIS (US)** | AIS archive (incl. Class B) | US public domain / CC0 (NOAA Office for Coastal Management); underlying USCG NAIS "Level C" historical data. Raw AIS carries a USCG no-retransmit / no-fee condition | Primary class-conditional benign-behavior corpus; small-craft Class-B coverage. Cite NOAA OCM. |
| **Danish Maritime Authority AIS** | AIS archive | Free historical open data under the Danish PSI act (Act 596 of 24 Jun 2005); no re-identification of individuals; provided without warranty | Additional/European benign-behavior coverage. Historical CSV only (live feed is a paid subscription). |
| **Global Fishing Watch** | Behavior-labeled tracks | **CC BY-NC 4.0 — non-commercial**; attribution required; free account + terms acceptance | Extends fishing-behavior coverage. See non-commercial note below. |
| **Video-derived tracks** | Tracks extracted from shore EO video (SMD on-shore) | Derived from SMD (see its terms) | Non-cooperative motion check; comparison to the AIS world-frame envelope is conditional on defensible camera calibration. |

**AIS is used offline only** — to learn the benign-behavior model from historical archives. It is never a runtime input to the defense (see `THREAT_MODEL.md`). Redistribute **derived behavior features**, not bulk re-hosted AIS feeds — this also satisfies the USCG no-retransmit condition on MarineCadastre raw AIS.

**Non-commercial constraint (Global Fishing Watch).** GFW data is CC BY-NC 4.0. Any released feature derived from GFW is non-commercial and must attribute Global Fishing Watch. If a fully permissive (commercial-friendly) release is desired, the GFW-derived features must be separable/omittable; the AIS-archive-derived behavior model does not carry this restriction.

**Non-commercial + no-derivatives (McShips).** McShips is governed by a **CC BY-NC-ND** license (or an equivalent academic research license): attribution required, **non-commercial** use only, and **no derivatives** of the dataset may be redistributed. Practical consequences for this repo:
- Do **not** re-host McShips imagery.
- Training / evaluation on McShips locally for academic research is in scope.
- Public release of **adapted** McShips annotation products (e.g. remapped COCO exports) may conflict with ND — keep McShips-derived annotation files **separable/omittable** from any permissive release slice, and prefer shipping code + regenerate instructions over redistributing McShips derivatives. Attribute McShips whenever results depend on it.

## Notes
Licenses for third-party sources marked "See original dataset terms" are governed
entirely by their respective providers; consult each source's official distribution
for current terms before use or redistribution. McShips, ABOShips, MarineCadastre,
DMA, and GFW terms above were recorded from provider documentation (Jul 2026).

## Citations
Please cite McShips, SeaShips, SMD, and ABOShips per their respective providers when using those sources; cite MarineCadastre, the Danish Maritime Authority, and Global Fishing Watch per their data-use terms when using the trajectory corpora.
