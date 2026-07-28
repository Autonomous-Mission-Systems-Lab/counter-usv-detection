# Evaluation Metrics

Precise definitions of every metric reported in this project. Prior work in this area is hampered by inconsistent attack-success definitions; this document fixes every success condition so results are reproducible and comparable.

## Detector performance
- **Clean mAP** — mean average precision on clean (unattacked) EO evaluation data, COCO-style (mAP@[.5:.95] primary, mAP@.5 reported). Computed with `pycocotools`.

## Attack success (RQ1 — attack feasibility)
Both metrics below feed **RQ1**, which measures and **compares** the feasibility of the two attack routes against the raw EO detector. They have different success conditions, so the comparison is reported as a *relative* path-of-least-resistance statement, not a single scalar. Both are scored under the marine-EOT distribution and at the grey-box / black-box access levels so feasibility is not a white-box artifact.
- **Evasion success rate (ESR)** — the evasion-route feasibility metric. Fraction of attacked instances where the target's true-class detection is suppressed below the detection confidence threshold (a contact detected when clean becomes undetected). Report the confidence/IoU thresholds used. Scope: full non-detection only (partial evasion / kinematic-layer track-starvation are out of scope).
- **Targeted-misclassification success rate (TMSR)** — the disguise-route feasibility metric. An attacked hostile instance counts as a success **iff**: (a) the contact is still detected (box IoU ≥ 0.5 with ground truth), **and** (b) the predicted class is a *benign* class (fishing / recreational), **and** (c) the clean prediction on the same instance was the correct *hostile* class. This isolates hostile→benign flips from generic label noise.

## Defense performance (RQ2 — defense against disguise; evaluated on attacked + clean data)
- **Defense detection rate (DDR)** — fraction of successful attacks (per the above) that the defense flags as inconsistent (EO-asserted class vs. track kinematics). Reported under **two attack conditions**:
  - **Patch condition** — the benign label is induced by an adversarial patch that satisfies TMSR.
  - **Perfect-disguise oracle condition** — the defense is told the contact carries the target benign class, with no patch involved. The oracle DDR is the defense's **floor against visual disguise of any kind**; patch-condition results decompose as oracle performance × patch reliability.
- **False-alarm rate (FAR)** — fraction of **clean, benign** contacts the defense incorrectly flags as inconsistent. Report on held-out AIS tracks. Also report a **video-derived non-cooperative FAR** only for SMD features that pass the world-frame calibration/uncertainty gate; otherwise report SMD as a separate scale-normalized image-plane sensitivity check rather than applying the AIS metric model to incomparable coordinates.
- The DDR/FAR tradeoff is reported as a curve over the consistency-score threshold (the "at what false-alarm cost" question).

## Adaptive adversary (RQ3)
- **Adaptive-adversary cost curve** — defense detection rate as a function of the **operational cost** imposed when the hostile craft kinematically mimics the spoofed benign class. Cost axes: added approach time, capped speed, restricted approach geometry, exposure window. Reported **per mimicked benign class**. Swept **under the perfect-disguise oracle** so the curve measures the kinematic defense alone, not patch fragility; a patch-conditioned slice is reported as the realism check.
- **Per-class discriminability** — for each benign class the adversary could spoof, whether kinematics effectively close off that disguise (slow/pattern-bound classes) or leave it open (intrinsically fast classes). Reported as a table tied to the threat's alarm-suppression logic.

## Latency
- **Time-to-flag vs. false-alarm** — because kinematic consistency needs track history, DDR and FAR are reported as a function of the **observation-window length** (how much track the defense has seen). Frames the operational time-budget: how early the inconsistency can be called, and at what false-alarm cost.

## Robustness
- **Marine-EOT survival** — attack success as a function of each marine-EOT transform axis (scale, rotation, motion blur, glare, spray, grazing angle, sea state); reported as per-axis curves, not just an aggregate.

## Transfer
- **Transfer slice** — attack success when optimized on a surrogate and evaluated on a held-out target family; reported separately and explicitly as a thin slice (see `TRANSFER_PROTOCOL.md`).

## Baseline comparisons
All defense metrics (DDR, FAR, time-to-flag, adaptive cost curve) are reported for the class–kinematics defense **and** for each baseline on the same data and thresholds:
- **Presence-only cross-check** (track exists but no EO detection) — the existing-fusion-defense equivalent.
- **APRICOT-style anomaly / patch-detection defense** (kernel-density / Bayesian-uncertainty).
- **PercepGuard-style supervised bounding-box-sequence classifier** (retrained on maritime tracks) — the closest prior art; this column doubles as the design ablation (image-plane bbox features vs. world-frame kinematics; supervised vs. one-class real-data training), including its behavior under the pixel-level adaptive attack.

## Reporting conventions
- Report the confidence and IoU thresholds used for all detection and attack-success metrics.
- Report results per detector model rather than aggregating across architectures.
- Report defense metrics under both the patch and perfect-disguise-oracle conditions (see DDR).
- Report FAR on AIS-held-out tracks and, conditionally, on the defensibly world-projected subset of video tracks. If SMD cannot be projected reliably, report its image-plane results separately and retain AIS carriage bias as an unresolved limitation rather than implying it has been bounded.
- Hostile and adaptive trajectories are synthesized at evaluation only; the benign-behavior scorer is never trained on them. State this wherever DDR is reported.
