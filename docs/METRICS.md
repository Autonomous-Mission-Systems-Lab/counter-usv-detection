# Evaluation Metrics

Precise definitions of every metric reported in this project. Prior work in this area is hampered by inconsistent attack-success definitions; this document fixes every success condition so results are reproducible and comparable.

**Research questions (three).** RQ1 = attack feasibility (evasion vs. disguise). RQ2 = defense against disguise. RQ3 = adaptive-adversary cost. Marine-EOT robustness and black-box transfer are **conditions on RQ1**, not separate questions.

## Detector performance
- **Clean mAP** — mean average precision on clean (unattacked) EO evaluation data, COCO-style (mAP@[.5:.95] primary, mAP@.5 reported). Computed with `pycocotools`.

## Attack success (RQ1 — attack feasibility)
Both metrics below feed **RQ1**, which measures and **compares** the feasibility of the two attack routes against the raw EO detector. They have different success conditions, so the comparison is reported as a *relative* path-of-least-resistance statement, not a single scalar. Both are scored under the marine-EOT distribution and at white / grey / black-box access levels so feasibility is not a white-box artifact.
- **Evasion success rate (ESR)** — the evasion-route feasibility metric. Fraction of attacked instances where the target's true-class detection is suppressed below the detection confidence threshold (a contact detected when clean becomes undetected). Report the confidence/IoU thresholds used. Scope: full non-detection only (partial evasion / kinematic-layer track-starvation are out of scope).
- **Targeted-misclassification success rate (TMSR)** — the disguise-route feasibility metric. An attacked hostile instance counts as a success **iff**: (a) the contact is still detected (box IoU ≥ 0.5 with ground truth), **and** (b) the predicted class is a *benign* class (fishing / recreational), **and** (c) the clean prediction on the same instance was the correct *hostile* class. This isolates hostile→benign flips from generic label noise.

## Defense performance (RQ2 — defense against disguise; evaluated on attacked + clean data)
The defense is **class-conditional behavior consistency**: EO-asserted class vs. observed behavior. Behavior is scored in **two feature arms**:
- **kinematics_only** — frozen track kinematics (speed, straightness, loiter, turn rate, …).
- **kinematics_geometry** — kinematics plus **asset-relative intent geometry** (closing rate, bearing-rate stability, min range / CPA, …) relative to a designated defended asset (`configs/defense/engagement_geometry.yaml`, `docs/ENGAGEMENT_GEOMETRY.md`).

- **Defense detection rate (DDR)** — fraction of successful attacks (per the above) that the defense flags as inconsistent. Reported under **two attack conditions** and **both feature arms**:
  - **Perfect-disguise oracle condition** (headline) — the defense is told the contact carries the target benign class, with no patch involved. The oracle DDR is the defense's **floor against visual disguise of any kind**.
  - **Patch condition** — the benign label is induced by an adversarial patch that satisfies TMSR. Decomposes as oracle performance × patch reliability; with TMSR ≈ 0 this table is deferred rather than forced empty.
- **False-alarm rate (FAR)** — fraction of **clean, benign** contacts the defense incorrectly flags as inconsistent. Report on held-out AIS tracks. For the geometry arm, report FAR as a **distribution over the locked asset-placement policy** (never a single asset). Video-derived non-cooperative FAR is **conditional** on a defensibly world-projected SMD subset; otherwise retain AIS carriage bias as an unresolved limitation (v1 does not require a dedicated SMD FAR campaign).
- The DDR/FAR tradeoff is reported as a curve over the consistency-score threshold (the "at what false-alarm cost" question).
- **Per-class DDR** — oracle defense detection rate for each mimicked benign class × feature arm (the RQ2 per-class readout). Slow/pattern-bound classes vs. intrinsically fast classes are reported from those rates; the kinematics-only vs. kinematics+geometry contrast is part of the answer. Real-track label-swap is the synthesis-free cross-check (not a second lane taxonomy).

## Adaptive adversary (RQ3)
- **Adaptive-adversary cost curve** — defense detection rate as a function of the **operational cost** imposed when the hostile craft mimics the spoofed benign class. Primary derived axis: added approach time \(\Delta t_\mathrm{add}=R(1/v_\mathrm{mimic}-1/v_\mathrm{max})\) with \(R=R_\mathrm{start}-R_\mathrm{commit}\) (**nm**), \(v\) in **kn**, \(\Delta t\) in **h** (report in **min**); \(v_\mathrm{max}\) = platform burst; unconstrained → 0. Companion axes: capped speed, commit range, approach-bearing offset. Also report **warning time / standoff at first flag** (\(t_\mathrm{flag}\) from annulus entry; \(R_\mathrm{flag}\) = range at flag, **nm**) via causal checkpoints (this absorbs the observation-window / early-flag readout for v1). Reported **per mimicked benign class** and **per feature arm**. Swept **under the perfect-disguise oracle**. Patch-conditioned slice deferred while TMSR ≈ 0 (decomposition = oracle DDR × patch reliability).
- Under kinematics-only, the fast-craft lane may impose little cost (envelope-admissible motion); under the geometry arm, evasion of the check requires changing approach geometry — that contrast is a result, not a failure mode to hide.

## Latency
- **Time-to-flag vs. false-alarm** — defined as DDR/FAR vs observation-window length. For v1 this is **not a separate campaign**; it is folded into the RQ3 warning-time / cost-curve readout above.

## Robustness (RQ1 condition)
- **Marine-EOT survival** — attack success as a function of each marine-EOT transform axis (scale, rotation, motion blur, glare, spray, grazing angle, sea state); reported from existing severity tables (aggregate and per-axis where already computed). This is a **robustness condition on RQ1**, not a separate research question — no new per-axis attack×defense campaign is required for v1.

## Transfer (RQ1 condition)
- **Transfer slice** — attack success when optimized on a surrogate and evaluated on a held-out target family; reported separately and explicitly as a thin slice (see `TRANSFER_PROTOCOL.md`). This is the **black-box access condition on RQ1**, not a separate research question.

## Baseline comparisons
Defense metrics are reported for the class-conditional behavior defense **(both feature arms)** and for:
- **Presence-only cross-check** (track exists but no EO detection) — the existing-fusion-defense equivalent.
- **PercepGuard-style supervised bounding-box-sequence classifier** — the closest prior art, but **not measured** in v1, and the two axes on which this defense differs from it are separated by design argument rather than by ablation. Three reasons, all stated as limitations rather than results:
  - *No corpus can pose the head-to-head.* The available video corpus has no `usv`, `fishing`, or `recreational` tracks and no hostile tracks, so the disguise pairings this work is about cannot be posed; the only scorable label swap (cargo vs. small craft) is separable at clip-disjoint AUC ≈ 0.97, which puts any defense at ceiling.
  - *A projected image-plane substitute would be circular.* Apparent target size requires vessel length, which the AIS corpus does not carry (`scripts/data/ingest_ais.py` ingests position, speed, course, heading, ship-type code and transceiver class only). Size could therefore come only from a per-class constant, which would make the range-dependence of image-plane features a property of the projection rather than a measurement. The evidence used instead comes from real detector boxes on real video: apparent loiter fraction separates cargo from small craft at Cohen's *d* ≈ 2.8, and apparent loiter is largely a function of range and camera geometry.
  - *A supervised-vs-one-class detection rate would measure the evaluation, not the defense.* With no hostile class available to train, a supervised class-vs-track check must detect the attack as predicted-vs-asserted class mismatch — a relative decision that always names some benign class, so the measured rate is set by the asserted-class policy chosen by the evaluator. A one-class envelope per class instead bounds each class absolutely, which is what makes a passing track pay in approach speed or geometry; the pooled-envelope ablation quantifies what removing those per-class bounds costs.
- **APRICOT-style anomaly / patch-detection defense** — **deferred** for v1 (not required).

## Reporting conventions
- Report the confidence and IoU thresholds used for all detection and attack-success metrics.
- Report results per detector model rather than aggregating across architectures.
- Report defense metrics under the **perfect-disguise-oracle** condition (headline) for both feature arms. Patch-condition DDR is oracle × patch reliability; with TMSR ≈ 0 do not invent a full patch×defense matrix.
- Report FAR on AIS-held-out tracks. Video-derived non-cooperative FAR is conditional on a defensibly world-projected SMD subset; otherwise retain AIS carriage bias as an unresolved limitation rather than implying it has been bounded.
- For the geometry arm, state the asset-placement policy (`configs/defense/engagement_geometry.yaml`) and report FAR as a distribution over placements.
- Hostile and adaptive trajectories are synthesized at evaluation only; the benign-behavior scorer is never trained on them. State this wherever DDR is reported.
