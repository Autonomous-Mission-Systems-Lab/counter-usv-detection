# Threat Model

The fixed reference threat model for all attacks and defenses evaluated in this project. Results in this repository are reported against this model.

## Asset defended
A shore-based EO counter-USV detector feeding a human operator, together with the **track** associated with each contact (from the EO tracker and/or coastal radar). The detector performs the **classify** step of the detect → classify → track → neutralize chain; the track supplies the kinematics the defense checks that class against.

## Adversary goals (both primary)
Two adversary goals are studied as first-class objectives. **RQ1** measures and **compares the feasibility of both** against the raw EO detector — evasion (ESR) vs. disguise (TMSR) — to establish the attacker's path of least resistance. Because the two have different success conditions, the comparison is a *relative* statement, not a single scalar.

- **Primary — evasion (non-detection).** Can the adversary suppress the EO detection of the USV entirely? This is a candidate *path of least resistance* against an EO-alone system: with a physically-realizable, marine-surviving perturbation, how easy or hard is it to drive the detector to miss the craft altogether (measured under RQ1)? Scope note: we study **full evasion** (the detector produces no USV detection); **partial evasion and track-starvation of the kinematic layer are out of scope** for this project.
- **Primary — targeted misclassification (disguise)** of a hostile small craft into a benign class (fishing / recreational). The contact stays present, so a naive presence check does not fire. Misclassification is the rational attack against a *layered* defense: evasion must defeat every sensor at once (an EO-evading craft still paints on coastal radar), whereas a benign label defeats the fused picture — the contact is seen, explained, and dismissed. Its raw-detector feasibility is measured under RQ1; it is the **residual threat** the class–kinematics consistency defense is built to catch (**RQ2**).
- **The misclassification goal is means-agnostic.** The benign label may be induced by an **adversarial patch** or by **ordinary visual disguise** — a hull genuinely built or dressed to resemble a benign vessel. The defense scores the asserted class against motion, so both are treated identically. The patch is studied as the ML-specific instantiation (and because a patch can flip the class while the platform retains a hostile configuration that physical disguise would preclude), but the defense's coverage extends to the zero-tech disguise adversary that a hardened detector alone cannot address.

## Division of labor across attack goals
- **Evasion** feasibility against the EO detector is quantified directly (RQ1). At the *system* level it is caught by the **presence layer**: an independent track (coastal radar) with no corresponding EO detection is itself an inconsistency — this is what existing fusion defenses already check and what the presence-only baseline implements. Note this coverage depends entirely on an independent track source; a successful evasion also kills an EO-only track, which is exactly why its feasibility is measured as a primary question rather than assumed away.
- **Misclassification (disguise)** is the case presence cannot reach: the contact *is* detected, just mislabeled. Its raw-detector feasibility is compared against evasion under RQ1; the class–kinematics check subsumes the presence check and adds the layer that catches it (RQ2).
- Kinematic scoring of *unclassified* radar contacts ("unknown contact on an attack-run profile") is a natural system-level extension, but it is classic track-anomaly detection rather than class consistency — noted, not claimed.

## Adversary knowledge
- **Default: grey-box** — detector architecture family known, weights unknown (the realistic case).
- **Upper bound: white-box** — full model access.
- **Realism check: black-box transfer** — attack optimized on a surrogate, tested on a held-out target (see `TRANSFER_PROTOCOL.md`).
- **Defense-aware (adaptive) case:** the adversary additionally knows a class–kinematics consistency check is applied, and can therefore attempt to defeat it (see capability).

## Adversary capability
- A **physically realizable** pattern applied to hull/superstructure, bounded in size and placement, required to survive a marine transform distribution (**marine-EOT**: scale, rotation, motion blur, glare, spray, grazing angle, sea state).
- **In the adaptive case, the adversary may additionally choose the craft's own motion** — which benign class to kinematically mimic and how faithfully (loiter, slow transit, fishing-like patterns). This is the realistic defense-aware attack and is a first-class part of the evaluation (RQ3 cost curve), not a footnote.
- **Digital, unconstrained perturbation** is reported only as an idealized upper bound.
- **Perfect-disguise oracle** — the strongest misclassification capability: the defense is simply *told* the contact carries the target benign class (no patch, no attack imagery). Models both an ideal patch and the zero-tech visually-disguised hull (see means-agnostic goal). This is the cleanest test of the defense because it removes patch fragility from the measurement; patch results decompose as oracle performance × patch reliability.
- All work is **digital and simulated-physical only** — no patches on real craft, no field deployment, no RF.

## EO baseline (undisguised-USV recognition)
The EO-alone system this project defends must be able to recognize an **undisguised** USV — otherwise the EO-alone vs. EO+consistency comparison would be unfair (a detector that never recognizes USVs has nothing for the defense to improve on). The fielded-representative baselines satisfy this: on the undisguised-`usv` test slice they localize the craft (**presence recall 91–96%**) and correctly label it `usv` (**recognition recall 84–96%**, `usv` AP@50 0.84–0.93), well above a presence-only floor and bracketed above by the perfect-EO oracle supplied at eval. See `results/detector_baselines/usv_capability.md` (generated by `scripts/detector/usv_capability.py`).

## Out of scope
- Sensor tampering.
- GPS / RF spoofing.
- **AIS spoofing.** AIS is used in this project **only offline**, to learn the benign-behavior model from historical archives — it is **never a runtime input** to the defense. Runtime contact kinematics come from the EO tracker / radar. Therefore runtime AIS spoofing cannot affect the defense and is out of scope by construction.
- Any intrusion into the C2 network or the sensor→inference pipeline.

This is strictly a perception-layer attack on the detector's inputs, plus — in the adaptive case — the craft's own kinematics.

## Why kinematics is the consistency check
The defense checks the EO-asserted **class** against the contact's observed **motion**, scored against a **class-conditional benign-behavior model learned from large-scale real trajectory data** (AIS archives; fishing-behavior classifiers). Two properties make this the chosen design over the earlier EO/IR thermal variant:
- **Real-data grounding, no by-construction circularity.** The benign side of the check is learned from real tracks and **no hostile data is used in scorer training**; hostile/adaptive trajectories appear only at evaluation. The claim is "this track is inconsistent with the real benign class model," not "we recognize attacks we designed." (The EO/IR variant, by contrast, had to author hostile thermal signatures in a simulator and then detect the planted mismatch — see the deferred sim spec.)
- **Independence from the attacked channel.** The EO patch perturbs pixels, not the physics of motion; the track is an independent observable. To defeat the check the adversary must physically change how the craft moves — the RQ3 adaptive-adversary cost that the evaluation quantifies.
