# Engagement geometry — design rationale

Shared defended-asset definition for the class-conditional behavior scorer's
**asset-relative** feature family. Config: `configs/defense/engagement_geometry.yaml`.
This note records *why* the asset channel exists and how it differs from the
excluded `range_to_shore` feature — so the distinction does not have to be
re-argued ad hoc in every report.

## What is locked here

- Defended asset as a WGS-84 point (no extent model in v1).
- Engagement / detection annulus (geometry features abstain outside it).
- Inbound-leg definition relative to closest point of approach (CPA).
- Asset-placement sampling policy for false-alarm validation — four
  defended-asset archetypes (berth approach, anchorage, offshore terminal,
  fairway stress) × multiple ports.
- Train encounter pairing: same materialized placements table as FAR;
  region-only pairing; envelopes fit only at realistic asset placements;
  geometry abstain (never impute) when gates fail.
- Units and cadence assumptions matching the AIS track contract.

Feature formulas themselves live in the scorer feature contract (v2); the
adversary motion model consumes the same asset definition so both arms of
the defense and the cost-curve sweep share one geometry.

## Static geography vs. evolving relationship to a defended point

The kinematics feature contract (`configs/defense/scorer_features.yaml`)
deliberately excluded `range_to_shore` under the rule **geography ≠
behavior**: a vessel's absolute position (or distance to a fixed shoreline
polyline) is not a class-conditional motion pattern, and the corpus has no
consistent shoreline product across regions.

Asset-relative features are a different object. They describe how the
**trajectory evolves relative to a designated defended point** — closing
rate, bearing-rate stability, min range / CPA, inbound-leg persistence.
Those are encounter kinematics (CPA / TCPA and constant-bearing-decreasing-
range constructs from COLREGS / VTS practice), not a static location tag.
Two vessels at the same range to shore can have opposite intent relative to
an asset; two vessels with the same SOG can be transiting past vs.
intercepting. The scorer still asks a class-consistency question: *do real
vessels of the asserted class approach a defended asset like this?*

## Why the class channel stays

A geometry-only "is this an intercept?" detector would be classic track-
anomaly / intent detection and would not consume the EO-asserted class —
orphaning the attack half of the project. Empirically, intercept-like
geometry is also **class-dependent** on real AIS (routine 13–16 kn fairway
transits look intercept-like under a loose threshold for merchant ships and
not for sailing / fishing). The asserted class is what normalizes the
geometry; see the design pilot recorded in the internal writeup.

## Placements are asset archetypes

Each placement class names the kind of thing that is actually attacked, so
every asset position can answer "what is this defending?":

| Class | Represents | Role |
|---|---|---|
| `berth_approach` | Ship alongside a pier / quay | fit |
| `anchorage` | Ship at anchor in a holding area | fit |
| `offshore_terminal` | SPM buoy, platform, isolated facility | FAR only |
| `fairway_stress` | **Not an asset** — pessimistic bound | FAR only |

Envelopes are fit only at the realistic archetypes (`berth_approach`,
`anchorage`). `fairway_stress` is retained because it is the *hardest*
placement — benign traffic has every legitimate reason to pass within a
few hundred metres — but nothing is permanently moored in a navigation
channel, so it is labelled a stress case and excluded from the fit.

## Placement sensitivity (do not hide it)

False-alarm rate for geometry features depends on where the asset sits.
Putting the asset ashore or in empty water produces a degenerate null
result. Headline FAR is therefore a **distribution over the locked
placement policy**, never a single coordinate. Degenerate placements are
rejected by the validity gates in the config.

Note the two failure modes are different: an *inland* point (the pilot's
first attempt, on a grid corner) is degenerate because nothing approaches
it, whereas a genuine berth is well populated — most of its annulus
contacts are simply moored, and abstain.

## Train pairing vs. FAR (same placements table)

Extraction is a pure function: track points + named asset → geometry
features or abstain. Train-time **encounter pairing** is separate corpus
construction:

- Materialize the placement policy once (seed ports × placement classes),
  digest it, and reuse that table for both envelope fit and FAR — no
  ad hoc re-placement between train and eval.
- Pair a track only with placements in the same seed port region; emit a
  row only when annulus / inbound-leg gates pass. Tracks that never enter
  an annulus stay kinematics-only (geometry masked).
- **Envelopes are fit only at `role: fit` placements** — the realistic
  asset archetypes (berth approach, anchorage). `far_only` placements stay
  in the table and are scored as false-alarm sensitivity, not as a second
  training policy. Holdout is by track split, not by asset.

## Cadence limit

AIS points are thinned to ~60 s. Bearing-rate and short-horizon closing
estimates are coarse; coverage and estimation sensitivity are reported
rather than cured by inventing denser samples at score time.

## Out of scope (here)

- Contact↔track association (assumed given; see `configs/defense/pipeline.yaml`).
- Multi-asset / moving-asset defense.
- Replacing the kinematic feature family — kinematics-only remains the
  frozen ablation arm; geometry is additive and class-conditional.
