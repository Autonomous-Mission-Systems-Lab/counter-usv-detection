# Maritime EO detection dataset — data card

A harmonized electro-optical (EO) maritime **object-detection** dataset: several public
vessel-detection corpora merged into one COCO-format annotation set with a shared class
taxonomy, plus a small **curated unmanned-surface-vehicle (USV) class**. Suitable for
training and evaluating maritime vessel/USV detectors.

This repository **does not re-host** the original imagery. Obtain each source from its
provider (see `docs/DATA_LICENSES.md`), place it under `data/raw/`, and regenerate the
derived annotations with the scripts in `scripts/`. Checksums for the derived release
are in `CHECKSUMS.derived.sha256`; the version stamp is `RELEASE.json`.

**Release policy.** Redistributed products are **annotations, split manifests,
audit/eval manifests, the taxonomy, and this card** — never raw imagery. Source terms
govern (see `docs/DATA_LICENSES.md`). McShips has **no stated license** from the authors'
distribution: do not re-host its imagery, and **omit** McShips-derived annotation
products from the permissive release slice (regenerate locally via `fetch_data.py`).

---

## 1. Summary

| Field | Value |
|---|---|
| **Task** | 2-D object detection (axis-aligned boxes) of vessels and USVs in EO imagery |
| **Format** | COCO (`annotations/coco_master.json`) + per-source COCO files |
| **Images / boxes** | **25,262 images / 71,501 boxes** |
| **Classes** | Unified vessel taxonomy + a curated `usv` class (see `taxonomy.yaml`) |
| **Maintainer** | Academic research (AMSL / Duke) |

---

## 2. Composition

### 2.1 Sources

| Source | Images | Boxes | Viewpoint | License (summary) |
|---|---:|---:|---|---|
| SeaShips | 7,000 | 9,221 | Shore coastline surveillance | Academic; CC BY 4.0 (mirror) |
| McShips (9k lite) | 7,996 | 11,330 | Web / in-the-wild | No stated license (citation requested) |
| ABOShips (original) | 9,041 | 41,967 | Onboard moving vessel | CC BY 4.0 (Zenodo 4736931) |
| SMD on-shore (sampled) | 964 | 8,683 | Fixed shore platform | Academic / research |
| Curated `usv` | 261 | 300 | Shore (curated web/video stills) | Per-image provenance; link-only |
| **Master** | **25,262** | **71,501** | — | — |

Canonical classes and per-source native→canonical label maps: `taxonomy.yaml`.
An optional coarse **role axis** (e.g. military/USV vs. civilian classes) is provided in
`taxonomy.yaml` for downstream tasks that need it; it is not required to use the boxes.

### 2.2 The curated `usv` class

261 images / 300 boxes of unmanned surface vehicles, curated from public web and video
stills with **per-image provenance** (platform + source URL in `raw/usv/manifest.csv`).
Single `usv` label; every retained image has ≥1 box (empty images are dropped at build).
This is **appearance/detection imagery only** — it carries no motion or trajectory data.
Viewpoint is predominantly shore/oblique; treat it as a small, provenance-tracked class,
not a balanced benchmark.

### 2.3 Size-eligibility flags (non-destructive)

Produced by `scripts/data/audit_eo.py`; nothing is deleted from the master.

- **`det_eligible` — shortest box side ≥ 8 px:** 67,397 / 71,501 = **94.3%**. Below this,
  boxes are noise-dominated and detection is unreliable.
- **`patch_eligible` — shortest box side ≥ 32 px:** 43,761 / 71,501 = **61.2%**. A stricter
  flag for tasks that require larger targets.
- Full sweeps at det {4,8,16,24,32} and {16,24,32,48,64} px are recorded in the audit
  summary so a consumer can pick a different floor.

Native resolution is retained on disk; a load-time letterbox + box-remap contract (and a
fixed-region overlay mask for SeaShips' burned-in timestamp band) is documented in
`HARMONIZATION.md`.

### 2.4 Splits (leakage-controlled)

`data/splits/eo_image_splits.csv` — per-image train/val/test = **22,804 / 1,229 / 1,229**.

- **Group-intact assignment:** exact + perceptual near-duplicates, source sequences
  (SMD clips, ABOShips recording-days) and curated-USV source clips are unioned into
  leakage groups; a whole group lands in one split. Provider splits are not trusted.
- **Viewpoint stratification:** shore-viewpoint sources (SeaShips, SMD) plus the `usv`
  class are eval-eligible; the onboard/web sources (ABOShips, McShips) are **train-only**,
  enabling a with/without-auxiliary training ablation.
- Leakage check **PASS** (0 groups spanning splits). Details: `results/splits/report.md`.
- **Limitation:** the `recreational` class is 100% onboard (train-only) and therefore
  has no shore-viewpoint val/test representation.

### 2.5 Small-craft subset

A manifest over the master selecting the small-craft classes (`small_craft`,
`recreational`, `fishing`, `sailing`): **25,970 instances / 8,960 images**. Each row is
viewpoint-tagged; only **~12%** are shore-viewpoint (the rest onboard). Files under
`eval_slices/`. Useful for a small-object / small-craft evaluation slice.

---

## 3. Collection & processing

| Step | Script / artifact |
|---|---|
| Acquire sources | `scripts/data/fetch_data.py` → `data/raw/` |
| Curate `usv` | `scripts/data/collect_usv.py` (web/video stills + provenance + QC + CVAT) |
| Harmonize labels | `scripts/data/build_coco_master.py` → `annotations/` |
| Size/QA audit | `scripts/data/audit_eo.py` → `audit/` |
| Small-craft subset | `scripts/data/build_eval_slice.py` → `eval_slices/` |
| Splits | `scripts/data/build_splits.py` → `splits/` |
| Package / checksum | `scripts/data/package_derived.py` |

Labels come from each provider's native annotations mapped through
`taxonomy.yaml` (`eo_sources.*.native`); curated `usv` boxes are drawn in CVAT.

---

## 4. Uses

| Use | Recommended data |
|---|---|
| Vessel/USV detector training | Full `train` split (includes onboard/web auxiliaries) |
| Shore-viewpoint evaluation | Shore val/test slice (SeaShips + SMD) + `usv` test |
| Viewpoint-generalization ablation | Train with/without the onboard auxiliary; compare on the shore slice |
| Small-object evaluation | Small-craft subset; filter by `det_eligible` / `patch_eligible` |

---

## 5. Distribution

| Item | Status |
|---|---|
| Raw imagery | **Not redistributed** — obtain from providers |
| Derived annotations, manifests | Released (checksummed); source terms apply |
| Curated `usv` images | Provenance/link in `raw/usv/manifest.csv`; do not re-host unless a source license allows |
| License summary | `docs/DATA_LICENSES.md` |

**Versioning.** `RELEASE.json` + `CHECKSUMS.derived.sha256`; regenerate after any
pipeline change. Prefer regenerating derived artifacts over hand-editing JSON/CSV.

---

## 6. Known limitations

1. **Curated `usv` set is small** (~38 val / ~38 test images) and viewpoint-skewed —
   report `usv` metrics with variance in mind; it is not a balanced benchmark.
2. **Viewpoint mismatch across sources** — onboard (ABOShips) and web (McShips) imagery
   differ from shore surveillance; kept train-only so evaluation stays shore-representative.
3. **Class imbalance** — large-vessel classes dominate; several small-craft classes and
   `usv` are thin. Report per-class AP.
4. **`recreational` has no shore-viewpoint eval** (100% onboard/train-only).
5. **McShips has no stated license** — train-only; omit derived annotations from the
   permissive release and never re-host imagery (see `docs/DATA_LICENSES.md`).
6. **USV provenance licenses pending backfill** — platform + source URL are recorded;
   full per-image license/attribution fields may still be empty in the manifest.

---

## 7. Citation

Cite each upstream source per `docs/DATA_LICENSES.md` (SeaShips, McShips, SMD, ABOShips)
and this dataset when the curated `usv` class or the harmonized annotations are used.
