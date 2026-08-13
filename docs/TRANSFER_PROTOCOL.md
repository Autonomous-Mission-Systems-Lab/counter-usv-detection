# Black-box Transfer Protocol

Defines the black-box **transfer** realism check (a black-box access condition on RQ1 attack feasibility): how an attack optimized without access to the deployed model's weights is evaluated. The full multi-architecture transferability study is future work; this protocol produces a thin, honest slice and defines the procedure follow-on work can extend.

## Roles
- **Surrogate model(s):** the detector(s) the adversary has access to and optimizes the attack against.
- **Target model(s):** the deployed detector the attack is actually evaluated on, whose weights the adversary never sees.

## Split
- Train ≥3 detector families (2 YOLO scales + 1 non-YOLO).
- Designate at least one family (and/or a held-out **data source**) as the **target**, never used to optimize the transferred attack.
- Optimize attacks on the surrogate(s); evaluate on the held-out target(s).

### Locked roster (baselines)

Concrete assignment lives in [`configs/detector/families.yaml`](../configs/detector/families.yaml):

| Family | Paradigm | Transfer role |
|---|---|---|
| `yolo11s` | CNN one-stage (small) | **surrogate** |
| `yolo11l` | CNN one-stage (large) | **surrogate** |
| `rtdetr_l` | DETR transformer | **held-out target** |

Attacks for the transfer slice are crafted only on the YOLO surrogates and
evaluated on RT-DETR (CNN→transformer). RT-DETR is still trained and scored
white-box in the main results; sequestration applies only to attack *crafting*.
Training workflow (RunPod): [`docs/RUNPOD.md`](RUNPOD.md).

## Transfer success definition
- Report **ESR** and **TMSR** (see `METRICS.md`) computed on the **target** model for attacks optimized on the **surrogate**.
- Report the **transfer gap**: target-model success minus surrogate-model (white-box) success on the same attacks — how much attack strength is lost crossing models.

## Scope
- This is a **thin slice**, not a full transferability matrix, and is reported as such.
- **White-box** (upper bound): craft and score on the same surrogate weights (`yolo11s` or `yolo11l`). Already measured for ESR/TMSR in the evasion/disguise runners.
- **Grey-box** (default realistic case): architecture family known, weights unknown. Instantiated here as **cross-scale YOLO transfer** — craft on `yolo11s`, hard-eval on `yolo11l` (and the reverse). Same patches; no gradients on the target.
- **Black-box transfer** (cross-family): craft on a YOLO surrogate, evaluate on held-out `rtdetr_l`. Numbers are expected to be weaker than white/grey.
- Data-source hold-out (train on sources A/B, transfer-test on source C) guards against dataset-specific overfitting; sequence-level (video) and MMSI/region-level (AIS tracks) split discipline ensures no leakage.

## Patch bank (required for grey/black eval)

White-box craft must be re-run with `--save-patches` so each surrogate writes
`results/attacks/<attack>/<surrogate>/patch_bank/` (attacked letterbox PNGs +
patch tensors + manifest). Transfer scoring loads that bank and never
re-optimizes:

```bash
python scripts/attacks/run_evasion.py --family yolo11s --device 0 --save-patches
python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --device 0
```

Config: `configs/attacks/access_levels.yaml`. Runner: `scripts/attacks/run_transfer.py`.
RunPod: `docs/RUNPOD.md` § Access-level transfer.
