# RunPod — EO detector training + evasion (ESR) GPU runs

Single-GPU CUDA jobs for (1) the three detector families in
`configs/detector/families.yaml` (`yolo11s`, `yolo11l`, `rtdetr_l`) and
(2) white-box evasion crafting / ESR scoring on the `usv` test slice. Author and
smoke-check on a laptop; run full training and full ESR sweeps on RunPod.

## Why RunPod (not a laptop GPU)

Full training is ~22.8k images × 100 epochs × 3 families. An M-series Mac (MPS)
is too slow and unreliable for RT-DETR; a rented
24 GB CUDA GPU finishes the suite in roughly a day of wall-clock for ~$10–30.

## What to provision

| Setting | Recommendation |
|---|---|
| GPU | **24 GB** class (RTX 4090 / L4 / A5000). 16 GB is tight for RT-DETR + batch 8; 24 GB is comfortable at the 640 training resolution. |
| Template | **Official RunPod PyTorch + CUDA** (torch preinstalled — setup reuses it, no ~3 GB reinstall) |
| **Volume Disk** (`/workspace`) | **≥ 40 GB** — this is where the ~10 GB EO bundle, venv, pip cache, and weights live. Undersizing this is the usual cause of `Disk quota exceeded`. |
| Container disk | Default is fine; setup routes pip cache + build tmp to `/workspace`, not the container disk |
| Network volume | Optional; use it if you want data to survive **Terminate** (not just Stop) |

SSH into the pod and note the host / port / key RunPod shows in the UI — you will
need them for `rsync`.

## Bundle size (~10 GB)

Only master-referenced EO imagery is synced. **Not** included: AIS (~6.7 GB),
SMD videos, legacy trees. Path list:
[`configs/detector/runpod_eo_paths.txt`](../configs/detector/runpod_eo_paths.txt).

| Path | Role |
|---|---|
| `data/raw/{aboships/Seaships,mcships/JPEGImages,seaships/JPEGImages,smd/frames,usv/{scraped,frames}}/` | pixels |
| `data/annotations/`, `data/splits/` | COCO master + leakage-controlled splits |
| `configs/`, `scripts/`, `src/`, `requirements*.txt` | code |

## Laptop → pod sync

Run these commands on your **Mac** (not inside the pod SSH session).

RunPod exposes two SSH styles; **prefer Direct TCP for rsync** (the
`ssh.runpod.io` proxy often breaks `rsync` with PTY / protocol errors).

### Option A — Direct TCP (recommended for sync)

In the pod **Connect** panel, copy the **IP + port** SSH line (usually
`ssh root@<IP> -p <PORT> -i ...`), then on the Mac:

```bash
cd /Users/jonreifschneider/Duke/Research/AMSL/counterUSV
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
./scripts/detector/sync_eo_train_bundle.sh root@<IP>:/workspace/counterUSV
```

### Option B — Proxy (`…@ssh.runpod.io`)

Works for interactive `ssh`. For sync, force no-TTY (`-T`); if you still see
`unexpected tag` / `doesn't support PTY`, switch to Option A.

```bash
export RSYNC_RSH='ssh -T -i ~/.ssh/runpod -o RequestTTY=no'
./scripts/detector/sync_eo_train_bundle.sh <poduser>@ssh.runpod.io:/workspace/counterUSV
```

First sync is dominated by ABOShips (~8 GB). Later syncs are incremental.
If the pod image lacks `rsync`, the sync script tries to `apt-get install rsync`
automatically; you can also install it once over SSH:

```bash
ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no root@<IP> \
  'apt-get update -qq && apt-get install -y rsync'
```

Alternative: upload a tarball of the same paths to a Network Volume / object
store and extract on the pod — same contents as the path list.

## On-pod setup (once per machine)

```bash
cd /workspace/counterUSV
bash scripts/detector/setup_runpod_eo.sh
```

This creates `.venv`, installs deps, **re-exports** `data/eo_views/` so image
symlinks resolve on the pod (Mac absolute symlinks are useless after sync), and
runs `train_detector.py --dry-run`.

**Always re-run `python scripts/data/export_eo_views.py` after syncing raw imagery.**
The exporter writes an **absolute** `path` in `data.yaml` (Ultralytics resolves
relative paths against CWD, not the yaml location) and regenerates machine-local
symlinks.

## Train (must survive SSH disconnect)

**Do not** run a multi-hour train in a bare SSH session. When the laptop sleeps,
the network drops, or `ssh.runpod.io` resets, the shell gets SIGHUP and kills
training. Always start under **tmux** (preferred) or `nohup`.

### Preferred: tmux

```bash
# Once per pod image if missing:
apt-get update -qq && apt-get install -y -qq tmux

cd /workspace/counterUSV
source .venv/bin/activate
tmux new -s train

# Inside the tmux session:
python scripts/detector/train_detector.py --all
# Detach without stopping training:  Ctrl-b then d
# Leave the laptop; reconnect later and attach:
#   tmux attach -t train
# List sessions:  tmux ls
```

Useful while detached (from a **new** SSH shell — do not Ctrl-C the train tty):

```bash
nvidia-smi                                          # GPU busy?
ps aux | grep '[t]rain_detector'
tail -n 5 results/detector_baselines/*/results.csv  # last epochs
df -h /workspace                                    # stay under volume quota
```

### Fallback: nohup

```bash
cd /workspace/counterUSV && source .venv/bin/activate
nohup python scripts/detector/train_detector.py --all > train.log 2>&1 &
tail -f train.log
```

### Other train invocations (still run inside tmux / nohup)

```bash
# One family
python scripts/detector/train_detector.py --family yolo11s
python scripts/detector/train_detector.py --family yolo11l
python scripts/detector/train_detector.py --family rtdetr_l

# Pipeline sanity (1 epoch, 2% of data) — also useful on a laptop
python scripts/detector/train_detector.py --family yolo11s --smoke

# Resume after kill / disconnect (picks up last.pt for an interrupted family)
python scripts/detector/train_detector.py --all --resume
```

Outputs land under `results/detector_baselines/<family>/`:

- `weights/best.pt`, `weights/last.pt`
- Ultralytics `results.csv`, plots, `args.yaml`
- `run_meta.json` — wall-clock, GPU name, config snapshot, transfer role

Aggregate: `results/detector_baselines/train_summary.json` (written when `--all` finishes).

## Transfer roles (unchanged by training)

All three families train identically. Roles in
`configs/detector/families.yaml` only constrain **later attack crafting**:

- **Surrogates** (`yolo11s`, `yolo11l`) — allowed for optimizing transfer attacks.
- **Held-out target** (`rtdetr_l`) — sequestered from attack optimization; still
  trained and evaluated white-box. See `docs/TRANSFER_PROTOCOL.md`.

## Pull results back

From the **repo root on your laptop** (same Direct TCP `RSYNC_RSH` as sync):

```bash
cd /Users/jonreifschneider/Duke/Research/AMSL/counterUSV
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
rsync -avh --no-owner --no-group -e "$RSYNC_RSH" \
  root@<IP>:/workspace/counterUSV/results/detector_baselines/ \
  results/detector_baselines/
```

Weights are gitignored (`*.pt`); keep them on disk / a volume / object storage.

Then run the clean-mAP report on the laptop (or on the pod before tearing down):

```bash
python scripts/detector/eval_detector_clean.py --all          # 640 (training resolution)
# → results/detector_baselines/report.md
# → results/detector_baselines/clean_map/<family>_s640.json
```

Evaluated at the 640 training resolution. Off-resolution eval (e.g. 1280) is not
run: it penalises resolution-sensitive architectures (RT-DETR degrades sharply
far off its training size) without a matching 1280-trained model to compare.

## Evasion attack (ESR) on RunPod

Full white-box evasion on the `usv` test slice (~38 patch-eligible targets × 150
optimize steps × marine-EOT expectation) is a GPU job. On CPU it is ~30–40 hours;
on a 24 GB CUDA pod it should finish in **tens of minutes**.

Same provisioning advice as detector training (24 GB class, Official PyTorch
template, ≥40 GB volume). Prefer **reusing** a pod that already has the EO train
bundle + `.venv` from detector setup.

### Laptop → pod (evasion overlay)

EO imagery must already be on the pod. Then push attack code + frozen **surrogate**
weights only (~70 MB; path list
[`configs/attacks/runpod_evasion_paths.txt`](../configs/attacks/runpod_evasion_paths.txt)):

```bash
cd /Users/jonreifschneider/Duke/Research/AMSL/counterUSV
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'

# Fresh pod only — skip if EO bundle + setup_runpod_eo.sh already done:
./scripts/detector/sync_eo_train_bundle.sh root@<IP>:/workspace/counterUSV
# (then on pod: bash scripts/detector/setup_runpod_eo.sh)

# Always (code + yolo11s/yolo11l best.pt + FROZEN.json):
./scripts/attacks/sync_evasion_runpod.sh root@<IP>:/workspace/counterUSV
```

`FROZEN.json` stores absolute Mac paths; `DetectorBaseline.from_freeze` falls
back to `weights_rel` under the repo root when those paths are missing, so the
same freeze works on the pod after the weights sync.

### On-pod setup / smoke

```bash
cd /workspace/counterUSV
bash scripts/attacks/setup_runpod_evasion.sh
```

Checks CUDA, loads the frozen surrogate via `weights_rel`, runs a tiny CUDA
autograd forward, and `--dry-run`s the evasion slice.

### Full ESR run (must survive SSH disconnect)

```bash
apt-get update -qq && apt-get install -y -qq tmux   # once if missing
source .venv/bin/activate
tmux new -s evasion

# Inside tmux — default recipe (yolo11s, 150 steps, full marine-EOT ESR sweep):
python scripts/attacks/run_evasion.py --family yolo11s --device 0

# Optional short GPU smoke before committing to the full slice:
python scripts/attacks/run_evasion.py --family yolo11s --device 0 --max-images 2 --steps 30

# Detach: Ctrl-b then d. Reattach: tmux attach -t evasion
```

`--device 0` (or leave config `device: auto`) puts both the differentiable
surrogate and the hard-detector eval on CUDA. Do **not** craft against
`rtdetr_l` (held-out transfer target — refused in code).

Outputs → `results/attacks/evasion/<family>/`:

- `report.md` — base ESR + marine-EOT survival (raw + patch-attributable)
- `esr_by_severity.json`, `instances.json`
- `gallery/` — clean vs patched PNGs for the first few targets

Useful while detached:

```bash
nvidia-smi
ps aux | grep '[r]un_evasion'
tail -n 20 results/attacks/evasion/yolo11s/instances.json  # grows as targets finish
df -h /workspace
```

### Pull ESR results back

From the **repo root on your laptop** (same Direct TCP `RSYNC_RSH`):

```bash
cd /Users/jonreifschneider/Duke/Research/AMSL/counterUSV
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
rsync -avh --no-owner --no-group -e "$RSYNC_RSH" \
  root@<IP>:/workspace/counterUSV/results/attacks/evasion/ \
  results/attacks/evasion/
```

### Second surrogate (optional)

```bash
# After yolo11s finishes (same tmux or a new session):
python scripts/attacks/run_evasion.py --family yolo11l --device 0
```

Black-box transfer to `rtdetr_l` is covered under **Access-level transfer** below
(re-craft with `--save-patches`, then `run_transfer.py`).

## Disguise attack (TMSR) on RunPod

Same provisioning and EO-bundle prerequisites as evasion. Disguise crafts a
patch per target benign class (**fishing** and **recreational**), so wall-clock
is roughly **2×** a single ESR family run (~same per-image cost × 2 classes).

### Laptop → pod

```bash
cd /Users/jonreifschneider/Duke/Research/AMSL/counterUSV
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'

# Fresh pod only — skip if EO bundle + setup_runpod_eo.sh already done:
./scripts/detector/sync_eo_train_bundle.sh root@<IP>:/workspace/counterUSV

./scripts/attacks/sync_disguise_runpod.sh root@<IP>:/workspace/counterUSV
```

### On-pod setup / full run

```bash
cd /workspace/counterUSV
bash scripts/attacks/setup_runpod_disguise.sh

source .venv/bin/activate
tmux new -s disguise

# Inside tmux — both benign classes (fishing + recreational):
python scripts/attacks/run_disguise.py --family yolo11s --device 0

# Smoke / single class:
python scripts/attacks/run_disguise.py --family yolo11s --device 0 --max-images 2 --steps 30
python scripts/attacks/run_disguise.py --family yolo11s --device 0 --benign-class fishing
```

Outputs → `results/attacks/disguise/<family>/<benign>/` plus
`results/attacks/disguise/<family>/summary.md`.

### Pull TMSR results back

```bash
rsync -avh --no-owner --no-group -e "$RSYNC_RSH" \
  root@<IP>:/workspace/counterUSV/results/attacks/disguise/ \
  results/attacks/disguise/
```

## Access-level transfer (grey / black-box) on RunPod

White-box ESR/TMSR craft on a surrogate, then **hard-eval the same patches** on
other detectors (no re-optimization). Requires a **patch bank** from craft with
`--save-patches`.

| Level | Craft | Eval |
|---|---|---|
| white | yolo11s (or l) | same family (existing craft report) |
| grey | yolo11s | yolo11l (and reverse) |
| black | yolo11s or l | held-out `rtdetr_l` |

### Laptop → pod

```bash
export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
# Fresh pod: EO bundle first, then:
./scripts/attacks/sync_transfer_runpod.sh root@<IP>:/workspace/counterUSV
```

Syncs code + **all three** frozen weights (`yolo11s`, `yolo11l`, `rtdetr_l`).

### On-pod

```bash
cd /workspace/counterUSV
bash scripts/attacks/setup_runpod_transfer.sh
source .venv/bin/activate
apt-get update -qq && apt-get install -y -qq tmux rsync   # if missing
tmux new -s transfer

# 1) Re-craft with patch export (GPU; prior runs lacked patch_bank/)
python scripts/attacks/run_evasion.py --family yolo11s --device 0 --save-patches
python scripts/attacks/run_evasion.py --family yolo11l --device 0 --save-patches

# Optional disguise banks (white-box TMSR was 0%; still fills the protocol):
python scripts/attacks/run_disguise.py --family yolo11s --device 0 --save-patches --benign-class fishing

# 2) Transfer eval (forward-only; default targets = grey peer + rtdetr_l)
python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --device 0
python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11l --device 0
python scripts/attacks/run_transfer.py --attack disguise --surrogate yolo11s \
  --benign-class fishing --device 0
```

Outputs → `results/attacks/transfer/{evasion,disguise}/<surrogate>_to_<target>/`
plus per-surrogate `*_summary.md`. Transfer gap vs white-box is in each
`report.md` when the craft `*_by_severity.json` is present.

### Pull

```bash
rsync -avh --no-owner --no-group -e "$RSYNC_RSH" \
  root@<IP>:/workspace/counterUSV/results/attacks/transfer/ \
  results/attacks/transfer/
# Keep patch banks with the craft trees if you want them locally:
rsync -avh --no-owner --no-group -e "$RSYNC_RSH" \
  root@<IP>:/workspace/counterUSV/results/attacks/evasion/ \
  results/attacks/evasion/
```

## Laptop checklist before paying for GPU time

```bash
# Config / wiring only (no download-heavy train)
python scripts/detector/train_detector.py --all --dry-run

# Optional short MPS/CPU smoke (slow; catches import/path bugs)
python scripts/detector/train_detector.py --family yolo11s --smoke --device mps   # or cpu

# Evasion wiring (no optimize)
python scripts/attacks/run_evasion.py --dry-run
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `EO view not found` / broken symlinks | Re-run `python scripts/data/export_eo_views.py` on the pod after raw sync. |
| CUDA OOM | Lower `--batch` (try 16→8→4); RT-DETR is the memory hog. For evasion, batch is 1 image — OOM is unexpected at 640; free other processes (`nvidia-smi`). |
| `missing path '.../images/val'` (wrong root) | `data.yaml` `path` must be absolute to `data/eo_views/yolo`; re-export or let `train_detector.py` rewrite it. |
| Lockfile CUDA wheels fail on pod | Prefer template torch + `ultralytics`/`pycocotools`/`pandas`/`pyyaml` (see `setup_runpod_eo.sh`); don't force Mac `requirements.lock.txt` CUDA wheels. |
| Training on wrong split | Views come from `data/splits/eo_image_splits.csv`; ABOShips/McShips are train-only by construction — do not re-split on the pod. |
| `Connection reset` / `Broken pipe` mid-train | Training was in the SSH foreground. Check `nvidia-smi` / `ps`; if dead, `tmux new -s train` then `python scripts/detector/train_detector.py --all --resume`. Always use tmux/nohup for long runs. Same for evasion (`tmux new -s evasion`). |
| `Disk quota exceeded` | Enlarge network volume (≥40 GB); `rm -rf /workspace/.cache/pip /workspace/tmp/*`. |
| `FileNotFoundError` on freeze weights | Sync `results/detector_baselines/<family>/weights/best.pt` + `FROZEN.json` via `sync_evasion_runpod.sh`. Absolute Mac paths in the freeze are OK — loader falls back to `weights_rel`. |
| `PermissionError` crafting on `rtdetr_l` | Expected — held-out transfer target. Craft on `yolo11s` / `yolo11l` only; eval via `run_transfer.py`. |
| Transfer: `patch bank manifest missing` | Re-run craft with `--save-patches` before `run_transfer.py`. |
| Evasion dry-run: 0 targets | `data/raw/usv/` missing on the pod — re-sync the EO train bundle. |
