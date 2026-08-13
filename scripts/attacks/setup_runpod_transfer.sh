#!/usr/bin/env bash
# On-pod setup for access-level transfer eval.
#
# Flow:
#   1. Craft YOLO patches WITH --save-patches (GPU; writes patch_bank/)
#   2. Hard-eval those banks on grey peer + rtdetr_l (GPU recommended, forward-only)
#
#   bash scripts/attacks/setup_runpod_transfer.sh
#   source .venv/bin/activate && tmux new -s transfer

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

echo "[setup-transfer] repo=${REPO_ROOT}"

if [[ ! -d "${VENV}" ]]; then
  echo "[setup-transfer] ERROR: missing ${VENV} — run setup_runpod_eo.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python - <<'PY'
import torch
print(f"[setup-transfer] torch={torch.__version__} cuda={torch.cuda.is_available()}"
      f" device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
if not torch.cuda.is_available():
    raise SystemExit("[setup-transfer] ERROR: CUDA required for craft; provision a CUDA pod")
PY

need=(
  data/annotations/coco_master.json
  data/splits/eo_image_splits.csv
  data/raw/usv/frames
  configs/attacks/access_levels.yaml
  configs/attacks/evasion.yaml
  configs/attacks/disguise.yaml
  configs/attacks/patch.yaml
  configs/attacks/marine_eot.yaml
  results/detector_baselines/FROZEN.json
  results/detector_baselines/yolo11s/weights/best.pt
  results/detector_baselines/yolo11l/weights/best.pt
  results/detector_baselines/rtdetr_l/weights/best.pt
)
for f in "${need[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[setup-transfer] ERROR: missing ${f}" >&2
    exit 1
  fi
done

python - <<'PY'
import sys
sys.path.insert(0, "src")
from counterusv.models.detector import DetectorBaseline
from counterusv.attacks.transfer import (
    access_level, default_transfer_targets, load_access_levels_config,
)
cfg = load_access_levels_config()
print(f"[setup-transfer] surrogates={list(cfg.surrogates)} held_out={cfg.held_out_target}")
print(f"[setup-transfer] yolo11s defaults → {default_transfer_targets('yolo11s', cfg)}")
for fam in list(cfg.surrogates) + [cfg.held_out_target]:
    b = DetectorBaseline.from_freeze(fam, device="0")
    print(f"[setup-transfer] freeze OK: {fam} role={b.transfer_role}")
assert access_level("yolo11s", "yolo11l", cfg) == "grey"
assert access_level("yolo11s", "rtdetr_l", cfg) == "black"
assert access_level("yolo11s", "yolo11s", cfg) == "white"
print("[setup-transfer] access-level map OK")
PY

echo "[setup-transfer] evasion transfer dry-run (expects patch bank if already crafted)"
python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --dry-run || true

echo
echo "[setup-transfer] ready. Suggested RunPod sequence (tmux):"
echo "  source .venv/bin/activate"
echo "  tmux new -s transfer"
echo
echo "  # --- 1) Craft with patch export (GPU; required once per surrogate) ---"
echo "  python scripts/attacks/run_evasion.py --family yolo11s --device 0 --save-patches"
echo "  python scripts/attacks/run_evasion.py --family yolo11l --device 0 --save-patches"
echo "  # Optional disguise (TMSR was 0% white-box; still protocol-complete):"
echo "  python scripts/attacks/run_disguise.py --family yolo11s --device 0 --save-patches --benign-class fishing"
echo
echo "  # --- 2) Transfer eval (forward-only; grey peer + rtdetr_l) ---"
echo "  python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11s --device 0"
echo "  python scripts/attacks/run_transfer.py --attack evasion --surrogate yolo11l --device 0"
echo "  python scripts/attacks/run_transfer.py --attack disguise --surrogate yolo11s --benign-class fishing --device 0"
echo
echo "Outputs → results/attacks/transfer/{evasion,disguise}/"
echo "Pull:"
echo "  rsync -avh --no-owner --no-group -e \"\$RSYNC_RSH\" \\"
echo "    root@<IP>:${REPO_ROOT}/results/attacks/transfer/ results/attacks/transfer/"
echo "  # also pull patch banks if you want them locally:"
echo "  rsync -avh --no-owner --no-group -e \"\$RSYNC_RSH\" \\"
echo "    root@<IP>:${REPO_ROOT}/results/attacks/evasion/ results/attacks/evasion/"
