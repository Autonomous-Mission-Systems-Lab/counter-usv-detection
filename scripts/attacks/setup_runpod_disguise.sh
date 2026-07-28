#!/usr/bin/env bash
# On-pod (RunPod) setup / sanity check for the disguise (TMSR) GPU run.
#
# Prerequisites:
#   1. EO train bundle already synced + setup_runpod_eo.sh run once
#      (data/, .venv with CUDA torch + ultralytics).
#   2. Disguise overlay synced:
#        ./scripts/attacks/sync_disguise_runpod.sh root@IP:/workspace/counterUSV
#
# Then on the pod:
#   cd /workspace/counterUSV && bash scripts/attacks/setup_runpod_disguise.sh
#
# This does NOT reinstall torch — it reuses the detector venv, verifies CUDA +
# frozen surrogate weights, dry-runs the disguise slice, and prints the tmux
# command for the full run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

VENV="${VENV:-${REPO_ROOT}/.venv}"
FAMILY="${FAMILY:-yolo11s}"

echo "[setup-disguise] repo=${REPO_ROOT}"

if [[ ! -d "${VENV}" ]]; then
  echo "[setup-disguise] ERROR: missing ${VENV}" >&2
  echo "  Run scripts/detector/setup_runpod_eo.sh first (creates the CUDA venv)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python - <<'PY'
import torch
print(f"[setup-disguise] torch={torch.__version__} cuda={torch.cuda.is_available()}"
      f" device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
if not torch.cuda.is_available():
    raise SystemExit(
        "[setup-disguise] ERROR: CUDA not available — provision a CUDA RunPod "
        "template (Official PyTorch) and re-run setup_runpod_eo.sh"
    )
PY

need=(
  data/annotations/coco_master.json
  data/splits/eo_image_splits.csv
  data/raw/usv/frames
  configs/attacks/disguise.yaml
  configs/attacks/patch.yaml
  configs/attacks/marine_eot.yaml
  results/detector_baselines/FROZEN.json
  "results/detector_baselines/${FAMILY}/weights/best.pt"
)
for f in "${need[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[setup-disguise] ERROR: missing ${f}" >&2
    echo "  Sync EO bundle + disguise overlay (see docs/RUNPOD.md § Disguise)." >&2
    exit 1
  fi
done

# Spot-check freeze loads via weights_rel (absolute Mac paths in FROZEN.json
# will not resolve on the pod).
python - <<PY
import sys
sys.path.insert(0, "src")
from counterusv.models.detector import DetectorBaseline
from counterusv.attacks.disguise import DifferentiableSurrogate, to_torch_device
fam = "${FAMILY}"
b = DetectorBaseline.from_freeze(fam, device="0")
print(f"[setup-disguise] freeze OK: {fam} weights={b.weights}")
dev = to_torch_device("auto")
print(f"[setup-disguise] torch device from auto: {dev}")
# Cheap forward on GPU — catches device / head-shape issues before a long craft.
s = DifferentiableSurrogate.from_family(fam, device="0")
import torch
# Input must require grad: surrogate weights are frozen; the attack's learnable
# patch is upstream of the head, same as this smoke.
x = torch.rand(1, 3, 640, 640, device=s.device, requires_grad=True)
boxes, scores = s.forward_scores(x)
print(f"[setup-disguise] raw head OK: boxes={tuple(boxes.shape)} scores={tuple(scores.shape)}")
assert scores.shape[1] >= 9, "expected usv class slot in head"
loss = scores[:, s.class_index("usv"), :].mean()
loss.backward()
assert x.grad is not None and float(x.grad.abs().sum()) > 0
print("[setup-disguise] CUDA autograd OK")
PY

echo "[setup-disguise] disguise dry-run (slice wiring, no optimize)"
python scripts/attacks/run_disguise.py --family "${FAMILY}" --device 0 --dry-run

echo
echo "[setup-disguise] ready. Full TMSR run (survive SSH disconnect):"
echo "  apt-get update -qq && apt-get install -y -qq tmux   # once if missing"
echo "  source .venv/bin/activate"
echo "  tmux new -s disguise"
echo "  # inside tmux:"
echo "  python scripts/attacks/run_disguise.py --family ${FAMILY} --device 0"
echo "  # detach: Ctrl-b then d"
echo "  # optional smoke first:"
echo "  python scripts/attacks/run_disguise.py --family ${FAMILY} --device 0 --max-images 2 --steps 30"
echo
echo "Outputs → results/attacks/disguise/${FAMILY}/"
echo "Pull back from the laptop (same RSYNC_RSH as sync):"
echo "  rsync -avh --no-owner --no-group -e \"\$RSYNC_RSH\" \\"
echo "    root@<IP>:${REPO_ROOT}/results/attacks/disguise/ results/attacks/disguise/"
