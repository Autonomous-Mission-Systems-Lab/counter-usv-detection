#!/usr/bin/env bash
# On-pod (RunPod) setup for EO detector baseline training.
#
# Run from the repo root AFTER the EO train bundle has been synced:
#   cd /workspace/counterUSV && bash scripts/detector/setup_runpod_eo.sh
#
# Design for constrained pod disk:
#   * REUSE the RunPod PyTorch template's CUDA torch (do NOT reinstall ~3 GB).
#   * venv uses --system-site-packages so it inherits that torch/torchvision.
#   * Install ONLY what detector training needs: ultralytics + pycocotools.
#     (Trajectory libs geopandas/movingpandas/pyarrow are a later phase — skip.)
#   * Keep pip cache + build tmp on /workspace (volume), not the small container disk.
# Then: re-export YOLO/COCO views (local symlinks) and dry-run the trainer.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

# Determine the volume root (prefer /workspace) for caches/tmp so we don't blow
# the container disk with pip cache and wheel builds.
VOL_ROOT="/workspace"
[[ -d "${VOL_ROOT}" && -w "${VOL_ROOT}" ]] || VOL_ROOT="${REPO_ROOT}"
export PIP_CACHE_DIR="${VOL_ROOT}/.cache/pip"
export TMPDIR="${VOL_ROOT}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

echo "[setup] repo=${REPO_ROOT}"
echo "[setup] python=$("${PYTHON}" -c 'import sys; print(sys.version.split()[0])')"
echo "[setup] volume=${VOL_ROOT} (pip cache + TMPDIR here)"
echo "[setup] disk:"; df -h "${VOL_ROOT}" 2>/dev/null | sed 's/^/[setup]   /' || true

# rsync for laptop<->pod syncs (many templates omit it).
if ! command -v rsync >/dev/null 2>&1; then
  echo "[setup] installing rsync"
  if command -v apt-get >/dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync
  elif command -v yum >/dev/null; then
    yum install -y -q rsync
  else
    echo "[setup] WARNING: no apt/yum — install rsync manually before re-syncing"
  fi
fi

# Does the base environment already have a CUDA-capable torch we can reuse?
SYSTEM_TORCH_OK=0
if "${PYTHON}" - <<'PY' 2>/dev/null
import sys
try:
    import torch, torchvision  # noqa: F401
except Exception:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 2)
PY
then
  SYSTEM_TORCH_OK=1
  echo "[setup] reusing base-environment torch ($("${PYTHON}" -c 'import torch; print(torch.__version__)')) — no reinstall"
else
  echo "[setup] no usable base torch detected (no CUDA or not installed)"
fi

# Create venv. Inherit system site-packages when we can reuse torch.
if [[ ! -d "${VENV}" ]]; then
  echo "[setup] creating venv ${VENV} (system-site-packages=${SYSTEM_TORCH_OK})"
  if [[ "${SYSTEM_TORCH_OK}" -eq 1 ]]; then
    "${PYTHON}" -m venv --system-site-packages "${VENV}"
  else
    "${PYTHON}" -m venv "${VENV}"
  fi
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "[setup] upgrading pip tooling"
python -m pip install -U --no-cache-dir pip wheel setuptools

# Detector training needs ultralytics + pycocotools. ultralytics pulls the rest
# (numpy/opencv/pillow/pyyaml/matplotlib/pandas/tqdm/psutil). Do NOT install torch
# here when we're reusing the base one.
# pandas + pyyaml are used by scripts/data/export_eo_views.py and train_detector.py;
# ultralytics doesn't always pull pandas, so install them explicitly.
EO_DEPS=("ultralytics==8.4.91" pycocotools pandas pyyaml)
if [[ "${SYSTEM_TORCH_OK}" -eq 1 ]]; then
  echo "[setup] installing ${EO_DEPS[*]} (reusing base torch)"
  pip install --no-cache-dir "${EO_DEPS[@]}"
else
  echo "[setup] installing ${EO_DEPS[*]} + CUDA torch"
  echo "[setup] NOTE: if torch download fails on disk, increase the pod Volume Disk"
  echo "        (Edit Pod → Volume Disk) or use the official PyTorch template."
  pip install --no-cache-dir "${EO_DEPS[@]}"
  # torch/torchvision come from ultralytics' deps or the template; install a CUDA
  # build explicitly only if still missing:
  python - <<'PY' || pip install --no-cache-dir torch torchvision
import torch  # noqa: F401
PY
fi

# CUDA sanity.
python - <<'PY'
import torch
print(f"[setup] torch={torch.__version__} cuda={torch.cuda.is_available()}"
      f" device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu/mps'}")
if not torch.cuda.is_available():
    print("[setup] WARNING: CUDA not available — full training should use a CUDA pod")
PY

# Required EO artifacts.
need=(
  data/annotations/coco_master.json
  data/splits/eo_image_splits.csv
  data/taxonomy.yaml
  configs/detector/families.yaml
)
for f in "${need[@]}"; do
  if [[ ! -e "${f}" ]]; then
    echo "[setup] ERROR: missing ${f} (sync incomplete?)" >&2
    exit 1
  fi
done

echo "[setup] re-exporting EO views (machine-local symlinks + portable data.yaml)"
python scripts/data/export_eo_views.py

# Spot-check a few symlinks resolve.
python - <<'PY'
from pathlib import Path
root = Path("data/eo_views/yolo/images/train")
links = list(root.glob("*"))[:5]
ok = sum(1 for p in links if p.is_file() and p.resolve().is_file())
print(f"[setup] symlink spot-check: {ok}/{len(links)} readable")
if links and ok == 0:
    raise SystemExit("[setup] ERROR: YOLO image symlinks do not resolve — "
                     "raw imagery under data/raw/ is missing or mis-nested")
import yaml as _yaml
d = _yaml.safe_load(Path("data/eo_views/yolo/data.yaml").read_text())
p = str(d.get("path", ""))
assert p.startswith("/"), f"data.yaml path must be absolute for Ultralytics, got {p!r}"
print(f"[setup] data.yaml path is absolute: {p}")
PY

echo "[setup] trainer dry-run"
python scripts/detector/train_detector.py --all --dry-run

echo
echo "[setup] ready. Train with:"
echo "  source .venv/bin/activate"
echo "  python scripts/detector/train_detector.py --all          # all 3 families"
echo "  python scripts/detector/train_detector.py --family yolo11s"
echo "  python scripts/detector/train_detector.py --family yolo11s --smoke   # 1-epoch sanity"
echo "Outputs → results/detector_baselines/<family>/"
