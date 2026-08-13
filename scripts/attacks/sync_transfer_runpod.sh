#!/usr/bin/env bash
# Sync the access-level transfer overlay to a RunPod workspace.
#
# Run on your LAPTOP. EO imagery must already be on the pod
# (scripts/detector/sync_eo_train_bundle.sh). This pushes attack code + all
# three frozen detector weights (YOLO surrogates + rtdetr_l for black-box eval).
#
#   export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
#   ./scripts/attacks/sync_transfer_runpod.sh root@<IP>:/workspace/counterUSV
#
# On the pod:
#   bash scripts/attacks/setup_runpod_transfer.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LIST="${REPO_ROOT}/configs/attacks/runpod_transfer_paths.txt"

SSH_BASE="${RSYNC_RSH:-ssh}"
if [[ "${SSH_BASE}" == ssh ]]; then
  SSH_BASE="ssh"
elif [[ "${SSH_BASE}" == ssh\ * ]]; then
  SSH_BASE="ssh ${SSH_BASE#ssh }"
fi
case " ${SSH_BASE} " in
  *" -T "*) ;;
  *) SSH_BASE="${SSH_BASE} -T" ;;
esac
case " ${SSH_BASE} " in
  *" RequestTTY=no "*) ;;
  *) SSH_BASE="${SSH_BASE} -o RequestTTY=no" ;;
esac
RSYNC_RSH="${SSH_BASE}"

remote_sh() {
  # shellcheck disable=SC2086
  ${RSYNC_RSH} -n "${REMOTE_HOST}" "$@" </dev/null
}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <user@host>:<remote_repo_dir>" >&2
  exit 2
fi

DEST="$1"
if [[ "${DEST}" != *:* ]]; then
  echo "error: destination must be user@host:/remote/path" >&2
  exit 2
fi
REMOTE_ROOT="${DEST##*:}"
REMOTE_HOST="${DEST%:"${REMOTE_ROOT}"}"

echo "[sync-transfer] repo=${REPO_ROOT}"
echo "[sync-transfer] host=${REMOTE_HOST}"
echo "[sync-transfer] remote_root=${REMOTE_ROOT}"

if ! remote_sh "command -v rsync >/dev/null 2>&1"; then
  echo "[sync-transfer] remote has no rsync — installing…"
  remote_sh "if command -v apt-get >/dev/null; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; elif command -v yum >/dev/null; then yum install -y -q rsync; else exit 1; fi"
fi

remote_sh "mkdir -p $(printf '%q' "${REMOTE_ROOT}")"

PATHS=()
while IFS= read -r rel || [[ -n "${rel}" ]]; do
  [[ -z "${rel}" || "${rel}" =~ ^[[:space:]]*# ]] && continue
  PATHS+=("${rel%/}")
done < "${LIST}"

sync_one() {
  local rel="$1" src="${REPO_ROOT}/$1"
  if [[ ! -e "${src}" ]]; then
    echo "[sync-transfer] SKIP missing: ${rel}" >&2
    return 0
  fi
  echo "[sync-transfer] ${rel}"
  if [[ -d "${src}" ]]; then
    local target="${REMOTE_ROOT}/${rel}"
    remote_sh "mkdir -p $(printf '%q' "${target}")"
    rsync -avh --progress --no-owner --no-group -e "${RSYNC_RSH}" \
      "${src}/" "${REMOTE_HOST}:${target}/" </dev/null
  else
    local parent="${REMOTE_ROOT}/$(dirname "${rel}")"
    remote_sh "mkdir -p $(printf '%q' "${parent}")"
    rsync -avh --progress --no-owner --no-group -e "${RSYNC_RSH}" \
      "${src}" "${REMOTE_HOST}:${parent}/" </dev/null
  fi
}

for rel in "${PATHS[@]}"; do
  sync_one "${rel}"
done

echo
echo "[sync-transfer] done. On the pod:"
echo "  cd ${REMOTE_ROOT}"
echo "  bash scripts/attacks/setup_runpod_transfer.sh"
