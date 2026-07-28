#!/usr/bin/env bash
# Sync the EO-detector training bundle to a RunPod (or any remote) workspace.
#
# Run this on your LAPTOP (where the data lives), not inside the pod.
#
# Preferred (RunPod Connect → Direct TCP IP + port — most reliable for rsync):
#   export RSYNC_RSH='ssh -T -p <PORT> -i ~/.ssh/runpod -o RequestTTY=no'
#   ./scripts/detector/sync_eo_train_bundle.sh root@<IP>:/workspace/counterUSV
#
# Proxy SSH (ssh.runpod.io) — often flaky for rsync; prefer Direct TCP if this fails:
#   export RSYNC_RSH='ssh -T -i ~/.ssh/runpod -o RequestTTY=no'
#   ./scripts/detector/sync_eo_train_bundle.sh <poduser>@ssh.runpod.io:/workspace/counterUSV
#
# After sync, on the pod:
#   bash scripts/detector/setup_runpod_eo.sh
#   python scripts/detector/train_detector.py --all

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LIST="${REPO_ROOT}/configs/detector/runpod_eo_paths.txt"

# Build an ssh command that never requests a TTY. RunPod's proxy prints
# "doesn't support PTY" into the stream if a TTY is requested, which corrupts rsync.
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
# NOTE: never add ssh -n here — rsync uses this ssh as transport over stdin/stdout;
# -n redirects that from /dev/null and kills the stream (rsync error 12).
RSYNC_RSH="${SSH_BASE}"

remote_sh() {
  # -n so standalone ssh calls never steal a surrounding stdin fd.
  # shellcheck disable=SC2086
  ${RSYNC_RSH} -n "${REMOTE_HOST}" "$@" </dev/null
}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <user@host>:<remote_repo_dir>" >&2
  echo "  direct:  RSYNC_RSH='ssh -T -p PORT -i ~/.ssh/runpod' $0 root@IP:/workspace/counterUSV" >&2
  echo "  proxy:   RSYNC_RSH='ssh -T -i ~/.ssh/runpod' $0 PODUSER@ssh.runpod.io:/workspace/counterUSV" >&2
  exit 2
fi

DEST="$1"
if [[ "${DEST}" != *:* ]]; then
  echo "error: destination must be user@host:/remote/path" >&2
  exit 2
fi
REMOTE_ROOT="${DEST##*:}"
REMOTE_HOST="${DEST%:"${REMOTE_ROOT}"}"

echo "[sync] repo=${REPO_ROOT}"
echo "[sync] host=${REMOTE_HOST}"
echo "[sync] remote_root=${REMOTE_ROOT}"
echo "[sync] rsh=${RSYNC_RSH}"

# Ensure remote has rsync (many RunPod PyTorch images omit it).
if ! remote_sh "command -v rsync >/dev/null 2>&1"; then
  echo "[sync] remote has no rsync — installing (apt/yum)…"
  if ! remote_sh "if command -v apt-get >/dev/null; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync; elif command -v yum >/dev/null; then yum install -y -q rsync; else exit 1; fi"; then
    echo "[sync] ERROR: could not install rsync on the pod." >&2
    echo "  SSH in and run:  apt-get update && apt-get install -y rsync" >&2
    exit 1
  fi
fi

remote_sh "mkdir -p $(printf '%q' "${REMOTE_ROOT}")"

# Read the path list into an array BEFORE any ssh/rsync (so nothing consumes it).
PATHS=()
while IFS= read -r rel || [[ -n "${rel}" ]]; do
  [[ -z "${rel}" || "${rel}" =~ ^[[:space:]]*# ]] && continue
  PATHS+=("${rel%/}")   # normalize: drop any trailing slash
done < "${LIST}"

echo "[sync] ${#PATHS[@]} paths to sync"

# Copy one path so its RELATIVE location is preserved exactly under REMOTE_ROOT.
#   dir  rel=foo/bar  -> mkdir REMOTE/foo/bar ; rsync REPO/foo/bar/ -> REMOTE/foo/bar/
#   file rel=foo.txt  -> mkdir REMOTE/(dir)   ; rsync REPO/foo.txt   -> REMOTE/(dir)/
sync_one() {
  local rel="$1" src="${REPO_ROOT}/$1"
  if [[ ! -e "${src}" ]]; then
    echo "[sync] SKIP missing: ${rel}" >&2
    return 0
  fi
  echo "[sync] ${rel}"
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

# Docs for the on-pod workflow (not necessarily in the path list).
sync_one "docs/RUNPOD.md" || true
sync_one "docs/TRANSFER_PROTOCOL.md" || true
sync_one "README.md" || true

echo
echo "[sync] done. On the pod:"
echo "  cd ${REMOTE_ROOT}"
echo "  ls scripts/detector/setup_runpod_eo.sh   # should exist"
echo "  bash scripts/detector/setup_runpod_eo.sh"
echo "  python scripts/detector/train_detector.py --all"
