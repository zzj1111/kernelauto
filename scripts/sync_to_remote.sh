#!/usr/bin/env bash
# Push everything a fresh machine needs to run the arm: the repo (via git bundle or rsync,
# so it works even where the remote cannot reach GitHub) and the model weights (rsync).
# Checks reachability first and prints what it will do before doing it.
#
# Usage:
#   scripts/sync_to_remote.sh user@b200-host [options]
#
# Options (all also settable as env vars; nothing is hardcoded):
#   --remote-root DIR   where to put things on the remote   (env REMOTE_ROOT, default: ~/autoscaffold)
#   --models "D1 D2"    local model dirs to copy            (env SYNC_MODELS, default: none)
#   --repo-mode MODE    git | rsync                         (env REPO_MODE, default: git)
#                        git   = push current branch to the GitHub remote, clone/pull on the box
#                        rsync = copy the working tree directly (no GitHub needed on the remote)
#   --dry-run           show commands without executing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-\$HOME/autoscaffold}"
SYNC_MODELS="${SYNC_MODELS:-}"
REPO_MODE="${REPO_MODE:-git}"
DRY=""
REMOTE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --models) SYNC_MODELS="$2"; shift 2 ;;
    --repo-mode) REPO_MODE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; exit 1 ;;
    *) REMOTE="$1"; shift ;;
  esac
done
[ -n "$REMOTE" ] || { echo "usage: $0 user@host [--models \"...\"] [--repo-mode git|rsync]" >&2; exit 1; }

run() { echo "+ $*"; [ -n "$DRY" ] || "$@"; }

echo "== reachability =="
ssh -o ConnectTimeout=10 "$REMOTE" "echo remote ok: \$(hostname) && mkdir -p $REMOTE_ROOT" || {
  echo "FAIL: cannot ssh to $REMOTE" >&2; exit 1; }

echo "== repo ($REPO_MODE) =="
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
if [ "$REPO_MODE" = git ]; then
  DIRTY="$(git -C "$REPO_ROOT" status --porcelain)"
  [ -z "$DIRTY" ] || { echo "FAIL: working tree has uncommitted changes; commit or use --repo-mode rsync" >&2; exit 1; }
  URL="$(git -C "$REPO_ROOT" remote get-url origin)"
  echo "remote will clone/pull $URL ($BRANCH)"
  run ssh "$REMOTE" "if [ -d $REMOTE_ROOT/repo/.git ]; then git -C $REMOTE_ROOT/repo fetch origin && git -C $REMOTE_ROOT/repo checkout $BRANCH && git -C $REMOTE_ROOT/repo pull --ff-only origin $BRANCH; else git clone -b $BRANCH $URL $REMOTE_ROOT/repo; fi"
else
  # Working tree straight over the wire. Excludes are run artifacts, not code.
  run rsync -a --info=progress2 \
      --exclude .git --exclude outputs/ --exclude cudaforge_logs/ --exclude '*.pyc' \
      --exclude __pycache__/ --exclude .venv/ \
      "$REPO_ROOT/" "$REMOTE:$REMOTE_ROOT/repo/"
fi

if [ -n "$SYNC_MODELS" ]; then
  echo "== models =="
  run ssh "$REMOTE" "mkdir -p $REMOTE_ROOT/models"
  for m in $SYNC_MODELS; do
    [ -d "$m" ] || { echo "FAIL: $m is not a directory" >&2; exit 1; }
    run rsync -a --info=progress2 "$m" "$REMOTE:$REMOTE_ROOT/models/"
  done
else
  echo "== models: none requested (pass --models). Alternative on the remote:"
  echo "     hf download Mingyi-Hong/drkernel-8b-coldstart-eosfix --local-dir \$ROOT/models/drkernel-8b  (needs HF_TOKEN)"
  echo "     hf download Qwen/Qwen3-8B --local-dir \$ROOT/models/Qwen3-8B"
fi

echo
echo "Next, ON THE REMOTE:"
echo "  bash $REMOTE_ROOT/repo/scripts/setup_b200_uv.sh   # builds ~/kernel"
echo "  ARM_PYTHON=\$HOME/kernel/bin/python ARM_MODEL=$REMOTE_ROOT/models/<base> \\"
echo "    JUDGE_MODEL=$REMOTE_ROOT/models/<judge> bash $REMOTE_ROOT/repo/scripts/b200_preflight.sh"
