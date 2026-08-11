#!/usr/bin/env bash
# Build the Python environment for the auto-scaffold arm on a fresh machine (written for the
# B200 move, works on any CUDA >= 12.8 box). Idempotent: re-running upgrades in place.
#
# The pins are the EXACT versions the H100 source machine ran green for a full unattended
# 10-cycle smoke on 2026-08-11 — torch 2.8.0+cu128 / triton 3.4.0 already target Blackwell
# (sm_100), so the same set is expected to work on B200 unchanged. Change a pin only when a
# preflight failure tells you to.
#
# Everything is overridable; nothing is hardcoded to a user or a home directory:
#   VENV_DIR      where to create the venv          (default: <repo>/.venv)
#   PYTHON        base interpreter to seed it from  (default: python3.12, else python3)
#   ARM_CUDA_HOME CUDA toolkit with a working nvcc  (default: /usr/local/cuda-12.9 if present,
#                                                    else newest /usr/local/cuda-12.*)
#   TORCH_INDEX   torch wheel index                 (default: https://download.pytorch.org/whl/cu128)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"
VENV_DIR="${VENV_DIR:-$HOME/kernel}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ -z "${ARM_CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-12.9 $(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -rV); do
    [ -x "$c/bin/nvcc" ] && ARM_CUDA_HOME="$c" && break
  done
fi
if [ -z "${ARM_CUDA_HOME:-}" ] || [ ! -x "$ARM_CUDA_HOME/bin/nvcc" ]; then
  echo "FAIL: no CUDA toolkit with nvcc found; install CUDA >= 12.8 or set ARM_CUDA_HOME" >&2
  exit 1
fi
echo "repo=$REPO_ROOT venv=$VENV_DIR python=$PYTHON cuda=$ARM_CUDA_HOME"

[ -d "$VENV_DIR" ] || "$PYTHON" -m venv "$VENV_DIR"
PIP="$VENV_DIR/bin/pip"
"$PIP" install -q --upgrade pip

# Preferred path: the exact 357-package freeze of the H100 venv that ran the full unattended
# smoke (requirements-b200-freeze.txt, committed in-repo). Two lines get special handling:
#   - flash_attn is a file:// wheel reference to a path on the source machine. Installed
#     separately: FLASH_ATTN_WHEEL can point at a local copy (sync_to_remote can carry it);
#     unset, we skip it — prebuilt cu12torch2.8 wheels may lack sm_100 kernels anyway, and
#     sdpa attention is the working fallback until a Blackwell wheel is verified.
#   - flashinfer-python is stripped on sm_100+ below, so it is never installed from the lock.
LOCK="$REPO_ROOT/requirements-b200-freeze.txt"
if [ -f "$LOCK" ]; then
  echo "installing from exact freeze: $LOCK"
  FILTERED="$(mktemp)"
  grep -vE "^(flash_attn|flashinfer-python)" "$LOCK" > "$FILTERED"
  "$PIP" install -r "$FILTERED" --extra-index-url "$TORCH_INDEX"
  rm -f "$FILTERED"
  if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "${FLASH_ATTN_WHEEL:-}" ]; then
    "$PIP" install "$FLASH_ATTN_WHEEL"
  else
    echo "NOTE: flash_attn not installed (no FLASH_ATTN_WHEEL); preflight will warn."
  fi
else
  # Fallback: top-level pins only, pip resolves the rest.
  "$PIP" install "torch==2.8.0" --index-url "$TORCH_INDEX"
  "$PIP" install "vllm==0.11.0" "ray[default]==2.52.1" "transformers==4.56.1"
  CONSTRAINTS="$(mktemp)"
  printf 'torch==2.8.0\nvllm==0.11.0\ntriton==3.4.0\n' > "$CONSTRAINTS"
  "$PIP" install -c "$CONSTRAINTS" openai wandb ninja pytest pyarrow pandas
  rm -f "$CONSTRAINTS"
fi

# The verl fork itself, on top of the pinned set. NOTE the distribution is named "verl" — the
# same name verl-agent uses — which is WHY this venv must not be shared with an ALFWorld env:
# whichever repo installs last silently replaces the other's trainer.
"$PIP" install -e "$REPO_ROOT" --no-deps

# flashinfer's prebuilt wheels broke vLLM on sm_100 during the ALFWorld B200 move; vLLM falls
# back to its own kernels without it. Only strip it where the failure exists.
CAP_MAJOR="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)"
if [ "${CAP_MAJOR:-0}" -ge 10 ] && "$PIP" show flashinfer-python >/dev/null 2>&1; then
  echo "removing flashinfer-python (known-bad on sm_100+)"
  "$PIP" uninstall -y flashinfer-python
fi

echo
echo "Environment ready. Verify it:"
echo "  ARM_PYTHON=$VENV_DIR/bin/python ARM_CUDA_HOME=$ARM_CUDA_HOME \\"
echo "    ARM_MODEL=<base-model-dir> JUDGE_MODEL=<judge-model-dir> \\"
echo "    bash $REPO_ROOT/scripts/b200_preflight.sh"
