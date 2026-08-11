#!/usr/bin/env bash
# One-command environment build with uv — the same packaging style the ALFWorld move used.
#
# Installs the EXACT 357-package set frozen from the H100 venv that ran the full unattended
# 10-cycle smoke on 2026-08-11 (requirements-b200-freeze.txt). torch 2.8.0+cu128 / triton
# 3.4.0 already target Blackwell (sm_100), so the same set is expected to work on B200.
#
# Nothing is hardcoded to a user or machine:
#   VENV_DIR          where to create the venv        (default: ~/kernel — the directory
#                                                      name IS the env name uv shows on activate)
#   UV_CACHE_DIR      uv's wheel cache                (uv default: ~/.cache/uv; point at a big
#                                                      disk when the root filesystem is small)
#   FLASH_ATTN_WHEEL  local flash-attn wheel to install (skips the wget below)
#   TORCH_INDEX       extra wheel index               (default: pytorch.org cu128)
#
# Needs: internet egress to PyPI (and github.com for the flash-attn wheel unless
# FLASH_ATTN_WHEEL is given). For an air-gapped box, ask for the offline wheel bundle instead.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/kernel}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
LOCK="$REPO_ROOT/requirements-b200-freeze.txt"

if ! command -v uv >/dev/null; then
  echo "uv not found — installing to ~/.local/bin (needs internet; or install uv yourself first)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -f "$LOCK" ] || { echo "FAIL: $LOCK missing — incomplete clone?" >&2; exit 1; }

echo "repo=$REPO_ROOT venv=$VENV_DIR uv=$(uv --version)"

# uv brings its own CPython if the box lacks 3.12 (the freeze is cp312).
uv venv --python 3.12 "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"

# The freeze minus the two lines that cannot install as written:
#   flash_attn — a file:// path on the source machine; installed from wheel/URL below.
#   flashinfer-python — its prebuilt wheels broke vLLM on sm_100 (ALFWorld B200 move);
#     stripped here and guarded again after install.
FILTERED="$(mktemp)"
grep -vE "^(flash_attn|flashinfer-python)" "$LOCK" > "$FILTERED"
# --no-deps: the freeze is a complete closed set, so nothing needs resolving — and the live
# env it mirrors contains metadata inconsistencies pip tolerated but uv's resolver refuses
# (outlines 0.1.11 declares outlines-core==0.1.26 while 0.2.11 is installed and working;
# verl declares numpy<2 while vllm 0.11/torch 2.8 need and run 2.2.6). Resolving would
# "fix" those into a DIFFERENT env than the one that passed the smoke.
# unsafe-best-match: the pytorch index carries old copies of common packages (certifi, ...);
# uv's default first-index-wins strategy would pin to those. Both indexes are trusted.
uv pip install --no-deps -r "$FILTERED" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
rm -f "$FILTERED"

if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "${FLASH_ATTN_WHEEL:-}" ]; then
  uv pip install "$FLASH_ATTN_WHEEL"
else
  echo "fetching flash-attn prebuilt wheel (official release)"
  TMPW="$(mktemp -d)"
  if wget -nv -P "$TMPW" "$FLASH_ATTN_URL"; then
    uv pip install "$TMPW"/flash_attn-*.whl
  else
    echo "WARN: could not fetch flash-attn; training must use attention=sdpa (preflight warns)"
  fi
  rm -rf "$TMPW"
fi

# The repo IS the verl package (name="verl" — the same name verl-agent uses, which is why this
# venv must never be shared with an ALFWorld env: whichever installs last replaces the other).
uv pip install -e "$REPO_ROOT" --no-deps

CAP_MAJOR="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo 0)"
if [ "${CAP_MAJOR:-0}" -ge 10 ] && uv pip show flashinfer-python >/dev/null 2>&1; then
  echo "removing flashinfer-python (known-bad on sm_100+)"
  uv pip uninstall flashinfer-python
fi

echo
echo "Environment ready at $VENV_DIR. Verify it:"
echo "  ARM_PYTHON=$VENV_DIR/bin/python ARM_MODEL=<base-model-dir> JUDGE_MODEL=<judge-dir> \\"
echo "    bash $REPO_ROOT/scripts/b200_preflight.sh"
