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
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
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

# Torch first so nothing else drags in a CPU wheel; vllm second because it pins its own triton
# (3.4.0, which is what compiles the policy's generated kernels — it MUST be Blackwell-capable).
"$PIP" install "torch==2.8.0" --index-url "$TORCH_INDEX"
"$PIP" install "vllm==0.11.0" "ray[default]==2.52.1" "transformers==4.56.1"

# The verl fork itself. --no-deps after the heavy pins are in place would skip needed extras,
# so let pip resolve but keep torch pinned: constraint file wins over transitive requirements.
CONSTRAINTS="$(mktemp)"
printf 'torch==2.8.0\nvllm==0.11.0\ntriton==3.4.0\n' > "$CONSTRAINTS"
"$PIP" install -e "$REPO_ROOT" -c "$CONSTRAINTS"
"$PIP" install -c "$CONSTRAINTS" openai wandb ninja pytest pyarrow pandas
rm -f "$CONSTRAINTS"

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
