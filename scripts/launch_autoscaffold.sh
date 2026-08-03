#!/usr/bin/env bash
# Launch the CudaForge auto-scaffold training loop (cudascaffold.run_arm) with every
# machine-specific setting overridable from the command line — no editing .py files to move
# this to a new server. Every flag has an ARM_* / AUTOSCAFFOLD_* env var equivalent (shown in
# --help); flags win if both are given. Defaults match the source machine's own setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [n_cycles]

  n_cycles                 how many scaffold cycles to run this process (default: 20)

Placement / hardware
  --gpus LIST               training GPUs                         (env ARM_GPUS,        default 0,1)
  --n-gpus N                 how many of --gpus verl trains on      (env ARM_N_GPUS,      default 2)
  --tp N                     rollout tensor-parallel size           (env ARM_TP,          default 2)
  --reward-gpu ID            GPU for the kernel benchmark subprocess (env ARM_REWARD_GPU,  default 3)
  --arch-list LIST           TORCH_CUDA_ARCH_LIST for your GPU       (env ARM_TORCH_CUDA_ARCH_LIST, default 9.0)
                              (H100/H200=9.0, A100=8.0, 4090=8.9 — check: nvidia-smi --query-gpu=compute_cap)

Toolchain
  --venv-python PATH          interpreter to run verl/vllm with       (env ARM_PYTHON, default: this script's own \$PATH resolution of python3)
  --cuda-home PATH             CUDA toolkit with a working nvcc        (env ARM_CUDA_HOME, default /usr/local/cuda-12.9)

Model / data
  --model PATH                base HF model to train from             (env ARM_MODEL)
  --train-file PATH            training parquet                        (env ARM_TRAIN_FILE)
  --val-file PATH               held-out validation parquet              (env ARM_VAL_FILE)
  --domain {cuda,triton}        which dataset/reward domain              (env ARM_DOMAIN, default cuda)

Rubric judge (must already be serving — see scripts/serve_rubric_judge.sh)
  --rubric-url URL              OpenAI-compatible endpoint               (env RUBRIC_VLLM_URL, default http://127.0.0.1:8210/v1/chat/completions)
  --rubric-model NAME            served-model-name of the judge           (env RUBRIC_MODEL_NAME, default rubric-judge)

Teacher (GPT controller)
  --openai-key-file PATH        file containing OPENAI_API_KEY=...       (env AUTOSCAFFOLD_OPENAI_KEY_FILE)
                                  (or just export OPENAI_API_KEY directly — that always wins)

Experiment identity / storage
  --exp NAME                    experiment name, namespaces all state    (env ARM_EXP, default cuda_scaffold_8b)
  --exp-root PATH                 base dir for per-exp logs/scaffold state (env ARM_EXP_ROOT)
  --ckpt-root PATH                 base dir for verl checkpoints            (env ARM_CKPT_ROOT)
  --root PATH                       repo root (auto-detected; only override if this script moved) (env ARM_ROOT)

  -h, --help                     show this help and exit

Every setting can also just be exported before calling this script, e.g.:
  ARM_EXP=my_run ARM_GPUS=0,1 ARM_MODEL=/path/to/model $(basename "$0") 20

This wraps: cd \$ROOT && python -m cudascaffold.run_arm <n_cycles>
EOF
}

N_CYCLES=20
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) export ARM_GPUS="$2"; shift 2 ;;
    --n-gpus) export ARM_N_GPUS="$2"; shift 2 ;;
    --tp) export ARM_TP="$2"; shift 2 ;;
    --reward-gpu) export ARM_REWARD_GPU="$2"; shift 2 ;;
    --arch-list) export ARM_TORCH_CUDA_ARCH_LIST="$2"; shift 2 ;;
    --venv-python) export ARM_PYTHON="$2"; shift 2 ;;
    --cuda-home) export ARM_CUDA_HOME="$2"; shift 2 ;;
    --model) export ARM_MODEL="$2"; shift 2 ;;
    --train-file) export ARM_TRAIN_FILE="$2"; shift 2 ;;
    --val-file) export ARM_VAL_FILE="$2"; shift 2 ;;
    --domain) export ARM_DOMAIN="$2"; shift 2 ;;
    --rubric-url) export RUBRIC_VLLM_URL="$2"; shift 2 ;;
    --rubric-model) export RUBRIC_MODEL_NAME="$2"; shift 2 ;;
    --openai-key-file) export AUTOSCAFFOLD_OPENAI_KEY_FILE="$2"; shift 2 ;;
    --exp) export ARM_EXP="$2"; shift 2 ;;
    --exp-root) export ARM_EXP_ROOT="$2"; shift 2 ;;
    --ckpt-root) export ARM_CKPT_ROOT="$2"; shift 2 ;;
    --root) export ARM_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
    *) N_CYCLES="$1"; shift ;;
  esac
done

ROOT="${ARM_ROOT:-$REPO_ROOT}"
export ARM_ROOT="$ROOT"

CUDA_HOME="${ARM_CUDA_HOME:-/usr/local/cuda-12.9}"
export ARM_CUDA_HOME="$CUDA_HOME"
PY="${ARM_PYTHON:-$(command -v python3)}"
export ARM_PYTHON="$PY"
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export CUDA_HOME

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  : # already set directly, nothing to check
elif [[ -n "${AUTOSCAFFOLD_OPENAI_KEY_FILE:-}" && ! -f "$AUTOSCAFFOLD_OPENAI_KEY_FILE" ]]; then
  echo "warning: --openai-key-file $AUTOSCAFFOLD_OPENAI_KEY_FILE does not exist and OPENAI_API_KEY is unset" >&2
fi

echo "=== launch_autoscaffold.sh ==="
echo "root=$ROOT exp=${ARM_EXP:-cuda_scaffold_8b} gpus=${ARM_GPUS:-0,1} reward_gpu=${ARM_REWARD_GPU:-3}"
echo "model=${ARM_MODEL:-<default>} domain=${ARM_DOMAIN:-cuda} arch_list=${ARM_TORCH_CUDA_ARCH_LIST:-9.0}"
echo "python=$PY cuda_home=$CUDA_HOME"
echo "rubric_url=${RUBRIC_VLLM_URL:-http://127.0.0.1:8210/v1/chat/completions}"
echo "n_cycles=$N_CYCLES"
echo "==============================="

cd "$ROOT"
exec "$PY" -m cudascaffold.run_arm "$N_CYCLES"
