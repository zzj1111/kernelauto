#!/usr/bin/env bash
# Start the rubric judge — an OpenAI-compatible vLLM server the reward function calls to score
# kernel quality (see cudaforge/README.md). Must be up before launch_autoscaffold.sh's
# RUBRIC_VLLM_URL will resolve. All machine-specific bits are flags/env vars, not hardcoded.
set -euo pipefail

MODEL="${JUDGE_MODEL:-}"
NAME="${JUDGE_SERVED_NAME:-rubric-judge}"
PORT="${JUDGE_PORT:-8210}"
HOST="${JUDGE_HOST:-127.0.0.1}"
GPU="${JUDGE_CUDA_VISIBLE_DEVICES:-}"
GPU_MEM="${JUDGE_GPU_MEM:-0.45}"
MAX_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
VENV_PYTHON="${ARM_PYTHON:-$(command -v python3)}"
CUDA_HOME="${ARM_CUDA_HOME:-/usr/local/cuda-12.9}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --model PATH --gpu N [options]

  --model PATH        HF model dir/name for the judge (required)      (env JUDGE_MODEL)
  --gpu N              CUDA_VISIBLE_DEVICES for this server (required) (env JUDGE_CUDA_VISIBLE_DEVICES)
  --name NAME           served-model-name, must match RUBRIC_MODEL_NAME (env JUDGE_SERVED_NAME, default rubric-judge)
  --port N               (env JUDGE_PORT, default 8210)
  --host HOST              (env JUDGE_HOST, default 127.0.0.1)
  --gpu-mem FRAC             gpu_memory_utilization (env JUDGE_GPU_MEM, default 0.45 — leave room if
                               this GPU is shared with the kernel benchmark subprocess)
  --max-len N                  max_model_len (env JUDGE_MAX_MODEL_LEN, default 16384)
  --venv-python PATH             interpreter with vllm installed (env ARM_PYTHON, default: python3 on PATH)
  --cuda-home PATH                 (env ARM_CUDA_HOME, default /usr/local/cuda-12.9)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --gpu-mem) GPU_MEM="$2"; shift 2 ;;
    --max-len) MAX_LEN="$2"; shift 2 ;;
    --venv-python) VENV_PYTHON="$2"; shift 2 ;;
    --cuda-home) CUDA_HOME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$GPU" ]]; then
  echo "error: --model and --gpu are required" >&2
  usage >&2
  exit 1
fi

export PATH="$(dirname "$VENV_PYTHON"):$CUDA_HOME/bin:$PATH"
export CUDA_HOME
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== serve_rubric_judge.sh: model=$MODEL name=$NAME gpu=$GPU port=$PORT gpu_mem=$GPU_MEM ==="
exec "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$NAME" \
  --host "$HOST" --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --disable-log-requests
