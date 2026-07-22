#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 0) GPU / Environment Setting
# ============================================================
# verl training GPUs (4 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# reward/bench GPU (1 GPU)
export REWARD_CUDA_VISIBLE_DEVICES=4

# rubric vLLM GPUs (2 GPUs)
export RUBRIC_CUDA_VISIBLE_DEVICES=5,6

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running (do not hardcode it)}"
# export WANDB_ENTITY="your-wandb-entity"   # optional: your W&B team/entity (uncomment & set)

# Training model (actor/critic)
export Model_path="Qwen/Qwen3-32B"

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export RAY_BACKEND_LOG_LEVEL=debug

max_response_length=16384
loss_mode=grpo

project_name=CudaForge_RL
exp_name="d0117r3_GRPO_level123_Qwen332_TP4_train4GPU_only_rubric_32B_TP2"

CKPTS_DIR="${CKPTS_DIR:-./checkpoints/${project_name}/${exp_name}}"

mkdir -p logs


# ============================================================
# 1) Conda
# ============================================================
source "${CONDA_ACTIVATE:?Set CONDA_ACTIVATE to <your-miniconda>/bin/activate}"
conda activate "${CONDA_ENV:?Set CONDA_ENV to your training conda env (name or path)}"


# ============================================================
# 2) Start Rubric vLLM (Qwen3-32B) on GPU5,6 (TP=2), port 8081
# ============================================================
RUBRIC_MODEL="Qwen/Qwen3-32B"
RUBRIC_PORT=8081
RUBRIC_HOST="127.0.0.1"

# Used by your CudaForge.py rubric client
export RUBRIC_VLLM_URL="http://${RUBRIC_HOST}:${RUBRIC_PORT}/v1/chat/completions"
export RUBRIC_MODEL_NAME="${RUBRIC_MODEL}"
export RUBRIC_VLLM_TIMEOUT_SEC="30"

VLLM_LOG="logs/vllm_rubric_${RUBRIC_PORT}_qwen332_tp2_gpu56.log"

# Qwen3-32B rubric is heavy; keep safe utilization first
VLLM_GPU_UTIL="0.70"
VLLM_MAX_LEN="32768"

echo "[launch] Starting rubric vLLM (${RUBRIC_MODEL}) on port ${RUBRIC_PORT} using GPUs ${RUBRIC_CUDA_VISIBLE_DEVICES} (TP=2) ..."

# Kill any existing process occupying the port (best-effort)
if command -v lsof >/dev/null 2>&1; then
  PID_ON_PORT=$(lsof -ti tcp:${RUBRIC_PORT} || true)
  if [[ -n "${PID_ON_PORT}" ]]; then
    echo "[launch] Port ${RUBRIC_PORT} is in use by PID=${PID_ON_PORT}, killing it..."
    kill -9 ${PID_ON_PORT} || true
  fi
fi

CUDA_VISIBLE_DEVICES=${RUBRIC_CUDA_VISIBLE_DEVICES} \
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "${RUBRIC_MODEL}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "${RUBRIC_PORT}" \
  --max-model-len "${VLLM_MAX_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  --tensor-parallel-size 2 \
  > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!
echo "[launch] rubric vLLM PID=${VLLM_PID}, log=${VLLM_LOG}"

cleanup() {
  echo "[cleanup] Stopping rubric vLLM PID=${VLLM_PID} ..."
  kill ${VLLM_PID} >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Wait until vLLM is ready (first start may download model, can take minutes)
echo "[launch] Waiting for rubric vLLM readiness: http://${RUBRIC_HOST}:${RUBRIC_PORT}/v1/models ..."
READY=0
for i in $(seq 1 900); do
  if command -v curl >/dev/null 2>&1; then
    if curl -s "http://${RUBRIC_HOST}:${RUBRIC_PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
  else
    READY=1
    break
  fi

  if (( i % 30 == 0 )); then
    echo "[launch] still waiting... (${i}/900)"
  fi
  sleep 1
done

if [[ "${READY}" -ne 1 ]]; then
  echo "[error] rubric vLLM did not become ready in time (900s). Log tail:"
  tail -n 120 "${VLLM_LOG}" || true
  exit 1
fi

echo "[launch] rubric vLLM is ready."


# ============================================================
# 3) Start verl GRPO training (on GPUs 0-3, TP=4)
# ============================================================
echo "[launch] Starting verl GRPO training (4 GPUs, rollout TP=4) ..."

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  trainer.project_name=CudaForge_RL \
  algorithm.adv_estimator=grpo \
  data.train_files=./dataset/CudaForge/train_new.parquet \
  data.val_files=./dataset/CudaForge/test.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=8192 \
  data.max_response_length=16384 \
  actor_rollout_ref.model.lora_rank=128 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.model.path=$Model_path \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.kl_loss_coef=0.03 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  critic.optim.lr=1e-5 \
  critic.model.path=$Model_path \
  critic.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  reward_model.enable=False \
  reward_model.reward_manager=dapo \
  custom_reward_function.path=./cudaforge/reward_rubric_ablation.py \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.save_freq=20 \
  trainer.test_freq=100 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="DAPO" \
  trainer.experiment_name=${exp_name} \
  trainer.total_epochs=4
