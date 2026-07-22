# `experiments/` — training launch scripts

**Always run these from the repository root**, e.g. `bash experiments/train_exp_1.sh`.
The scripts use repo-root-relative paths (`./cudaforge/...`, `./dataset/...`).

## Prerequisites (env vars)

All host-specific paths were removed from these scripts; supply them via environment variables:

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `WANDB_API_KEY` | yes | W&B key. Scripts fail fast if unset (never hardcode it). |
| `CONDA_ACTIVATE` | yes | Path to your `<miniconda>/bin/activate`. |
| `CONDA_ENV` | yes | Your training conda env (name or path). |
| `CKPTS_DIR` | no | Checkpoint output dir. Defaults to `./checkpoints/<project>/<exp>`. |
| `WANDB_ENTITY` | no | Your W&B team/entity (commented out by default; uncomment/set to use). |
| `RUBRIC_VLLM_URL`, `RUBRIC_MODEL_NAME` | auto | Rubric LLM-judge endpoint. The `*+Rubric*` scripts start their own vLLM server and set these automatically. |

Example:

```bash
export WANDB_API_KEY=...              # your key
export CONDA_ACTIVATE=~/miniconda3/bin/activate
export CONDA_ENV=cuda                 # your env name
bash experiments/Qwen3-32B_KL003_1e-5_RewardFunction+Rubric.sh
```

## Training scripts

Reward `bench_rubric` = `cudaforge/reward_bench_rubric.py` (real benchmark + LLM rubric);
`rubric_ablation` = `cudaforge/reward_rubric_ablation.py` (no benchmark, LLM-only).

| Script | Model | Reward | Train data | Notes |
|--------|-------|--------|-----------|-------|
| `Qwen3-32B_KL003_1e-5_RewardFunction+Rubric.sh` | Qwen3-32B | bench_rubric | `train_stitchCUDA_skill.parquet` | Full pipeline: 4 train GPUs + 1 bench GPU + 2 rubric-vLLM GPUs. lr 1e-5. **Latest.** |
| `Qwen3-32B_KL003_1e-6_Rubric.sh` | Qwen3-32B | rubric_ablation | `train_new.parquet` | Ablation (no benchmark signal). lr 1e-6. |
| `Qwen3-32B_KL003_RewardFunction.sh` | Qwen3-32B | bench_rubric | `train_new.parquet` | Bench-only (no rubric vLLM). |
| `train_exp_1.sh` | Qwen3 | bench_rubric | `Level1/train.parquet` | Earlier experiment. |
| `train_exp_2.sh` | Qwen3 | bench_rubric | `Level1/train.parquet` | Earlier experiment. |
| `train_exp_32B.sh` | Qwen3-32B | bench_rubric | `train.parquet` | Earlier experiment. |

All use verl GRPO (`algorithm.adv_estimator=grpo`) with LoRA and KL loss coef 0.03 (`KL003`).

## `launch/` — one-shot tmux wrappers

Each wrapper opens a detached `tmux` session and runs the matching training script, so a run
survives disconnects. Run from repo root too (`bash experiments/launch/run.sh`).

| Wrapper | Runs |
|---------|------|
| `Qwen3_KL003_1e-5_RewardFunction+Rubric.sh` | `../Qwen3-32B_KL003_1e-5_RewardFunction+Rubric.sh` |
| `Qwen3_KL003_1e-6_Rubric.sh` | `../Qwen3-32B_KL003_1e-6_Rubric.sh` |
| `Qwen3_KL003_RewardFunction.sh` | `../Qwen3-32B_KL003_RewardFunction.sh` |
| `run.sh` | `../train_exp_1.sh` |
| `run_grpo.sh` | `../train_exp_2.sh` |
| `run_grpo_32B.sh` | `../train_exp_32B.sh` |

> Removed during cleanup: `run_grpo_30B.sh` — it wrapped `train_exp_30B.sh`, which does not exist
> (recoverable from git history if needed).
