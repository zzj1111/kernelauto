# Installing the CudaForge AutoScaffold stack with `uv` on a new server

Scope: this covers **StitchCUDA + `cudascaffold/` + `cudaforge/`** — the RL loop that trains a
policy to write CUDA kernels, with an LLM "Teacher" that edits the scaffold prompt between
cycles. It does not cover the other unrelated projects (ALFWorld/GiGPO, math evals, ...) that
happen to share the source machine's venv today.

Everything below was reverse-engineered from the live setup on the source server on 2026-08-03
(a training run was active in `exp_cudaforge/cuda_scaffold_qwen3` while this was written — none
of the commands here were run against it). Version numbers are what's actually installed and
verified working, not just what the repo's own `requirements*.txt`/`setup.py` list (those are
looser upstream-verl defaults; e.g. `setup.py` pins `numpy<2.0.0` but the working env runs
`numpy==2.2.6`, which vLLM 0.11.0 / torch 2.8 actually need).

## 0. What you're transferring

Four things move independently — code, Python env, model/data assets, and secrets. Don't
conflate them.

| What | Where on source | Size | How to move |
|---|---|---|---|
| Repo (code) | `/mnt/data1/zha00175/StitchCUDA` | ~50MB tracked (+ 1.4MB untracked parquet) | `git push` + clone, or `rsync` |
| Python env | `/dev/shm/verl_env` (tmpfs — **gone on reboot**, see §5) | 21GB, 360 packages, multi-project | rebuild with `uv`, don't copy |
| Model weights | `/mnt/data1/zha00175/models/Qwen3-8B` (+ judge model) | tens of GB | `rsync`/`hf download` on target |
| Secrets | `/mnt/data1/zha00175/tool-agent-secrets/openai.env` | — | recreate by hand, never git/rsync |

## 1. System prerequisites on the target server

- **GPU**: NVIDIA, Compute Capability matching what you set in `TORCH_CUDA_ARCH_LIST` below.
  Source is 8x H200 (sm_90 / "9.0"). Check the target with:
  `nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`
- **Driver**: must support CUDA 12.8 (torch 2.8's bundled CUDA). Source driver is 575.57.08;
  anything ≥ ~550 generally works, but confirm with `nvidia-smi`.
- **CUDA toolkit with `nvcc`** for compiling generated kernels (`cudaforge/kernel_runner.py`
  shells out to `nvcc`) — install CUDA 12.8 or 12.9. **The distro's default `/usr/bin/nvcc` is
  usually too old and will silently fail to target modern GPUs** (on the source machine,
  system nvcc was CUDA 11.5, which cannot target sm_90/Hopper at all — this cost real debugging
  time, see `exp_cudaforge/DECISIONS.md`). Verify: `nvcc --version` under whatever PATH you set
  in step 4 must report ≥ 12.8 and must not be the system package manager's copy.
- **Python 3.12** exactly (the flash-attn wheel below is a cp312 binary).
- **gcc/g++** (source uses Ubuntu 22.04's gcc 11.4) and **ninja** (installed via pip, but torch
  looks for a `ninja` *executable* on `PATH`, not the pip package alone — see step 4's PATH note).
- `uv` itself: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 2. Get the code onto the target

The repo already has a real GitHub remote (`git@github.com:zzj1111/StitchCUDA.git`). As of this
writing there are uncommitted files that make up the actual autoscaffold work and won't exist on
the target unless you commit them first:

```
cudascaffold/                              # the whole auto-scaffold controller — currently untracked
cudaforge/reward_bench_rubric.py           # modified (rubric dimension fixes, see DECISIONS.md)
dataset/CudaForge/build_clean_split.py
dataset/CudaForge/train_new_clean.parquet  # 1.4MB, small enough to commit directly
_mse_wrap.cpp attention_fused_kernel.cu bn_relu_conv_kernels.cu
fused_pointwise.cu kernels.cu maxpool2d_kernel.cu vlad_cuda_ops.cpp
```

Decide how you want to move this (commit+push vs. tarball/rsync) — I didn't push anything on
your behalf. Either way, exclude generated/runtime junk that `.gitignore` already knows about:
`outputs/`, `cudaforge_logs/`, `**/checkpoints`, `**/wandb`, `__pycache__/`.

On the target:
```bash
git clone git@github.com:zzj1111/StitchCUDA.git
cd StitchCUDA
```

## 3. Build the environment with `uv`

```bash
cd StitchCUDA
uv venv --python 3.12 .venv        # or wherever you want it — see §5 re: /dev/shm
source .venv/bin/activate

# torch first (its ABI/version gates the flash-attn wheel below)
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# rest of the pinned stack
uv pip install -r requirements-uv.txt

# flash-attn: NOT a compile-from-source step and NOT something you need to copy off the
# source machine — verl's own scripts/install_vllm_sglang_mcore.sh gets it the same way:
# the official prebuilt wheel from the flash-attention GitHub release.
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# (the source machine happens to have this cached at
#  /home/zha00175/CudaForge_plus/verl/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
#  if the target has no internet egress — copy that file instead of wget'ing.)

# the repo IS the verl package (pyproject.toml name="verl") — install it editable so
# cudascaffold's `python -m cudascaffold.run_arm` (which imports verl via cwd) and any
# direct `import verl` both resolve here, not to some other verl checkout on the box.
# --no-deps matters: setup.py's own install_requires pins "numpy<2.0.0" (stale — the
# verified-working combo above needs numpy==2.2.6 for vllm 0.11.0/torch 2.8), and a
# plain `uv pip install -e .` will silently re-resolve and DOWNGRADE numpy to 1.26.4.
# Confirmed by actually running this: without --no-deps, numpy 2.2.6 -> 1.26.4.
uv pip install -e . --no-deps
```

Verified end-to-end on 2026-08-03 in a scratch venv (not the live training one): `torch`,
`flash_attn`, `verl` (editable from this repo), `vllm`, and `cudascaffold` (all submodules —
`scaffold`, `adapters`, `gates`, `loop`, `observation`, `teacher`, `run_arm`) all import cleanly,
`torch.cuda.is_available()` is `True`, and `cudaforge/reward_bench_rubric.py` loads the same way
verl loads it (`importlib.util.spec_from_file_location`, not a package import) and exposes
`compute_score`. Not verified: an actual training step (would contend for GPUs with the live
run) and the rubric judge server end-to-end.

One more thing this run surfaced: installing into a venv that lives on the NFS mount
(`/mnt/data1`) makes even plain `import torch` noticeably slow (tens of seconds — Python import
opens many small files, and NFS per-file latency adds up) versus local disk. Not fatal, but if
the target server's import/compile times feel sluggish, check whether the venv is on network
storage before assuming something's actually broken.

Note on the source venv: the pip metadata there shows `verl` editable-installed from a
*different* directory (`/home/zha00175/verl_clean`, a separate fork with its own divergent
commits for unrelated work). That's stale/misleading — every actual training invocation `cd`s
into `StitchCUDA` first and runs `python -m verl.trainer.main_ppo`, and `-m` prepends the CWD to
`sys.path`, so `StitchCUDA/verl/` is what really executes, shadowing the editable link. On the
target, just `uv pip install -e .` from inside `StitchCUDA` and there's no ambiguity.

### Deliberately excluded from `requirements-uv.txt`

Present in the source venv but not needed for this training path — skip them unless you hit an
ImportError that says otherwise:
- `sglang` — only used if `actor_rollout_ref.rollout.name=sglang`; CudaForge launches use `vllm`.
- `transformer_engine` / Megatron / `mbridge` — only imported by verl's Megatron checkpoint code
  (`verl/utils/megatron/*`, `verl/utils/checkpoint/megatron_checkpoint_manager.py`); this setup
  trains with FSDP + LoRA, never touches that path.
- `TransferQueue` — imported in `verl/trainer/main_ppo.py` but behind `if config.transfer_queue.enable`,
  which the launch scripts never set.
- `liger-kernel` — an optional verl extra (`GPU_REQUIRES`), not actually installed in the source
  venv either.

If you later also want to run the ALFWorld/GiGPO side of this venv on the same box, that's a
different, larger dependency set (verl-agent, skyrl_gym, etc.) — out of scope here.

## 4. Environment variables (every launch script sets these — see `exp_cudaforge/*/launch.sh.env`)

```bash
# CUDA toolchain — MUST put the venv's bin AND the CUDA 12.8/12.9 toolkit ahead of anything else.
# Two real failure modes this avoids: (1) picking up an ancient system nvcc that can't target
# your GPU's arch, (2) torch shelling out to `ninja` as a bare executable name, which only
# resolves if the venv's bin/ (where pip put the ninja binary) is on PATH.
export PATH=/path/to/StitchCUDA/.venv/bin:/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
export TORCH_CUDA_ARCH_LIST=9.0     # H200/H100/Hopper. Ampere (A100) = 8.0, Ada (4090) = 8.9, etc.

# cudascaffold Teacher (GPT-based controller) — create this file yourself on the target,
# do NOT copy the source machine's key file over the network/into git.
#   printf 'OPENAI_API_KEY=sk-...\n' > /path/to/your/openai.env
# then either export OPENAI_API_KEY directly, or point teacher.py's key_file default at it
# (cudascaffold/teacher.py:17, DEFAULT_KEY_FILE — currently hardcoded to a source-machine path).

# Rubric judge (a separate vLLM server scoring kernel quality — see cudaforge/README.md)
export RUBRIC_VLLM_URL=http://127.0.0.1:8210/v1/chat/completions
export RUBRIC_MODEL_NAME=rubric-judge
export RUBRIC_VLLM_TIMEOUT_SEC=180
```

`cudascaffold/run_arm.py` also hardcodes `ROOT = "/mnt/data1/zha00175/StitchCUDA"` (line ~30) —
update that to the target's repo path, or the run will read/write dataset and checkpoint paths
on the wrong (source) filesystem.

## 5. About `/dev/shm/verl_env`

The source env lives in `/dev/shm`, i.e. tmpfs — RAM-backed, and **wiped on every reboot**
(this has already bitten this project once; see the `alfworld-autoscaffold-harness` memory note
for the ALFWorld side of the same lesson). It's fast to import from, which is presumably why it
was put there, but it means the "environment" on the source machine is not actually durable
infrastructure — it's rebuilt from scratch after every reboot via whatever install script
originally produced it (not present as a single script in this repo; this document is the
reconstruction of that missing script).

Recommendation for the new server: put `.venv` on persistent disk (as in §3). Only move it to
`/dev/shm` deliberately, once you've confirmed local disk import/compile speed is actually a
bottleneck, and with a rebuild script on hand for the next reboot.

## 6. Assets that are not part of the environment

- **Base model**: `/mnt/data1/zha00175/models/Qwen3-8B` — `rsync` it over, or re-download with
  `huggingface-cli download Qwen/Qwen3-8B` (check whether the source copy is stock or already
  modified before assuming they're identical).
- **Rubric judge model**: `Qwen2.5-VL-32B-Instruct` (see `exp_cudaforge/start_rubric.sh`) or
  `Qwen3-8B` again (see `exp_cudaforge/judge_serve.sh` — the two scripts disagree on which judge
  model to use; `judge_serve.sh` is the newer one, per its longer rationale comments).
- **Training data**: `dataset/CudaForge/{train,test}.parquet`, `train_new_clean.parquet` — small,
  travel with the repo (see §2).
- **Checkpoints**: `/mnt/data1/zha00175/cuda_scaffold_ckpts/*` — ~36GB per saved step, only
  bring these if you're resuming a run rather than starting fresh.

## 7. Smoke test before trusting a full run

1. `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"`
2. `python -c "import flash_attn, vllm, verl; print('ok')"`
3. Compile-only check of the CUDA toolchain, mirroring what `kernel_runner.py` does:
   `nvcc -arch=sm_90 -o /tmp/t /usr/local/cuda-12.9/samples/.../someSample.cu` (or just run
   `python -m cudaforge.kernel_runner` against one known-good kernel payload) — confirm it
   compiles for your GPU's actual arch, not just "some" CUDA.
4. Bring up the rubric judge (`exp_cudaforge/judge_serve.sh`, path-adjusted) and confirm
   `curl $RUBRIC_VLLM_URL` responds before launching a full training cycle — a dead judge
   endpoint fails training cycles late and confusingly rather than at startup.
