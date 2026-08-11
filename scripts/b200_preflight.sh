#!/usr/bin/env bash
# Preflight for running the kernel arm on a new machine (written for the B200 move).
# Checks only — it changes nothing, so it is safe to run on a box you share.
# Every FAIL here is something that made a run silently wrong or dead on a past machine.
set -u

PY="${ARM_PYTHON:-$(command -v python)}"
# Same discovery as setup_b200_env.sh: newest toolkit with a working nvcc, override wins.
# A hardcoded 12.9 here failed the documented setup->preflight flow on any box whose setup
# had just discovered a different toolkit.
if [ -z "${ARM_CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-12.9 $(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -rV); do
    [ -x "$c/bin/nvcc" ] && ARM_CUDA_HOME="$c" && break
  done
fi
CUDA_HOME="${ARM_CUDA_HOME:-/usr/local/cuda-12.9}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ok=0; bad=0
pass() { echo "  OK   $*"; ok=$((ok+1)); }
fail() { echo "  FAIL $*"; bad=$((bad+1)); }

echo "== GPU =="
CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ' ')
if [ -n "$CAPS" ]; then pass "compute capability: $CAPS(adapters.py auto-detects this)"
else fail "nvidia-smi returned nothing — no driver, or wrong container runtime"; fi

echo "== CUDA toolchain =="
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
  REL=$("$CUDA_HOME/bin/nvcc" --version | grep -oE "release [0-9]+\.[0-9]+" | grep -oE "[0-9]+\.[0-9]+")
  pass "nvcc $REL at $CUDA_HOME (override: ARM_CUDA_HOME)"
  case "$CAPS" in *10.*|*12.*)
    awk -v r="$REL" 'BEGIN{exit !(r+0 < 12.8)}' && \
      fail "sm_100+ GPU but nvcc $REL < 12.8 cannot target it — generated kernels will not compile" ;;
  esac
else
  fail "no nvcc at $CUDA_HOME/bin — set ARM_CUDA_HOME to a toolkit that can target $CAPS"
fi

echo "== Python env ($PY) =="
"$PY" - <<'PYEOF'
import importlib.util
import sys
for mod, why in [("torch", "training"), ("vllm", "rollouts and the judge"),
                 ("ray", "verl workers"), ("openai", "the Teacher client")]:
    if importlib.util.find_spec(mod) is None:
        print(f"  FAIL {mod} not importable ({why})"); sys.exit(1)
import torch
print(f"  OK   torch {torch.__version__} cuda={torch.version.cuda} "
      f"devices={torch.cuda.device_count()}")
if importlib.util.find_spec("flash_attn") is None:
    print("  WARN flash_attn not installed — verl FSDP training needs attention=sdpa, "
          "or pass FLASH_ATTN_WHEEL to setup_b200_env.sh")
if importlib.util.find_spec("flashinfer") is not None:
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    if cap >= (10, 0):
        print("  FAIL flashinfer is installed on sm_100+ — its prebuilt wheels broke vLLM on "
              "the B200 (same failure as the ALFWorld move). pip uninstall flashinfer-python")
        sys.exit(1)
    print("  OK   flashinfer present but GPU is pre-sm_100")
PYEOF
[ $? -eq 0 ] && ok=$((ok+1)) || bad=$((bad+1))

echo "== Repo inputs =="
for f in dataset/Triton/test.parquet dataset/CudaForge/test.parquet; do
  [ -f "$ROOT/$f" ] && pass "$f" || fail "$f missing — the repo ships it; incomplete clone?"
done

echo "== Models =="
if [ -n "${ARM_MODEL:-}" ] && [ -d "${ARM_MODEL:-}" ]; then pass "base model: $ARM_MODEL"
else fail "ARM_MODEL unset or not a directory. Pull the eos-fixed base first:
         hf download Mingyi-Hong/drkernel-8b-coldstart-eosfix --local-dir <dir>  (private; needs HF_TOKEN)"
fi
if [ -n "${JUDGE_MODEL:-}" ] && [ -e "${JUDGE_MODEL:-}" ]; then pass "judge model: $JUDGE_MODEL"
else fail "JUDGE_MODEL unset — e.g. hf download Qwen/Qwen3-8B (public), then scripts/serve_rubric_judge.sh"
fi

echo "== Teacher =="
if [ -n "${OPENAI_API_KEY:-}" ] || { [ -n "${AUTOSCAFFOLD_OPENAI_KEY_FILE:-}" ] && [ -f "$AUTOSCAFFOLD_OPENAI_KEY_FILE" ]; }; then
  pass "OpenAI credentials reachable (never put the key on a command line)"
else fail "no OPENAI_API_KEY and no AUTOSCAFFOLD_OPENAI_KEY_FILE — the Teacher cannot run"; fi

echo "== Scratch space =="
SHM_G=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
if [ "${SHM_G:-0}" -ge 50 ]; then pass "/dev/shm ${SHM_G}G free (Ray tmp + kernel build dirs live here; wiped on reboot)"
else fail "/dev/shm has ${SHM_G:-?}G free — Ray tmp plus per-candidate build dirs want >=50G"; fi

echo
echo "$ok ok, $bad failing. When everything passes:"
echo "  1. scripts/serve_rubric_judge.sh --model \"\$JUDGE_MODEL\"   (own GPU, port 8210)"
echo "  2. ARM_EXP=<name> ARM_GPUS=0,1 ARM_REWARD_GPU=3 ARM_MODEL=<base> scripts/launch_autoscaffold.sh <cycles>"
exit "$bad"
