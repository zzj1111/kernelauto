#!/usr/bin/env bash
# One entry point for the whole arm: judge + training + watchdog.
#
# First run: pass your machine's paths as flags — they are SAVED as the default config
# (an env.sh file, the same one the watchdog reads). Every later run reuses the saved
# defaults; any flag you pass again overrides that value and updates the saved file.
#
#   # first time (records everything, then launches):
#   scripts/start.sh --exp b200_run1 --target-step 200 \
#       --model ~/autoscaffold/models/drkernel-8b-coldstart-eosfix \
#       --judge-model ~/autoscaffold/models/Qwen3-8B --judge-gpu 7 \
#       --gpus 0,1 --reward-gpu 3 --ckpt-root /bigdisk/ckpts \
#       --openai-key-file ~/autoscaffold/openai.key  100
#
#   # afterwards:
#   scripts/start.sh            # same run, saved settings, default 100 cycles
#   scripts/start.sh --gpus 4,5 # one override, also saved for next time
#
#   --config FILE   where settings live (default $HOME/autoscaffold/env.sh); logs, HALT and
#                   watchdog state live next to it
#   --save-only     record settings and print them, launch nothing
#   --no-watchdog   launch training without the babysitter (not recommended overnight)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ARM_CONFIG:-$HOME/autoscaffold/env.sh}"
SAVE_ONLY=0
NO_WATCHDOG=0
N_CYCLES=100

# ---- flag -> variable table (single source for parsing, saving, and printing) ---------------
declare -A MAP=(
  [--model]=ARM_MODEL           [--judge-model]=JUDGE_MODEL
  [--gpus]=ARM_GPUS             [--n-gpus]=ARM_N_GPUS         [--tp]=ARM_TP
  [--reward-gpu]=ARM_REWARD_GPU [--judge-gpu]=JUDGE_GPU
  [--exp]=ARM_EXP               [--exp-root]=ARM_EXP_ROOT     [--ckpt-root]=ARM_CKPT_ROOT
  [--target-step]=ARM_TARGET_STEP [--domain]=ARM_DOMAIN       [--val-n]=ARM_VAL_N
  [--train-file]=ARM_TRAIN_FILE [--val-file]=ARM_VAL_FILE
  [--python]=ARM_PYTHON         [--cuda-home]=ARM_CUDA_HOME   [--root]=ARM_ROOT
  [--openai-key-file]=AUTOSCAFFOLD_OPENAI_KEY_FILE
  [--gpu-desc]=ARM_GPU_DESC     [--judge-gpu-mem]=JUDGE_GPU_MEM
)
ORDER=(ARM_ROOT ARM_PYTHON ARM_CUDA_HOME ARM_DOMAIN ARM_MODEL JUDGE_MODEL JUDGE_GPU
       ARM_EXP ARM_EXP_ROOT ARM_CKPT_ROOT ARM_TARGET_STEP ARM_VAL_N ARM_GPUS ARM_N_GPUS
       ARM_TP ARM_REWARD_GPU ARM_TRAIN_FILE ARM_VAL_FILE AUTOSCAFFOLD_OPENAI_KEY_FILE
       OPENAI_API_KEY ARM_GPU_DESC JUDGE_GPU_MEM)

declare -A OVERRIDE=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --save-only) SAVE_ONLY=1; shift ;;
    --no-watchdog) NO_WATCHDOG=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    -*)
      VAR="${MAP[$1]:-}"
      [ -n "$VAR" ] || { echo "unknown flag: $1 (see --help)" >&2; exit 1; }
      OVERRIDE[$VAR]="$2"; shift 2 ;;
    *) N_CYCLES="$1"; shift ;;
  esac
done

BASE="$(dirname "$CONFIG")"
mkdir -p "$BASE"

# ---- merge: saved config first, explicit flags override -------------------------------------
declare -A CUR=()
if [ -f "$CONFIG" ]; then
  while IFS= read -r ln; do
    [[ "$ln" == export\ *=* ]] || continue
    k="${ln#export }"; k="${k%%=*}"
    v="${ln#export "$k"=}"
    CUR[$k]="$v"
  done < "$CONFIG"
fi
for k in "${!OVERRIDE[@]}"; do CUR[$k]="${OVERRIDE[$k]}"; done

# `export OPENAI_API_KEY=...` in the calling shell is the preferred way to hand over the key:
# it is captured here into the saved config so watchdog relaunches inherit it, and the teacher
# reads the env var FIRST (before any key file). A freshly exported key replaces a saved one.
if [ -n "${OPENAI_API_KEY:-}" ]; then CUR[OPENAI_API_KEY]="$OPENAI_API_KEY"; fi

# first-run defaults, only where nothing is saved and no flag was given
: "${CUR[ARM_ROOT]:=$REPO_ROOT}"
[ -x "$HOME/kernel/bin/python" ] && : "${CUR[ARM_PYTHON]:=$HOME/kernel/bin/python}"
: "${CUR[ARM_DOMAIN]:=triton}"
: "${CUR[ARM_EXP_ROOT]:=$BASE/exp}"
: "${CUR[ARM_CKPT_ROOT]:=$BASE/ckpts}"
: "${CUR[ARM_VAL_N]:=3}"

MISSING=()
for k in ARM_MODEL ARM_EXP ARM_TARGET_STEP ARM_PYTHON; do
  [ -n "${CUR[$k]:-}" ] || MISSING+=("$k")
done
if [ -z "${CUR[OPENAI_API_KEY]:-}" ] && [ -z "${CUR[AUTOSCAFFOLD_OPENAI_KEY_FILE]:-}" ]; then
  MISSING+=("OPENAI_API_KEY(export 即可)或 --openai-key-file")
fi
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "FAIL: no saved value and no flag for: ${MISSING[*]}" >&2
  echo "      first run needs at least: --model --exp --target-step (+ --python if ~/kernel is elsewhere)" >&2
  exit 1
fi

# ---- save: this file IS the default settings, and the one watchdog reads --------------------
{
  echo "# written by scripts/start.sh $(date '+%Y-%m-%d %H:%M:%S') — edit by hand or override with flags"
  for k in "${ORDER[@]}"; do
    [ -n "${CUR[$k]:-}" ] && echo "export $k=${CUR[$k]}"
  done
} > "$CONFIG"
chmod 600 "$CONFIG"          # may hold the API key
echo "=== settings saved to $CONFIG ==="
grep -v "^#" "$CONFIG" | sed "s/\(OPENAI_API_KEY=\).\{6\}.*/\1******/"
[ "$SAVE_ONLY" = 1 ] && exit 0

# ---- launch ---------------------------------------------------------------------------------
set -a; source "$CONFIG"; set +a

for pdir in /proc/[0-9]*; do
  cmd="$(tr '\0' ' ' < "$pdir/cmdline" 2>/dev/null || true)"
  case "$cmd" in *python*" -m cudascaffold.run_arm"*)
    echo "FAIL: run_arm already alive (pid ${pdir##*/}); stop it before starting another" >&2
    exit 1 ;;
  esac
done

if [ -n "${JUDGE_GPU:-}" ] && [ -n "${JUDGE_MODEL:-}" ]; then
  if ! ss -tln | grep -q ":8210"; then
    echo "starting rubric judge on GPU $JUDGE_GPU ..."
    nohup bash "$REPO_ROOT/scripts/serve_rubric_judge.sh" --model "$JUDGE_MODEL" \
        --gpu "$JUDGE_GPU" >> "$BASE/judge.log" 2>&1 &
    for i in $(seq 1 60); do
      sleep 10
      ss -tln | grep -q ":8210" && break
      [ "$i" = 60 ] && { echo "FAIL: judge not up after 10min; see $BASE/judge.log" >&2; exit 1; }
    done
  fi
  echo "judge up on :8210"
fi

echo "starting training ($N_CYCLES cycles) -> $BASE/run.log"
nohup bash "$REPO_ROOT/scripts/launch_autoscaffold.sh" "$N_CYCLES" >> "$BASE/run.log" 2>&1 &

if [ "$NO_WATCHDOG" != 1 ]; then
  # exactly one watchdog per config: kill any previous one for THIS env file first
  for pdir in /proc/[0-9]*; do
    cmd="$(tr '\0' ' ' < "$pdir/cmdline" 2>/dev/null || true)"
    case "$cmd" in *python*watchdog.py*"$CONFIG"*) kill "${pdir##*/}" 2>/dev/null || true ;; esac
  done
  nohup "$ARM_PYTHON" "$REPO_ROOT/scripts/watchdog.py" --env-file "$CONFIG" \
      ${JUDGE_GPU:+--judge-gpu "$JUDGE_GPU"} >/dev/null 2>&1 &
  echo "watchdog up (5-min patrol; log: $BASE/watchdog.log)"
fi

echo
echo "watch:  tail -f $BASE/run.log"
echo "state:  ${ARM_EXP_ROOT}/${ARM_EXP}/state.json"
echo "halt:   touch $BASE/HALT  (watchdog stops relaunching; rm to resume)"
