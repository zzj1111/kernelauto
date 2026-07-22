set -euo pipefail

# Run this from the repository root.

SESSION="Qwen_grpo"
ENGINE=vllm
SCRIPT_HGAE_7B="./experiments/train_exp_2.sh"

# Run A
SEED_A=1


tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "$SEED_A"
  tmux send-keys -t "$SESSION:$SEED_A" \
  "bash ${SCRIPT_HGAE_7B}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
