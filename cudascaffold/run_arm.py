"""Entry point: assemble the CUDA auto-scaffold arm and run the loop.

The Teacher's prompt and observation are NOT redefined here — they come from
`cudascaffold/observation.py`, which is the ALFWorld module unchanged. Everything domain-specific
reaches it through `CUDA_DOMAIN` (scaffold.py): the category labels, what one instance is and how
it is scored, whether reference solutions exist, whether text can be attached per instance. That
is the whole point of the Domain-descriptor pattern — swapping ALFWorld for CUDA kernels is a
data change, not a prompt rewrite.

What the Teacher sees each cycle (assembled by observation.assemble_observation):
  signals.per_task_gap[cat]    bare vs injected mean reward on the SAME train rows, this policy
  signals.zero_gradient_groups[cat] GRPO groups with no reward variance (all-fail or all-succeed):
                               gradient); this domain's equivalent of an all-fail rollout group
  signals.valid_seen           held-out mean reward on test.parquet, never scaffolded
  signals.eval_trajectory      that number across all cycles, with each point's draws
  decision_history             its own past proposals, the A/B verdict on each, and what
                               held-out success did afterwards

What it may change:
  text_ops  free-form text per scope: general | scratch | improve_l1 | improve_l2 | improve_l3
  p_ops     per-category injection probability, capped at P_MAX and rate-limited per cycle
Both go through the same gates as the ALFWorld arm: text must beat the live scaffold in a paired
A/B before it takes effect, and p edits submitted alongside rejected text are discarded with it.
"""
from __future__ import annotations

import os

# This repo trains with stock verl GRPO: the loss is conditioned on the prompt that was
# actually used, and there is no bare-prompt re-conditioning anywhere in it. The shared
# observation module derives its description of the loss from ARM_BARE_LOSS, so it must be
# False here or the Teacher is told a mechanism that does not exist — and every judgement it
# makes about what text can buy follows from that mechanism. Set before importing observation.
#
# Assigned, not setdefault: there is no legitimate other value for this arm, so an inherited
# ARM_BARE_LOSS=True — exported by a launch script, or left in a shell by a previous
# experiment — must not win. setdefault would silently keep it and the Teacher would reason
# correctly about a mechanism this repo does not have.
os.environ["ARM_BARE_LOSS"] = "False"

from . import adapters as A
from . import scaffold as S
from . import teacher as T
from . import loop as L

# Auto-detects the repo checkout (this file lives at <repo>/cudascaffold/run_arm.py) so a
# clone anywhere works with no edits; ARM_ROOT overrides explicitly if you need something else.
ROOT = os.environ.get("ARM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_DOM_NAME = os.environ.get("ARM_DOMAIN", "cuda")


def default_cfg():
    exp = os.environ.get("ARM_EXP", "cuda_scaffold_8b")
    exp_root = os.environ.get("ARM_EXP_ROOT", "/mnt/data1/zha00175/exp_cudaforge")
    root = f"{exp_root}/{exp}"
    work = f"{root}/work"
    os.makedirs(work, exist_ok=True)
    return {
        "exp": exp,
        "project": "CudaForge_AutoScaffold",
        # --- placement (user-assigned): train 0,1 | kernel benchmark 3 | rubric judge 7 ---
        "gpus": os.environ.get("ARM_GPUS", "0,1"),
        "n_gpus": int(os.environ.get("ARM_N_GPUS", "2")),
        "tp_size": int(os.environ.get("ARM_TP", "2")),
        "reward_gpu": os.environ.get("ARM_REWARD_GPU", "3"),
        "rubric_url": os.environ.get("RUBRIC_VLLM_URL", "http://127.0.0.1:8210/v1/chat/completions"),
        "rubric_model": os.environ.get("RUBRIC_MODEL_NAME", "rubric-judge"),
        "rubric_timeout": int(os.environ.get("RUBRIC_VLLM_TIMEOUT_SEC", "120")),
        # --- model / data ---
        "model": os.environ.get("ARM_MODEL",
                                "/mnt/data1/zha00175/models/drkernel-8b-coldstart"),
        # Defaults follow the DOMAIN: the triton domain trained on the CudaForge parquet once
        # already in a mis-set env, and nothing failed loudly — the categories still parsed,
        # only every domain fact was about the wrong task. (train_new.parquet is the CudaForge
        # dump carrying `level` and `task_name`; the Triton split carries both natively.)
        "train_file": os.environ.get("ARM_TRAIN_FILE",
                                     f"{ROOT}/dataset/Triton/train.parquet"
                                     if _DOM_NAME == "triton"
                                     else f"{ROOT}/dataset/CudaForge/train_new.parquet"),
        "val_file": os.environ.get("ARM_VAL_FILE",
                                   f"{ROOT}/dataset/Triton/test.parquet"
                                   if _DOM_NAME == "triton"
                                   else f"{ROOT}/dataset/CudaForge/test.parquet"),
        "reward_path": os.environ.get("ARM_REWARD", f"{ROOT}/cudaforge/reward_bench_rubric.py"),
        # --- optimisation (from experiments/Qwen3-32B_KL003_1e-6_Rubric.sh, scaled to 2 GPUs) ---
        "lora_rank": int(os.environ.get("ARM_LORA_RANK", "128")),
        "lora_alpha": int(os.environ.get("ARM_LORA_ALPHA", "128")),
        "lr": os.environ.get("ARM_LR", "1e-6"),
        "kl_loss_coef": os.environ.get("ARM_KL", "0.03"),
        "train_batch_size": int(os.environ.get("ARM_TRAIN_BS", "8")),
        "ppo_mini_batch_size": int(os.environ.get("ARM_MINI_BS", "4")),
        "micro_bs": int(os.environ.get("ARM_MICRO_BS", "1")),
        "rollout_n": int(os.environ.get("ARM_ROLLOUT_N", "6")),
        "gpu_mem": float(os.environ.get("ARM_GPU_MEM", "0.35")),
        "max_prompt_length": int(os.environ.get("ARM_MAX_PROMPT", "8192")),
        "max_response_length": int(os.environ.get("ARM_MAX_RESP", "16384")),
        "total_epochs": int(os.environ.get("ARM_EPOCHS", "20")),
        # --- loop knobs ---
        "steps_per_cycle": int(os.environ.get("ARM_K", "10")),
        "val_n": int(os.environ.get("ARM_VAL_N", "3")),
        "n_per_task": int(os.environ.get("ARM_NPT", "8")),
        # Which domain this arm trains. The descriptor decides the scaffold's scopes, what the
        # Teacher is told about the task, and how splice labels a row — hard-coding it meant a
        # second dataset could not be run without editing code.
        "domain": {"cuda": S.CUDA_DOMAIN, "triton": S.TRITON_DOMAIN}[_DOM_NAME],
        "teacher_priors": os.environ.get("ARM_PRIORS", "0") == "1",
        # How many cycles the scaffold may stay empty before the pre-check is bypassed. The
        # previous run declined 20/20 and ended having measured nothing about whether text helps.
        # Set to 0 to disable the floor and restore pure Teacher discretion.
        "intervene_floor_cycles": int(os.environ.get("ARM_FLOOR_CYCLES", "3")),
        "n_cycles": int(os.environ.get("ARM_N_CYCLES", "20")),
        "base_seed": 20260801,
        # --- paths ---
        "ckpt_root": os.environ.get(
            "ARM_CKPT",
            f"{os.environ.get('ARM_CKPT_ROOT', '/mnt/data1/zha00175/cuda_scaffold_ckpts')}/{exp}"),
        "work": work,
        "log_dir": f"{root}/logs",
        "scaffold_path": f"{root}/scaffold.json",
        "journal_path": f"{root}/journal.json",
        "state_path": f"{root}/state.json",
        "train_log": f"{root}/train.log",
        "ray_tmp": os.environ.get("ARM_RAY_TMP", f"/dev/shm/zray_{exp}"),
    }


# Everything the loop needs to resume as if it had never stopped. `train_rollouts` belongs here
# even though it is only telemetry: triage reads it as a SERIES, and the whole point of the series
# is spotting a category that has been at zero for several cycles. Dropping it on restart resets
# that memory to one cycle, so a restart quietly costs the pre-check the evidence it was given the
# series for. Restarts are not rare here — the watchdog does them, and so does every code change.
STATE_KEYS = ("cycle", "step", "scaffold", "sr_history", "best", "best_step",
              "decision_history", "last_eval", "train_rollouts",
              "teacher_unreachable_cycles")


def _load(path, default):
    import json
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def build_fns(cfg):
    os.makedirs(cfg["log_dir"], exist_ok=True)

    def log(msg):
        with open(f"{os.path.dirname(cfg['scaffold_path'])}/orch.log", "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    cfg["log"] = log

    def train_fn(scaf, frm, to):
        A.persist_scaffold(scaf, cfg["scaffold_path"])
        return A.train_adapter(scaf, frm, to, cfg)

    def eval_fn(ckpt, val_n):
        return A.eval_adapter(ckpt, val_n, cfg)

    def signals_fn(ckpt, scaf):
        """Signals WITHOUT a measurement pass.

        The dedicated bare+injected sweep is gone: it was two of the seven Ray+vLLM startups per
        cycle, and startups are GPU context create/destroy, the operation that wedged this node.
        Training already runs the same experiment — p_task decides injection per group, by a hash
        independent of difficulty — so the contrast is free. ARM_SIGNALS_PASS=1 restores the old
        behaviour for a side-by-side check.
        """
        free = cfg.get("_last_signals") or {}
        if os.environ.get("ARM_SIGNALS_PASS", "0") == "1":
            sig = A.signals_adapter(ckpt, scaf, cfg, seed=cfg["base_seed"] + _step_of(ckpt))
            sig["zero_gradient_groups"] = free.get("zero_gradient_groups") or {}
            sig["per_task_gap_from_training"] = free.get("per_task_gap") or {}
            return sig
        return {"per_task_gap": free.get("per_task_gap") or {},
                "zero_gradient_groups": free.get("zero_gradient_groups") or {},
                # From the NON-INJECTED training groups, so "worst instances under the bare
                # prompt" — which is what the observation calls them — stays literally true.
                "failures": (free.get("failures") or [])[:40],
                "successes": []}

    def measure_ab_fn(ckpt, cur, cand, tasks):
        return A.measure_ab_adapter(ckpt, cur, cand, tasks, cfg,
                                    seed=cfg["base_seed"] + _step_of(ckpt))

    def teacher_fn(obs, scaffold=None):
        return T.propose(obs, call_fn=T.openai_call, domain=cfg["domain"],
                         priors=cfg["teacher_priors"], scaffold=scaffold)

    # v2: the investigative Teacher (teacherflow kernel domain) — same decision grammar
    # and normalize() validation, different acquisition: budgeted read-only tools over the
    # cycle's scored candidates (recorder), transcript persisted per cycle.
    if os.environ.get("AUTOSCAFFOLD_TEACHER") == "v2":
        from . import teacher_v2 as T2
        teacher_fn = T2.make_teacher_fn(cfg)

    def triage_fn(obs):
        return T.triage(obs, call_fn=T.openai_call)

    return {"train_fn": train_fn, "eval_fn": eval_fn, "signals_fn": signals_fn,
            "measure_ab_fn": measure_ab_fn, "teacher_fn": teacher_fn, "triage_fn": triage_fn,
            "persist_fn": lambda s: A.persist_scaffold(s, cfg["scaffold_path"]),
            "journal_fn": lambda h: A.persist_scaffold(h, cfg["journal_path"]),
            "state_fn": lambda st: A.persist_scaffold(
                {k: st[k] for k in STATE_KEYS if k in st}, cfg["state_path"]),
            "log": log}


def _step_of(ckpt):
    try:
        return int(str(ckpt).rstrip("/").split("global_step_")[-1])
    except Exception:
        return 0


def main(n_cycles=20):
    cfg = default_cfg()
    fns = build_fns(cfg)
    saved = _load(cfg["state_path"], None)
    if saved and "step" in saved:
        state = {**L.new_state(), **saved}
        # A scaffold written before items existed has no `items` key, so every id-based edit is
        # impossible and the observation would show an empty item list beside non-empty injected
        # text — the Teacher would then "add" what is already being injected. Migration turns each
        # legacy per-scope string into the one item it always was. Cheap and idempotent, so it
        # runs on every resume rather than being gated on a version check that could go stale.
        state["scaffold"] = S.migrate_items(state["scaffold"], cfg["domain"])
        A.check_resume_consistency(state.get("step") or 0,
                                   os.path.join(cfg["ckpt_root"]))
        fns["log"](f"[autoscaffold] RESUME {cfg['exp']} at cycle {state.get('cycle')} "
                   f"step={state['step']} scaffold v{state['scaffold'].get('version')} "
                   f"best={state.get('best')}@{state.get('best_step')}")
    else:
        step0 = A.existing_ckpt_step(cfg)
        if step0:
            # No state.json yet the ckpt dir is not empty: either a deliberate continue after
            # the exp dir was wiped, or an EXP NAME COLLISION over a shared ckpt_root. Loud,
            # because the second reads as the first until valid_seen makes no sense.
            fns["log"](f"[autoscaffold] NOTE: no state.json but checkpoints exist at step "
                       f"{step0} under this exp name — continuing FROM THEM. If this was meant "
                       f"to be a fresh run, use a new ARM_EXP or point ARM_CKPT_ROOT elsewhere.")
        state = L.new_state(step0=step0, scaffold=S.empty_scaffold(cfg["domain"]))
        state["best_step"] = step0
        prior = _load(cfg["journal_path"], [])
        if isinstance(prior, list) and prior:
            state["decision_history"] = prior
            fns["log"](f"[autoscaffold] loaded {len(prior)} prior decisions as Teacher memory")
        A.persist_scaffold(state["scaffold"], cfg["scaffold_path"])
        fns["log"](f"[autoscaffold] arm={cfg['exp']} model={os.path.basename(cfg['model'])} "
                   f"train_gpus={cfg['gpus']} reward_gpu={cfg['reward_gpu']} "
                   f"scopes={cfg['domain'].scopes()} K={cfg['steps_per_cycle']} "
                   f"VAL_N={cfg['val_n']} empty scaffold")

    target = int(os.environ.get("ARM_TARGET_STEP", "0") or 0)
    if target:
        cfg["stop_fn"] = lambda st: st.get("step", 0) >= target
        fns["log"](f"[autoscaffold] absolute target step {target} "
                   f"(n_cycles={n_cycles} caps this process only)")
    return L.run(state, fns, cfg, n_cycles)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
