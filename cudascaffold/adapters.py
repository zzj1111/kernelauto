"""The side-effecting functions the loop needs, wired to StitchCUDA's verl + reward.

Shape mirrors the ALFWorld arm (agent_system/skill_opt/autoscaffold/adapters.py) so the
domain-agnostic loop/gates/observation/teacher carry over unchanged. What differs is entirely
below the interface:

  - There is no environment manager to intercept, so the scaffold reaches the policy by
    REWRITING THE TRAINING PARQUET each cycle (see splice.py). Core verl stays untouched.
  - Reward is a subprocess that compiles the candidate with nvcc, checks it against the PyTorch
    reference, and times it. Correctness is a hard gate; speedup is the magnitude.
  - Held-out eval is verl's own validation pass over test.parquet, which is NEVER scaffolded —
    so the eval anchor is bare by construction rather than by a flag we could forget to set.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

# All overridable via env var; every default below is what the source machine happens to use,
# not a requirement. See UV_INSTALL.md / scripts/launch_autoscaffold.sh for how to set these on
# a different box instead of editing this file.
ROOT = os.environ.get("ARM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Default: whatever python is running this process (correct as long as the launch script has
# already activated the right venv / put it first on PATH). ARM_PYTHON overrides explicitly;
# the /dev/shm-or-conda fallback is only there for the source machine's own historical setup.
VENV_PY = os.environ.get("ARM_PYTHON") or (
    "/dev/shm/verl_env/bin/python" if os.path.exists("/dev/shm/verl_env/bin/python")
    else "/mnt/data1/zha00175/miniconda/envs/verl/bin/python" if
    os.path.exists("/mnt/data1/zha00175/miniconda/envs/verl/bin/python")
    else sys.executable)

# Toolchain facts established by measurement, not assumption (see exp_cudaforge/DECISIONS.md):
#   - the system default nvcc is often too old to target the GPU's compute capability.
#   - torch looks for the ninja EXECUTABLE on PATH, so the venv's bin must be on it.
CUDA_HOME = os.environ.get("ARM_CUDA_HOME", "/usr/local/cuda-12.9")
CUDA_BIN = os.environ.get("ARM_CUDA_BIN", f"{CUDA_HOME}/bin")


def _detect_arch_list():
    """Ask the driver for the GPUs' compute capability instead of assuming one.

    base_env exports TORCH_CUDA_ARCH_LIST into every subprocess, and kernel_runner treats the
    env as authoritative over its own detection — so a wrong value here compiles every generated
    kernel for the wrong architecture. A cubin does not run across major versions: on a B200
    (sm_100) the old hardcoded "9.0" would make every candidate fail to load, score 0.0, and
    look exactly like a model that cannot write CUDA. Detection failing falls back to "9.0"
    loudly; ARM_TORCH_CUDA_ARCH_LIST still overrides everything for cross-compiling setups.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        caps = sorted({ln.strip() for ln in out.stdout.splitlines() if ln.strip()})
        if out.returncode == 0 and caps and all(re.fullmatch(r"\d+\.\d+", c) for c in caps):
            return ";".join(caps)
    except (OSError, subprocess.TimeoutExpired):
        pass
    print("[adapters] WARNING: could not detect GPU compute capability; "
          "defaulting TORCH_CUDA_ARCH_LIST=9.0 — set ARM_TORCH_CUDA_ARCH_LIST if that is wrong",
          flush=True)
    return "9.0"


TORCH_CUDA_ARCH_LIST = os.environ.get("ARM_TORCH_CUDA_ARCH_LIST") or _detect_arch_list()
VENV_BIN = os.path.dirname(VENV_PY)

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Rows per validation wave in a measurement pass. Sized so the reward's in-flight work stays
# well under what wedged the driver (85 concurrent) and near what training sustains safely
# (48 candidates per step, spread over eight scorer processes). Overridable per site.
_MEASURE_BATCH_ROWS = int(os.environ.get("ARM_MEASURE_BATCH_ROWS", "36"))

# Key under which worker_log_offsets stores the ray_tmp root, so _read_since can re-glob the tree
# at read time instead of trusting the directories that happened to exist at snapshot time.
_ROOT_KEY = "__ray_tmp_root__"


class StepFailed(RuntimeError):
    """A subprocess did not produce the artifact the loop needs.

    Raised rather than letting the loop advance on a checkpoint that was never written: without
    it the loop reports a stale score as this cycle's result and asks the Teacher to explain a
    number that no training produced.
    """


def base_env(cfg):
    """Environment every GPU subprocess needs. Assembled in one place so the training run, the
    eval run and any ad-hoc probe cannot disagree about the toolchain."""
    env = dict(os.environ)
    env["PATH"] = f"{VENV_BIN}:{CUDA_BIN}:" + env.get("PATH", "")
    env["CUDA_HOME"] = CUDA_HOME
    env["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST
    env["CUDA_VISIBLE_DEVICES"] = cfg["gpus"]
    env["REWARD_CUDA_VISIBLE_DEVICES"] = cfg["reward_gpu"]
    env["RUBRIC_VLLM_URL"] = cfg["rubric_url"]
    env["RUBRIC_MODEL_NAME"] = cfg["rubric_model"]
    env["RUBRIC_VLLM_TIMEOUT_SEC"] = str(cfg.get("rubric_timeout", 120))
    if os.environ.get("ARM_WANDB") != "0":
        has_cred = bool(os.environ.get("WANDB_API_KEY")) or \
            os.path.exists(os.path.expanduser("~/.netrc"))
        mode = os.environ.get("WANDB_MODE") or ("online" if has_cred else "offline")
        if mode == "online" and not has_cred:
            print("[autoscaffold] WANDB_MODE=online requested but no WANDB_API_KEY/.netrc — "
                  "falling back to offline (upload later with wandb sync)")
            mode = "offline"
        env["WANDB_MODE"] = mode
        env.setdefault("WANDB_DIR", cfg["log_dir"])
        env.setdefault("WANDB_ENTITY", os.environ.get("WANDB_ENTITY",
                                                      "mhong-university-of-minnesota"))
    env["RAY_DEDUP_LOGS"] = "0"
    env["HYDRA_FULL_ERROR"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Keep Ray's session dirs off the root filesystem: it is shared with every other user and
    # has repeatedly sat at 100%, which killed a run on the ALFWorld arm.
    env.setdefault("RAY_TMPDIR", cfg["ray_tmp"])
    os.makedirs(cfg["ray_tmp"], exist_ok=True)
    return env


def _run(cmd, logpath, env, cwd=ROOT):
    os.makedirs(os.path.dirname(os.path.abspath(logpath)), exist_ok=True)
    with open(logpath, "a") as f:
        f.write(f"\n\n===== {cmd} =====\n")
        f.flush()
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                              env=env, cwd=cwd)


def ckpt_dir(cfg, step):
    return os.path.join(cfg["ckpt_root"], f"global_step_{step}")


def ckpt_is_usable(path):
    """True only for a checkpoint that FINISHED writing.

    verl creates `global_step_N/actor/` and starts streaming shards long before the save is
    complete — observed live at step 10: the two 17 GB model shards existed and were still
    growing while optim/extra_state/data.pt did not exist yet, and the tracker file still read 0.
    An existence check would have called that resumable.

    Accepting a torn checkpoint is not a cosmetic error: train_adapter would skip retraining,
    the eval that follows would score an older step, and the Teacher would be asked to explain a
    number no training produced.

    Requires the full set verl actually writes for this config, verified against a completed
    save: model + optim + extra_state for EVERY rank of the world size, plus the trainer's
    data.pt (written last, which is what makes it a usable completion marker).
    """
    import glob
    import re as _re
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "data.pt")):
        return False
    actor = os.path.join(path, "actor")
    if not os.path.isdir(actor):
        return False
    models = glob.glob(os.path.join(actor, "model_world_size_*_rank_*.pt"))
    if not models:
        return False
    sizes = {int(m.group(1)) for m in
             (_re.search(r"model_world_size_(\d+)_rank_\d+\.pt$", os.path.basename(f))
              for f in models) if m}
    if len(sizes) != 1:            # shards from two world sizes -> not trustworthy
        return False
    ws = sizes.pop()
    for kind in ("model", "optim", "extra_state"):
        for rank in range(ws):
            if not os.path.isfile(os.path.join(actor, f"{kind}_world_size_{ws}_rank_{rank}.pt")):
                return False
    return True


def check_resume_consistency(state_step, ckpt_dir, allow_env="ARM_ALLOW_STATE_CKPT_MISMATCH"):
    """Refuse to resume when state.json and the checkpoints disagree.

    verl's resume_mode=auto silently starts from BASE weights when default_local_dir holds no
    usable checkpoint. Combined with a state.json that says step=N, that produces a run whose
    scaffold, journal and step counter continue from cycle N/2 while the policy restarted from
    zero — every signal the Teacher reads is then attributed to weights that do not exist.
    The same shape happens in reverse when an exp name is reused over a ckpt_root holding
    SOMEONE ELSE'S newer checkpoints. Both are config mistakes; fail loudly at launch instead
    of surfacing as an inexplicable valid_seen cliff five hours in.
    """
    if not state_step:
        return
    latest_f = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    latest = None
    if os.path.exists(latest_f):
        try:
            latest = int(open(latest_f).read().strip())
        except ValueError:
            pass
    if latest == state_step:
        return
    if os.environ.get(allow_env) == "1":
        print(f"[autoscaffold] WARNING: state step={state_step} vs checkpoint latest={latest} "
              f"under {ckpt_dir} — proceeding because {allow_env}=1")
        return
    raise SystemExit(
        f"REFUSING TO RESUME: state.json says step={state_step} but {ckpt_dir} has "
        f"latest={latest}. Continuing would train from the wrong weights while the journal "
        f"claims otherwise. Fix the ckpt_root/exp mismatch (or set {allow_env}=1 if this "
        f"divergence is intentional).")


def existing_ckpt_step(cfg):
    import glob
    best = 0
    for d in glob.glob(os.path.join(cfg["ckpt_root"], "global_step_*")):
        m = re.search(r"global_step_(\d+)$", os.path.basename(d))
        if m and ckpt_is_usable(d):
            best = max(best, int(m.group(1)))
    return best


def _train_cmd(cfg, train_file, to_step, val_before=False, test_freq=-1):
    """The verl invocation. Derived from experiments/Qwen3-32B_KL003_1e-6_Rubric.sh, with the
    model, GPU count and batch sizes moved into cfg so the arm is one config away from a
    different size.

    test_freq defaults to -1, not to a large number. ray_trainer.py:1377-1380 validates when
    `test_freq > 0 and (is_last_step or step % test_freq == 0)` — a large test_freq only defeats
    the modulo, and every training pass ends on its last step, so 999999 still ran a full
    held-out validation at the end of each cycle. Those reward lines land inside the window
    train_adapter snapshots for the Teacher, which is how 96 training rollouts read as 280:
    two thirds of the Teacher's per-category signal was the held-out set it is not allowed to
    see. The eval and A/B passes pass test_freq=1 explicitly and go through val_only, which
    ray_trainer.py:1128-1133 gates on val_before_train alone — they are unaffected."""
    return " ".join([
        f"{VENV_PY} -m verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files={cfg['val_file']}",
        f"data.train_batch_size={cfg['train_batch_size']}",
        # Load the dataset in the trainer process. verl's default of 8 dataloader workers forks
        # the trainer eight more times, and the trainer here is an 8B model plus a vLLM engine —
        # measured at 174 GB of host RSS in a two-step smoke, where a worker was then killed by
        # signal mid-run. The dataset is one file of short prompts, so the workers buy nothing.
        # The sibling ALFWorld arm carries the same line for the same reason, after an OOM kill
        # during a checkpoint save took a whole node with it.
        # No leading '+': this key already exists in this repo's config (see
        # verl/trainer/config/_generated_ppo_trainer.yaml), and hydra rejects appending to an
        # item that is already there. The ALFWorld arm needs the '+' because its verl does not
        # define it — same setting, different override syntax, and the difference is a hard
        # failure at startup rather than a silent one.
        f"data.dataloader_num_workers={cfg.get('dataloader_workers', 0)}",
        # Score each candidate ONCE. This verl has two reward paths and both are wired to our
        # compute_score: the agent loop calls reward_loop_worker.compute_score per rollout as it
        # finishes (agent_loop.py:557-578, gated on reward_model.use_reward_loop, which upstream
        # defaults to true), and the reward manager scores the batch again afterwards. Measured
        # on the 2026-08-10 A/B: 540 rows produced 1084 bench() invocations — every kernel
        # compiled, run and timed twice.
        #
        # That is not just 2x the cost. Each invocation forks a runner that creates a CUDA
        # context, and context creation is exactly what deadlocked the driver, so the duplicate
        # path doubled the load that took the node down. It also doubled the gate's n, which is
        # harmless only while the acceptance margin is off.
        #
        # The manager path is the one kept, because the controller's per-category and
        # per-instance signals are scraped from the line compute_score prints under it.
        # NOT disabled. An earlier commit turned this off believing the reward manager was a
        # duplicate path; it is not — with reward_model.reward_manager=dapo the manager crashes
        # on its first item (dapo.py:120 reads self.overlong_buffer_cfg.enable, which defaults
        # to None), so the reward loop is the ONLY path that scores anything here. Turning it
        # off produced "Error in reward_fn: 'NoneType' object has no attribute 'enable'" and no
        # checkpoint. The duplication that motivated the change is real but has another cause,
        # still being traced.
        # "reward_model.use_reward_loop=False",
        f"data.max_prompt_length={cfg['max_prompt_length']}",
        f"data.max_response_length={cfg['max_response_length']}",
        f"actor_rollout_ref.model.path={cfg['model']}",
        f"actor_rollout_ref.model.lora_rank={cfg['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={cfg['lora_alpha']}",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=True",
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_loss_coef']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['micro_bs']}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg['rollout_n']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['tp_size']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['gpu_mem']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg['micro_bs']}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={cfg['micro_bs']}",
        "actor_rollout_ref.rollout.temperature=0.9",
        "actor_rollout_ref.rollout.top_k=20",
        "actor_rollout_ref.rollout.top_p=0.95",
        "actor_rollout_ref.actor.clip_ratio_low=0.2",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
        "reward_model.enable=False",
        "reward_model.reward_manager=dapo",
        f"custom_reward_function.path={cfg['reward_path']}",
        f"trainer.val_before_train={'True' if val_before else 'False'}",
        f"trainer.n_gpus_per_node={cfg['n_gpus']}",
        "trainer.nnodes=1",
        f"trainer.default_local_dir={cfg['ckpt_root']}",
        f"trainer.save_freq={cfg['steps_per_cycle']}",
        # A checkpoint is ~36G and every cycle writes one; unbounded retention filled the 868G
        # root disk to 100% on 2026-08-10 and killed the save in progress. Two is the minimum
        # with a fallback: verl advances latest_checkpointed_iteration.txt only after a COMPLETED
        # save, so keeping N and N-1 covers a save cut off mid-write.
        f"trainer.max_actor_ckpt_to_keep={cfg.get('max_ckpt_keep', 2)}",
        f"trainer.test_freq={test_freq}",
        f"trainer.total_training_steps={to_step}",
        "trainer.resume_mode=auto",
        # wandb is ON by default (ARM_WANDB=0 disables). Mode is decided in base_env: online
        # when credentials exist, offline fallback (with a loud note) when they do not — a
        # run must not die for telemetry.
        ("trainer.logger='[\"console\"]'"
         if os.environ.get("ARM_WANDB") == "0" else "trainer.logger='[\"console\",\"wandb\"]'"),
        f"trainer.project_name={cfg['project']}",
        f"trainer.experiment_name={cfg['exp']}",
        f"trainer.total_epochs={cfg['total_epochs']}",
    ])


def train_adapter(scaffold, from_step, to_step, cfg):
    """Splice the scaffold into a fresh training parquet, then advance training to `to_step`.

    The parquet is rebuilt every cycle rather than edited in place so the exact input of each
    cycle stays on disk and a decision can be re-read later against the data it was made on.
    """
    from . import splice as SP

    dst = os.path.join(cfg["work"], f"train_s{to_step}.parquet")
    info = SP.build(cfg["train_file"], dst, scaffold,
                    seed=cfg["base_seed"] + to_step, mode="full")
    cfg.get("log", lambda *a: None)(
        f"[train] step {from_step}->{to_step}: injected {info['n_injected']}/{info['n_rows']} rows "
        f"{ {k: f'{v[0]}/{v[1]}' for k, v in info['per_level'].items()} }")

    # Snapshot the reward's own prints around the training stretch. Every rollout of this cycle is
    # scored anyway, so aggregating those lines by category costs nothing and yields thousands of
    # samples per category — against the eight the paid measurement pass used to provide.
    #
    # This is what breaks the chicken-and-egg in the pre-check: triage is told to look for a
    # category that is stuck, but the per-category breakdown only existed AFTER triage approved
    # the expensive pass. Now the cheap question gets a real answer, from data that already
    # exists, and one that does not depend on the noise-dominated held-out curve at all.
    off = worker_log_offsets(cfg)
    rc = _run(_train_cmd(cfg, dst, to_step), cfg["train_log"], base_env(cfg))
    out = ckpt_dir(cfg, to_step)
    if not ckpt_is_usable(out):
        raise StepFailed(
            f"training {from_step}->{to_step} wrote no usable checkpoint at {out} "
            f"(subprocess rc={rc.returncode}); see {cfg['train_log']}")
    try:
        cfg["_last_train_rollouts"] = per_category_from_log(off)
        # The free bare-vs-injected contrast, computed from the rollouts this cycle just produced.
        # `info["fired"]` is what splice.build actually did, row by row. Nothing is re-derived,
        # so the split cannot drift from the injection rule.
        cfg["_last_signals"] = signals_from_training(
            off, info.get("fired") or {}, cfg.get("rollout_n"), scaffold, cfg.get("domain"))
        cfg["_last_zero_gradient"] = cfg["_last_signals"]["zero_gradient_groups"]
    except Exception as e:                      # never let telemetry break a completed training run
        cfg["_last_train_rollouts"] = {}
        cfg["_last_signals"] = {}
        cfg["_last_zero_gradient"] = {}
        cfg.get("log", lambda *a: None)(f"[train] rollout stats unavailable: {str(e)[:120]}")
    return out


# One validation metric line, tolerant of two things this build does that the original pattern
# assumed away (both observed at step 10, 2026-08-01):
#   1. the value arrives as a numpy repr — `...mean@1:np.float64(0.4441)`, not `: 0.4441` —
#      because verl formats the metrics dict with repr() rather than float();
#   2. the per-source metric under val-core is spelled `acc`, and only val-aux carries `reward`.
# Matching one fixed spelling made this return (None, {}) on a log that in fact held the number.
_VAL_METRIC = re.compile(
    r"val-(?:core|aux)/([\w\-./]+?)/(reward|acc|score)/mean(?:@\d+)?:\s*"
    r"(?:np\.\w+\(\s*)?([-+0-9.eE]+)")


def parse_val(logpath):
    """(overall, {category: rate}) from the last validation block verl printed.

    Returns (None, {}) only when the log genuinely holds no validation metric. That distinction
    matters: eval_adapter raises StepFailed on it and the watchdog then cold-restarts the arm, so
    a parser that is merely too strict costs a full eval (three draws, ~45 min) and the cycle's
    Teacher decision. `_measure_pass` shares this parser and fails SILENTLY instead — empty
    per-source dicts reach the Teacher as an empty per_task_gap and an empty all_fail_groups,
    which reads as "no signal" rather than "not measured". Keep this permissive.
    """
    try:
        txt = ANSI.sub("", open(logpath, errors="ignore").read())
    except Exception:
        return None, {}
    by_kind = {}
    for m in _VAL_METRIC.finditer(txt):
        # Last occurrence wins: a log may contain both the val_before pass and the end-of-run one.
        by_kind.setdefault(m.group(2), {})[m.group(1)] = float(m.group(3))
    # `reward` is the trained objective; `acc`/`score` are only stand-ins for builds that do not
    # emit it, so never let a stand-in shadow the real thing.
    per = by_kind.get("reward") or by_kind.get("acc") or by_kind.get("score") or {}
    overall = None
    if per:
        overall = sum(per.values()) / len(per)
    else:
        for pat in (r"val/reward:\s*(?:np\.\w+\(\s*)?([-+0-9.eE]+)",):
            got = re.findall(pat, txt)
            if got:
                overall = float(got[-1])
                break
    return overall, per


def eval_adapter(checkpoint, val_n, cfg):
    """Standalone eval on the held-out file, averaged over `val_n` draws.

    test.parquet is never scaffolded, so this is bare by construction.

    The draws are what makes the spread between them a usable read on measurement noise. They do
    NOT differ by sampling temperature — val_kwargs.temperature is 0 and this used to say
    otherwise. What separates them is the batch they land in, measured directly against this
    repo's own vLLM: 16 concurrent copies of one prompt at temperature 0 produced 5 distinct
    completions, while five strictly serial repeats produced 5 identical ones. Since that is the
    whole source of the spread, copies inside ONE pass reproduce it, which is why _eval_merged is
    the default — three passes were three GPU context create/destroy cycles for one question.
    ARM_EVAL_SEPARATE_DRAWS=1 restores one subprocess per draw.
    """
    step = int(str(checkpoint).rstrip("/").split("global_step_")[-1])
    if val_n > 1 and os.environ.get("ARM_EVAL_SEPARATE_DRAWS", "0") != "1":
        return _eval_merged(checkpoint, step, val_n, cfg)
    # Row-tag a copy of the val file so verl's padding phantoms collapse in the scrape below,
    # exactly as _eval_merged and measure_ab_adapter do. Without this the "trustworthy"
    # per-category numbers carried the same double-count bug the A/B was cured of.
    import pandas as pd
    df = pd.read_parquet(cfg["val_file"])
    rows = []
    for x in df.extra_info:
        e = dict(x)
        if e.get("task_name"):
            e["task_name"] = f"{e['task_name']}{_ROW_MARK}{len(rows)}"
        rows.append(e)
    df["extra_info"] = rows
    tagged_val = os.path.join(cfg["work"], f"eval_s{step}_tagged.parquet")
    df.to_parquet(tagged_val, index=False)

    draws, pers, pers_n = [], [], []
    for d in range(val_n):
        elog = os.path.join(cfg["log_dir"], f"{cfg['exp']}_eval_s{step}_d{d}.log")
        env = base_env(cfg)
        env["VLLM_SEED"] = str(d)
        off = worker_log_offsets(cfg)
        cmd = _train_cmd(cfg, tagged_val, step, val_before=True, test_freq=1)
        # val_only evaluates data.val_files, not the train_file slot — point it at the
        # tagged copy THIS pass built (same idiom as measure_ab below). Without this the
        # eval scores whatever cfg['val_file'] happens to be; it only ever worked when a
        # stale tagged file sat at that path, and a fresh pod (B200, step 70) broke it.
        cmd = cmd.replace(f"data.val_files={cfg['val_file']}", f"data.val_files={tagged_val}")
        _run(cmd + " trainer.val_only=True", elog, env)
        sr, per = parse_val(elog)
        # parse_val keys on verl's data_source, and this dataset has ONE of those — a breakdown
        # with one key cannot say which category is weak, so the Teacher was diagnosing
        # categories from the training rollouts alone (n≈12-24) while 180 bare held-out answers
        # sat unread in the worker logs. Scrape those instead; data_source stays the fallback.
        # A draw only counts when BOTH reads landed: mixing category keys from one draw with
        # data_source keys from another averages every key against a phantom 0.0 in the
        # other's draws, halving every number the Teacher sees.
        if sr is None:
            continue
        by_cat = per_category_from_log(off)
        if by_cat:
            per = {c: v["correct_rate"] for c, v in by_cat.items()}
            pers_n.append({c: v["n"] for c, v in by_cat.items()})
        draws.append(sr)
        pers.append(per)
    if not draws:
        raise StepFailed(
            f"eval at step {step} parsed no validation reward from any of {val_n} draws; "
            f"see {cfg['log_dir']}/{cfg['exp']}_eval_s{step}_d*.log")
    avg = round(sum(draws) / len(draws), 4)
    keys = sorted({k for p in pers for k in p})
    per_task = {k: round(sum(p.get(k, 0.0) for p in pers) / len(pers), 4) for k in keys}
    out = {"avg": avg, "per_task": per_task, "draws": [round(x, 4) for x in draws],
           "n_draws": len(draws), "n_draws_requested": val_n}
    if len(pers_n) == len(pers) and pers_n:
        out["per_task_n"] = {k: max(p.get(k, 0) for p in pers_n) for k in keys}
        out["per_task_unit"] = ("held-out correct-rate per category, bare prompt; "
                                "n = candidates scored per draw")
    return out


def _eval_merged(checkpoint, step, val_n, cfg):
    """All `val_n` draws in ONE pass instead of one subprocess each.

    Three draws were three full Ray+vLLM startups — three GPU context create/destroy cycles for
    what is one question. Startups are the operation that wedged this node (see
    reward_bench_rubric._MAX_CONCURRENT), and eval was the single biggest source of them: 24 of
    the 42 phase logs in eight cycles.

    The draws still measure what they measured. They were never separated by anything but the
    batch they landed in: val_kwargs.temperature is 0, so the spread between draws comes from
    batch composition, which was verified directly against this repo's own vLLM (16 concurrent
    copies of one prompt at temperature 0 gave 5 distinct completions; five strictly serial
    repeats gave 5 identical ones). Copies inside one pass land in different batch positions for
    exactly the same reason copies across passes did.

    Each copy carries its draw index in `category` so one pass can be taken apart again — the
    same encoding the merged A/B uses.
    """
    import pandas as pd
    from . import splice as SP

    src = pd.read_parquet(cfg["val_file"])
    frames = []
    for d in range(val_n):
        df = src.copy()
        rows = []
        for x in df.extra_info:
            e = dict(x)
            cat = SP.level_of(e, None) or "unlabelled"
            e["category"] = f"{cat}{_ARM_SEP}d{d}"
            # Row id for the same reason measure_ab_adapter stamps one: verl pads validation
            # batches by repeating rows, and an untagged repeat is indistinguishable from the
            # legitimate copy of the same task in another draw. Guarded like measure_ab: a
            # row with no task_name must not become the literal string "None#r0@@d0".
            if e.get("task_name"):
                e["task_name"] = f"{e['task_name']}{_ROW_MARK}{len(rows)}{_ARM_SEP}d{d}"
            rows.append(e)
        df["extra_info"] = rows
        frames.append(df)
    merged = os.path.join(cfg["work"], f"eval_s{step}_merged.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(merged, index=False)

    elog = os.path.join(cfg["log_dir"], f"{cfg['exp']}_eval_s{step}_merged.log")
    env = base_env(cfg)
    env["VLLM_SEED"] = str(cfg.get("base_seed", 0) + step)
    off = worker_log_offsets(cfg)
    cmd = _train_cmd(cfg, merged, step, val_before=True, test_freq=1)
    # Same fix as the per-draw eval above: the merged multi-draw parquet must BE the
    # validation set, not ride along in the train_files slot where val_only ignores it.
    cmd = cmd.replace(f"data.val_files={cfg['val_file']}", f"data.val_files={merged}")
    _run(cmd + " trainer.val_only=True", elog, env)

    by_key = per_category_from_log(off)
    per_draw = {}
    for key, v in by_key.items():
        if _ARM_SEP not in key:
            continue
        cat, tag = key.rsplit(_ARM_SEP, 1)
        per_draw.setdefault(tag, {})[cat] = (v["correct_rate"], v["n"])
    if not per_draw:
        raise StepFailed(
            f"merged eval at step {step} parsed no per-category reward; see {elog}")

    draws, pers = [], []
    for tag in sorted(per_draw):
        cats = per_draw[tag]
        n = sum(c[1] for c in cats.values())
        if not n:
            continue
        draws.append(round(sum(c[0] * c[1] for c in cats.values()) / n, 4))
        pers.append({c: round(v[0], 4) for c, v in cats.items()})
    if not draws:
        raise StepFailed(f"merged eval at step {step} produced no usable draw; see {elog}")
    avg = round(sum(draws) / len(draws), 4)
    keys = sorted({k for p in pers for k in p})
    per_task = {k: round(sum(p.get(k, 0.0) for p in pers) / len(pers), 4) for k in keys}
    per_task_n = {k: max((dict(per_draw[t]).get(k, (0, 0))[1] for t in per_draw), default=0)
                  for k in keys}
    cfg.get("log", lambda *a: None)(
        f"[eval] step {step}: {val_n} draws in ONE pass ({len(draws)} parsed) draws={draws}")
    return {"avg": avg, "per_task": per_task, "draws": draws,
            "n_draws": len(draws), "n_draws_requested": val_n,
            "per_task_n": per_task_n,
            "per_task_unit": ("held-out correct-rate per category, bare prompt; "
                              "n = candidates scored per draw")}


# `category:` is optional so logs written before it was echoed still parse.
REWARD_LINE = re.compile(
    r"correctness:\s*(\d+),\s*speedup:\s*([0-9.eE+-]+),\s*data_source:(\S+?),\s*"
    r"task_name:(\S+?),\s*level:([^,\s]+)(?:,\s*category:([^,\s]+))?"
    # `reward` is what actually reached the optimiser, and it is NOT implied by correctness: a
    # correct, fast kernel zeroed by the rubric's major_hacking flag reports correctness 1 with
    # reward 0. Optional so logs written before the field existed still parse.
    r"(?:,\s*reward:([0-9.eE+-]+))?")


def _one_path_per_file(paths):
    """Collapse paths that are the same file on disk, keeping the real one.

    Ray leaves a `session_latest` SYMLINK beside the real `session_<timestamp>` directory, and
    both match `session_*`. Globbing therefore returns every worker log twice, and reading both
    counted every reward line twice: the 2026-08-10 A/B reported n around 400 per arm for 180
    problems. Nothing was scored twice — only counted twice — but the gate's sample size is what
    a margin would be computed from, so a doubled n makes the acceptance bar too easy by about
    sqrt(2).

    Identity is (device, inode), which also collapses hard links, not just the symlink Ray
    happens to create today.
    """
    seen, keep = {}, []
    for p in sorted(paths):
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        prev = seen.get(key)
        # Prefer the path that is not reached through a symlink, so offsets recorded before and
        # after a snapshot use the same name for the same bytes.
        if prev is None or (os.path.realpath(p) == p and os.path.realpath(prev) != prev):
            seen[key] = p
    return sorted(seen.values())


def _worker_logs(ray_tmp, _pattern=None):
    """Every Ray worker stdout under a ray tmp root, each file listed once."""
    import glob
    pat = os.path.join(*_pattern) if _pattern else os.path.join(
        ray_tmp, "ray", "session_*", "logs", "worker-*.out")
    return _one_path_per_file(glob.glob(pat))


def worker_log_offsets(cfg):
    """Byte offset of every Ray worker stdout right now.

    The reward runs inside Ray workers, and Ray captures their stdout into its own per-worker
    files rather than the trainer's log — so the `correctness: ... task_name: ...` lines never
    reach train.log. Snapshotting offsets before a measurement pass and reading only what is
    appended after it is what separates that pass's candidates from the training rollouts
    interleaved in the same files.
    """
    out = {_ROOT_KEY: cfg["ray_tmp"]}
    for f in _worker_logs(cfg["ray_tmp"]):
        try:
            out[f] = os.path.getsize(f)
        except OSError:
            pass
    return out


def _read_since(offsets):
    """Everything appended to any Ray worker stdout since the snapshot.

    The glob runs at READ time over the whole ray_tmp tree, not over the directories that existed
    when the snapshot was taken. Every phase calls ray.init() and gets a FRESH session_* directory,
    so a reader that only revisits the snapshot's directories looks in the session that has already
    finished and reads nothing new. Two measured failures of the old version:
      - training wrote to a new session -> 0 of its lines were seen;
      - the first cycle snapshotted an empty tree -> `pats` was empty, so nothing was read at all.
    A file absent from `offsets` is read from byte 0, which is correct: it did not exist at
    snapshot time, so all of it is new.
    """
    import glob
    offsets = dict(offsets or {})
    root = offsets.pop(_ROOT_KEY, None)
    files = set(offsets)
    if root:
        files.update(_worker_logs(root))
    else:                                    # older snapshot without a root: fall back
        for d in {os.path.dirname(os.path.dirname(f)) for f in offsets}:
            files.update(_worker_logs(os.path.dirname(d), _pattern=(d, "logs", "worker-*.out")))
    files = _one_path_per_file(files)
    chunks = []
    for f in sorted(files):
        try:
            with open(f, errors="ignore") as fh:
                fh.seek(offsets.get(f, 0))
                chunks.append(fh.read())
        except OSError:
            pass
    return ANSI.sub("", "".join(chunks))


def per_instance_from_log(logpath=None, limit=40, offsets=None, cfg=None):
    """Per-INSTANCE outcomes scraped from the reward function's own prints.

    verl's validation reports means per data_source, which cannot say which kernel task is
    stuck — and that is precisely what a per-instance scaffold has to target. The reward echoes
    task_name/level per candidate, so aggregating those lines gives, for each instance, how
    often it was correct and what speedup it reached.

    Reads the Ray worker logs (see worker_log_offsets), optionally only the part appended since
    `offsets`. Rows with task_name None are skipped: the from-scratch half of the dataset carries
    no task_name, so those candidates are not instance-addressable and counting them under a
    single "None" key would invent an instance that does not exist.

    Returns (per_instance, failures) where failures is the worst instances first: never correct
    ranks above sometimes-correct, and among those the ones tried most often rank first, since
    an instance that failed 6 times is better evidence than one that failed once.
    """
    if offsets is not None or cfg is not None:
        txt = _read_since(offsets if offsets is not None else worker_log_offsets(cfg))
    else:
        try:
            txt = ANSI.sub("", open(logpath, errors="ignore").read())
        except Exception:
            return {}, []
    agg = {}
    for m in REWARD_LINE.finditer(txt):
        correct, speedup, _ds, task, level, _category, _reward = m.groups()
        if task in ("None", ""):
            continue
        a = agg.setdefault(task, {"n": 0, "n_correct": 0, "speedups": [], "level": level})
        a["n"] += 1
        if correct == "1":
            a["n_correct"] += 1
            try:
                a["speedups"].append(float(speedup))
            except ValueError:
                pass
    out = {}
    for t, a in agg.items():
        sp = a["speedups"]
        out[t] = {"n": a["n"], "correct_rate": round(a["n_correct"] / a["n"], 3),
                  "mean_speedup": round(sum(sp) / len(sp), 3) if sp else 0.0,
                  "level": a["level"]}
    failures = sorted((t for t, v in out.items() if v["correct_rate"] < 1.0),
                      key=lambda t: (out[t]["correct_rate"], -out[t]["n"]))[:limit]
    return out, [{"instance": t, **out[t]} for t in failures]


def cat_of_level(level, category=None):
    """Scaffold category for one reward line. Mirrors splice.level_of: an explicit `category`
    wins, and only when absent is the label derived from `level`.

    Both are needed because the two datasets define categories differently — the CUDA set by
    level, the Triton set by operator family with level==0 on every row. Deriving from level
    there yields "improve_l0", which belongs to no domain, so every candidate would be dropped
    from the per-category aggregate and the Teacher would see an empty breakdown."""
    if isinstance(category, str) and category.strip() and category.strip() not in ("None", "nan"):
        return category.strip()
    s = str(level).strip()
    if s in ("None", "", "nan", "NaN"):
        return "scratch"
    try:
        return f"improve_l{int(float(s))}"
    except (TypeError, ValueError):
        return None


def signals_from_training(offsets, fired, rollout_n=None, scaffold=None, domain=None):
    """Everything the Teacher needs about the scaffold's in-context effect, from the rollouts
    TRAINING ALREADY PRODUCED. Costs nothing: no extra generation, no extra GPU process.

    This replaces a dedicated two-pass measurement (the same rows scored bare, then scored with
    the scaffold forced on). That pass was 2 of the 7 full Ray+vLLM startups every cycle, and
    every startup is a GPU context create/destroy — the operation that wedged this node on
    2026-08-04 and OOMed the ALFWorld arm on 2026-07-27. It was also redundant, because training
    is ALREADY a randomised experiment: splice.build injects per GROUP with probability p_task,
    the assignment is a hash of (seed, task) and therefore independent of task difficulty, and
    every rollout is scored. Injected and non-injected groups in the same window are a bare-vs-
    scaffolded contrast on the same policy trajectory, for free.

    `fired` is the per-task map splice.build RECORDED while writing this cycle's parquet — the
    ground truth, not a replay. An earlier version recomputed it from (seed, scaffold) and was
    wrong: injection needs the coin AND the scope to hold text, and replaying only the coin
    mislabelled 247 of 600 rows in a realistic case (p on every category, text on one), every one
    of them an un-injected row counted as injected.

    What this does NOT measure, and the dedicated pass did not either: whether the text TRANSFERS.
    Both measure in-context lift on rows that can read the text. Only valid_seen, measured bare,
    speaks to the objective.

    Caveats worth keeping in view:
      - the policy updates during the window, so the two arms are not the same weights; the
        randomised assignment makes that noise rather than bias, but it is noise.
      - injected groups produce gradients that shape the policy generating the later bare groups,
        so the arms interfere. The interference shrinks the measured gap, i.e. it is conservative.
      - at p=0.2 the injected arm has a quarter of the bare arm's rows; the difference's standard
        error is dominated by the smaller one.
    """
    import collections
    txt = _read_since(offsets)
    per_task = collections.defaultdict(list)
    fails_of = collections.defaultdict(list)   # task -> [(kind, err)] from failed rollouts
    cat_of = {}
    for m in REWARD_LINE.finditer(txt):
        correct, speedup, _ds, task, level, category, _reward = m.groups()
        if task in ("None", ""):
            continue
        cat = cat_of_level(level, category)
        if cat is None:
            continue
        try:
            r = 0.0 if correct != "1" else round(float(speedup), 6)
        except ValueError:
            continue
        per_task[task].append((int(correct == "1"), r))
        if correct != "1":
            kind, err = _fail_of(txt, m)
            if kind:
                fails_of[task].append((kind, err))
        cat_of[task] = cat

    sizes = collections.Counter(len(v) for v in per_task.values())
    want = rollout_n or (sizes.most_common(1)[0][0] if sizes else 0)

    gap = {}
    zero_grad = {}
    for task, results in per_task.items():
        if len(results) != want:            # drop the val_kwargs.n=1 rows sharing this window
            continue
        cat = cat_of[task]
        if task not in fired:          # not part of this cycle's parquet -> not our experiment
            continue
        arm = "injected" if fired[task] else "bare"
        g = gap.setdefault(cat, {"bare": [0, 0], "injected": [0, 0],
                                 "sp_bare": [], "sp_injected": []})
        g[arm][0] += sum(c for c, _ in results)
        g[arm][1] += len(results)
        g["sp_" + arm].extend(r for c, r in results if c == 1)
        z = zero_grad.setdefault(cat, {"zero_gradient": 0, "all_fail": 0, "all_succeed": 0,
                                       "total": 0})
        z["total"] += 1
        if len({r for _, r in results}) == 1:
            z["zero_gradient"] += 1
            z["all_fail" if all(c == 0 for c, _ in results) else "all_succeed"] += 1

    # `failures` for the Teacher: the worst instances UNDER THE BARE PROMPT. Restricted to
    # non-injected groups so that description stays true — the training window holds both arms,
    # and calling a scaffolded failure "bare" would be the same class of error as the signal names
    # this module has already had to correct twice today.
    failures = []
    for t, v in per_task.items():
        if len(v) != want or fired.get(t, False) or sum(c for c, _ in v) >= len(v):
            continue
        row = {"instance": t, "correct_rate": round(sum(c for c, _ in v) / len(v), 4),
               "n": len(v), "category": cat_of[t]}
        # The evidence half, mirroring what contrastive_traces gives the multi-turn domains:
        # not the trajectory (one attempt is one answer) but WHY the attempts failed — the
        # runner-reported kind per failed rollout, plus one verbatim error message. The shortest
        # non-empty message is the sample: compiler and mismatch errors front-load the cause,
        # and the long ones are stack traces whose useful line is the one the short ones lead with.
        kinds = collections.Counter(k for k, _ in fails_of.get(t, []))
        if kinds:
            row["fail_kinds"] = dict(kinds.most_common())
            errs = [e for _, e in fails_of.get(t, []) if e]
            if errs:
                row["sample_error"] = min(errs, key=len)[:200]
        sp = [r for c, r in v if c == 1]
        if sp:
            row["speedup_when_correct"] = round(sum(sp) / len(sp), 3)
        failures.append(row)
    failures.sort(key=lambda x: (x["correct_rate"], x["instance"]))

    out_gap = {}
    for cat, g in gap.items():
        b, i = g["bare"], g["injected"]
        row = {
            "bare": round(b[0] / b[1], 4) if b[1] else None, "n_bare": b[1],
            "injected": round(i[0] / i[1], 4) if i[1] else None, "n_injected": i[1],
            "gap": (round(i[0] / i[1] - b[0] / b[1], 4) if (b[1] and i[1]) else None),
            "unit": "mean per-rollout success, this cycle's training rollouts, "
                    "split by whether p_task fired for that group",
        }
        # Correctness alone hides this domain's other failure mode: kernels that pass the check
        # but are no faster than the PyTorch they replace. Median over CORRECT rollouts only,
        # withheld below 3 samples — at n=1 or 2 a single lucky kernel IS the median.
        for arm_name in ("bare", "injected"):
            sp = sorted(g["sp_" + arm_name])
            row[f"speedup_median_{arm_name}"] = (
                round(sp[len(sp) // 2], 3) if len(sp) >= 3 else None)
        # Why an arm is empty, because the three causes call for different actions and a bare
        # "gap: null" does not separate them.
        if not i[1] and scaffold is not None:
            from . import splice as SP
            has_text = bool(SP.render_block(scaffold, cat).strip())
            p = float((scaffold.get("p_task") or {}).get(cat,
                                                         scaffold.get("default_p", 0.0)) or 0.0)
            if not has_text:
                row["no_injection_reason"] = ("neither 'general' nor this category holds text, so "
                                              "nothing is spliced here whatever p_task says")
            elif p <= 0:
                row["no_injection_reason"] = f"p_task['{cat}'] is {p}, so no group was selected"
            else:
                row["no_injection_reason"] = (f"p_task['{cat}']={p} but no group happened to fire "
                                              f"among the {b[1] // max(1, want)} trained this cycle")
        elif not b[1]:
            row["no_bare_reason"] = ("every group of this category was injected, leaving no "
                                     "comparison arm")
        out_gap[cat] = row
    for cat, z in zero_grad.items():
        z["rollout_n"] = want
        z["unit"] = "GRPO groups with NO reward variance (no gradient)"
    # The questions this cycle actually trained on. The gate re-measures on exactly these, so the
    # three arms are compared on the same problems at the same policy rather than on a fresh
    # random draw from the full training file.
    seen = {t: cat_of[t] for t, v in per_task.items() if len(v) == want}
    return {"per_task_gap": out_gap, "zero_gradient_groups": zero_grad,
            "failures": failures, "seen_tasks": seen}


# NOT IMPLEMENTED: cross-cycle accumulation of per_task_gap with an exponential decay.
#
# A lambda was derived for it on 2026-08-04 — the local-level model theta_t = theta_{t-1} + w,
# y_t = theta_t + v, with r measured from three eval draws of ONE checkpoint (0.001915) and q
# from the cycle-to-cycle differences minus 2r/3 (0.000368), giving a steady-state Kalman gain
# of 0.524 and a retained weight of 0.476, rounded to 0.8 at the user's instruction.
#
# The function that applied it was written and never called: nothing accumulated the history it
# takes, no field carried its output, and the prompt never described it. It was removed on
# 2026-08-06 rather than left in place, because code that implies a mechanism the run does not
# have is worse than its absence — the next reader has no way to tell.
#
# per_task_gap is therefore THIS CYCLE'S measurement only, which is what the prompt says.
# Reinstating the decay needs three things together: the loop keeping per-cycle raw counts in
# state, the observation carrying the accumulated view, and the prompt saying that the weights
# are not episode counts.


def zero_gradient_groups_from_log(offsets, rollout_n=None):
    """Groups that contribute NO gradient, per category, from this window's training rollouts.

    A GRPO group is the `rollout_n` completions of ONE prompt, and the update is group-relative:
    if every completion in a group scores the same, every advantage in it is zero and that
    instance contributes NO gradient. That is the quantity the Teacher is told about, and it can
    only be computed where groups exist — i.e. in TRAINING (rollout_n=6), not in a measurement
    pass, which runs val_kwargs.n=1 and therefore has one completion per task by construction.

    The signals pass used to supply this number and could not have been right: with n=1 its
    "group" is a single candidate, so its all-fail count is just its failure count. At the measured
    success rates that overstated the zero-gradient share by 2.4x to 3.7x — on the signal the
    prompt leans on hardest.

    The condition is NO VARIANCE IN THE REWARD, not "everything failed". GRPO's advantage is
    r_i - mean(group): if every rollout in a group scores the SAME, every advantage is zero and
    that instance is silent — whether they all failed or all succeeded.

    Judging on failure alone is right here only by accident. In this domain a correct kernel's
    reward is driven by its measured speedup, which is continuous, so six correct rollouts have
    six different rewards and the group still teaches something: over 638 real groups, 318 had no
    reward variance and all 318 were all-fail, while the 6 all-correct groups had 6 distinct
    speedups each. Where the reward is BINARY the two come apart completely — the converged
    ALFWorld run reached a training success rate of 1.000 at step 197, so its all-fail count was
    0 while very nearly every group was in fact silent. A signal that reads zero exactly when the
    problem is total is worse than no signal.

    The breakdown ships with the number because the two causes have opposite remedies: all-fail
    means the task is out of reach and text may buy a foothold; all-succeed means it is too easy
    and text buys nothing, so the honest move is to withdraw or to train on something harder.

    Returns {category: {"zero_gradient": int, "all_fail": int, "all_succeed": int, "total": int,
                        "rollout_n_median": int, "unit": ...}}.
    """
    import collections
    txt = _read_since(offsets)
    per_task = collections.defaultdict(list)
    cat_of = {}
    for m in REWARD_LINE.finditer(txt):
        correct, speedup, _ds, task, level, category, _reward = m.groups()
        if task in ("None", ""):
            continue
        cat = cat_of_level(level, category)
        if cat is None:
            continue
        # The reward as logged: 0 when incorrect, otherwise monotone in the measured speedup.
        # Rounded because two rollouts differing in the 12th decimal are not "the same reward"
        # in any sense that matters, and float equality would call them different.
        try:
            r = 0.0 if correct != "1" else round(float(speedup), 6)
        except ValueError:
            continue
        per_task[task].append((int(correct == "1"), r))
        cat_of[task] = cat
    # Only genuine GRPO groups. verl runs its validation pass (val_kwargs.n=1) inside the SAME
    # training subprocess, so this window also holds one-completion "groups" — and a group of one
    # has no variance by definition, so counting them reported 0.800 zero-gradient where the truth
    # over real groups was 0.498. Filter to the configured rollout_n; absent that, to the modal
    # size, which is the training group in any window dominated by training.
    sizes_seen = collections.Counter(len(v) for v in per_task.values())
    want = rollout_n or (sizes_seen.most_common(1)[0][0] if sizes_seen else 0)
    skipped = sum(n for sz, n in sizes_seen.items() if sz != want)
    out = {}
    for task, results in per_task.items():
        if len(results) != want:
            continue
        a = out.setdefault(cat_of[task], {"zero_gradient": 0, "all_fail": 0, "all_succeed": 0,
                                          "total": 0, "sizes": [], "n_skipped_wrong_size": skipped})
        a["total"] += 1
        a["sizes"].append(len(results))
        if len({r for _, r in results}) == 1:
            a["zero_gradient"] += 1
            if all(c == 0 for c, _ in results):
                a["all_fail"] += 1
            else:
                a["all_succeed"] += 1
    for cat, a in out.items():
        sizes = sorted(a.pop("sizes"))
        a["rollout_n_median"] = sizes[len(sizes) // 2] if sizes else 0
        a["unit"] = ("GRPO groups (one prompt's rollout_n completions) with NO reward variance, "
                     "i.e. contributing no gradient")
    return out


def per_category_from_log(offsets):
    """Per-CATEGORY outcomes for one measurement pass, from the reward's own prints.

    Why not verl's validation metrics: verl aggregates by `data_source`, and this dataset has
    only two of them — `CudaForge` (all 200 scratch rows) and `CudaForgeImprovement` (all 200
    improvement rows, l1+l2+l3 merged). The scaffold's scopes are scratch/improve_l1/l2/l3, so
    a per-data_source breakdown shares no key with anything the Teacher can edit. It spent ten
    cycles saying "no reliable category breakdown"; that was literally true.

    The reward prints `level:` per candidate, which is exactly the field the categories are
    defined on, so aggregating those lines gives the breakdown verl cannot.

    Reports `correct_rate` rather than the shaped reward on purpose. The shaped reward is
    (clip(speedup)+0.3)*(1+lambda*(r-0.5)) with speedup clipped at 5.0, so on a small sample one
    kernel that happens to hit 5x moves the mean by ~0.6 — at n=8 that produced a measured
    "gap" of -0.72 between two passes whose prompts were byte-identical. Correctness is Bernoulli
    and behaves. Speedup is still reported, separately, so the correct-but-not-faster failure
    mode stays visible.
    """
    txt = _read_since(offsets)
    agg = {}
    rowed = {}  # row-tagged task -> [(correct, speedup)]: one PARQUET ROW, however often scored
    row_cat = {}

    def _tally(cat, corrects, speedups):
        a = agg.setdefault(cat, {"n": 0, "n_correct": 0, "speedups": []})
        a["n"] += 1
        a["n_correct"] += sum(corrects) / len(corrects)
        if speedups:
            a["speedups"].append(sum(speedups) / len(speedups))

    for m in REWARD_LINE.finditer(txt):
        correct, speedup, _ds, task, level, category, _reward = m.groups()
        cat = cat_of_level(level, category)
        if cat is None:
            continue
        if task and _ROW_MARK in task:
            # Collect, don't count: verl pads validation batches by repeating rows
            # (ray_trainer.py:596) and the repeats are scored too. Same id = same row; its
            # observations average into ONE sample below, so a padded repeat cannot double-weight
            # its row, and disagreeing observations (greedy re-bench timing) split the difference.
            rowed.setdefault(task, []).append((correct, speedup))
            row_cat[task] = cat
            continue
        sp = []
        if correct == "1":
            try:
                sp.append(float(speedup))
            except ValueError:
                pass
        _tally(cat, [int(correct == "1")], sp)

    for task, obs in rowed.items():
        sps = []
        for c, s in obs:
            if c == "1":
                try:
                    sps.append(float(s))
                except ValueError:
                    pass
        _tally(row_cat[task], [int(c == "1") for c, _ in obs], sps)

    out = {}
    for cat, a in agg.items():
        sp = sorted(a["speedups"])
        # n_correct is fractional only when a padding phantom's observations disagreed;
        # keep the common case printing as the integer it is.
        nc = round(a["n_correct"], 3)
        out[cat] = {"n": a["n"], "n_correct": int(nc) if nc == int(nc) else nc,
                    "correct_rate": round(a["n_correct"] / a["n"], 4) if a["n"] else 0.0,
                    "speedup_median": round(sp[len(sp) // 2], 3) if sp else None,
                    "n_faster_than_torch": sum(1 for s in sp if s > 1.0)}
    return out


def _measure_pass(checkpoint, parquet, tag, cfg, seed):
    """One frozen-policy validation pass over `parquet`, returning (mean_reward, per_source).

    Reuses verl's own val_only path rather than a bespoke rollout loop: the reward it applies is
    then bit-identical to the one training optimises, including the nvcc compile, the correctness
    gate and the rubric call. A separate implementation would be a second thing to keep in sync.
    """
    step = int(str(checkpoint).rstrip("/").split("global_step_")[-1])
    log = os.path.join(cfg["log_dir"], f"{cfg['exp']}_measure_s{step}_{tag}.log")
    env = base_env(cfg)
    env["VLLM_SEED"] = str(seed)
    cmd = _train_cmd(cfg, parquet, step, val_before=True, test_freq=1) + " trainer.val_only=True"
    cmd = cmd.replace(f"data.val_files={cfg['val_file']}", f"data.val_files={parquet}")
    # Feed the pass in waves. verl defaults data.val_batch_size to null, which means "the whole
    # file in one batch" (ray_trainer.py:398-400) — so the A/B handed 540 rows to the reward at
    # once, and every completed generation forked a kernel_runner that creates a CUDA context.
    # On 2026-08-10 that put 85 of them on one GPU inside 49 seconds and deadlocked the driver's
    # global rwsem; the node needed a reboot.
    #
    # The validation loop iterates its dataloader batch by batch (ray_trainer.py:550), generating
    # and scoring each before the next, so a bounded batch size bounds the peak directly — while
    # still loading the model exactly once, which chunking the file into separate passes would
    # not (each pass is a fresh verl process: an 8B load plus vLLM init).
    #
    # This is the structural bound. The node-wide slot cap stays as the backstop, but peak
    # concurrency should be set by how much work is in flight, not by how high the cap happens
    # to be set.
    cmd += f" data.val_batch_size={int(cfg.get('measure_batch_rows', _MEASURE_BATCH_ROWS))}"
    _run(cmd, log, env)
    return parse_val(log)


def _subset_tasks(src, dst, task_names, repeats=1):
    """Exactly the named problems, each repeated `repeats` times.

    Used by the gate so that all three arms are scored on the problems the cycle just trained on,
    rather than on an independent sample from the training file. Same problems across arms makes
    the comparison paired; problem difficulty cancels instead of contributing variance.
    """
    import pandas as pd
    from . import splice as SP

    df = pd.read_parquet(src)
    keys = [SP.task_key(df["extra_info"].iloc[i], i) for i in range(len(df))]
    df = df.assign(_k=keys)
    want = set(task_names)
    sub = df[df["_k"].isin(want)]
    if sub.empty:
        raise StepFailed(f"_subset_tasks: none of the {len(want)} named problems exist in {src}")
    out = pd.concat([sub] * max(1, int(repeats))).drop(columns=["_k"])
    out.to_parquet(dst, index=False)
    return out


def _count_in_categories(path, categories):
    import pandas as pd
    from . import splice as SP
    df = pd.read_parquet(path)
    want = set(categories)
    return sum(1 for i in range(len(df))
               if SP.level_of(df["extra_info"].iloc[i],
                              df["data_source"].iloc[i] if "data_source" in df.columns else None)
               in want)


def _subset(src, dst, n_per_cat, seed, only=None, repeats=1):
    """`n_per_cat=None` means every row of the category; an integer means that many, topping up by
    repetition when the category holds fewer. Passing a huge integer to mean "all" was a bug: it
    entered the top-up branch and asked pandas to concatenate 33 million copies of 30 rows."""
    """A fixed-size, category-balanced slice of the training set for measurement.

    Balanced because the categories are lopsided (200 scratch vs 41 improve_l3): a uniform sample
    would measure the scaffold's effect on `scratch` and call it the whole picture.

    `only` restricts the slice to named categories. The A/B uses it: ab_gate aggregates over the
    TOUCHED categories alone, so rows of a category the proposal did not edit are byte-identical
    across all three arms and cannot move the verdict — they were five sixths of every A/B's cost
    and contributed nothing. Spending that budget on the touched category instead is what makes a
    usable sample size affordable.
    """
    import pandas as pd
    from . import splice as SP

    df = pd.read_parquet(src)
    cats = [SP.level_of(df["extra_info"].iloc[i],
                        df["data_source"].iloc[i] if "data_source" in df.columns else None)
            for i in range(len(df))]
    df = df.assign(_cat=cats)
    if only:
        keep = set(only)
        df = df[df["_cat"].isin(keep)]
        if df.empty:
            raise StepFailed(f"_subset: none of the requested categories {sorted(keep)} "
                             f"exist in {src}")
    parts = []
    for _, g in df.groupby("_cat"):
        if n_per_cat is None:
            # Take the whole category. Distinct from a very large n_per_cat, which would fall into
            # the top-up branch below and try to repeat 30 rows into a billion.
            parts.append(g)
            continue
        if n_per_cat <= len(g):
            parts.append(g.sample(n=n_per_cat, random_state=seed))
            continue
        # Asking for more rows than the category HAS (Triton ships 100 per category, and the A/B
        # budget wants 300 when a proposal touches one). Repeat instead of truncating, because
        # the noise this sample size exists to average down is generation nondeterminism, not
        # task sampling: at step 40 two passes over byte-identical prompts produced different
        # code for 112 of 112 recoverable tasks. A repeated row is a genuine second draw of that
        # process, and lands at a different batch position — which is where the variation comes
        # from. It is NOT a second draw of task difficulty; that component stays at len(g). It
        # also does not need to be, since all three arms share the same row multiset and task
        # difficulty cancels in the paired difference.
        reps = -(-n_per_cat // len(g))
        parts.append(pd.concat([g] * reps).head(n_per_cat))
    out = pd.concat(parts).drop(columns=["_cat"])
    if repeats > 1:
        out = pd.concat([out] * int(repeats))
    out.to_parquet(dst, index=False)
    return out


def signals_adapter(checkpoint, scaffold, cfg, seed):
    """Train-side signals: the same games scored bare and with the live scaffold.

    Paired on purpose — same rows, same seed, only the scaffold differs — so the gap is the
    scaffold's effect on THIS policy rather than a difference between two samples of tasks.
    """
    from . import splice as SP

    step = int(str(checkpoint).rstrip("/").split("global_step_")[-1])
    base = os.path.join(cfg["work"], f"sig_s{step}_base.parquet")
    _subset(cfg["train_file"], base, cfg["n_per_task"], seed)
    bare_pq = os.path.join(cfg["work"], f"sig_s{step}_bare.parquet")
    inj_pq = os.path.join(cfg["work"], f"sig_s{step}_inj.parquet")
    SP.build(base, bare_pq, scaffold, seed=seed, mode="none")
    SP.build(base, inj_pq, scaffold, seed=seed, mode="force")

    # Offsets are snapshotted around EACH pass, not once around both. Reading a single window
    # spanning both passes blended the bare and injected candidates into one aggregate, so the
    # per-instance "failures" list described neither condition.
    off_bare = worker_log_offsets(cfg)
    bare_avg, bare_src = _measure_pass(checkpoint, bare_pq, "bare", cfg, seed)
    off_inj = worker_log_offsets(cfg)
    inj_avg, inj_src = _measure_pass(checkpoint, inj_pq, "inj", cfg, seed)

    bare_cat = per_category_from_log(off_bare)
    inj_cat = per_category_from_log(off_inj)

    # Per-instance view comes from the BARE pass alone: it describes the condition the policy
    # faces at evaluation, which is what an instance-level intervention has to move.
    per_instance, failures = per_instance_from_log(offsets=off_bare)

    per_task_gap = {}
    for cat in sorted(set(bare_cat) | set(inj_cat)):
        b = bare_cat.get(cat, {})
        i = inj_cat.get(cat, {})
        br, ir = b.get("correct_rate", 0.0), i.get("correct_rate", 0.0)
        per_task_gap[cat] = {
            "bare": br, "injected": ir, "gap": round(ir - br, 4),
            "n": b.get("n", 0),
            "metric": "fraction of candidates that compile, run and match the reference",
            "bare_correct": f"{b.get('n_correct', 0)}/{b.get('n', 0)}",
            "injected_correct": f"{i.get('n_correct', 0)}/{i.get('n', 0)}",
            # Correctness alone hides this domain's main failure mode: kernels that are correct
            # but no faster than the PyTorch reference they are scored against.
            "bare_speedup_median": b.get("speedup_median"),
            "injected_speedup_median": i.get("speedup_median"),
            "bare_n_faster_than_torch": b.get("n_faster_than_torch", 0),
        }

    # A reward of exactly 0 means wrong, uncompilable or timed out — indistinguishable in the
    # score. Within a GRPO group every rollout scoring 0 gives the update no spread and hence no
    # gradient, so the per-category zero-rate is this domain's read on where learning signal is
    # missing. Measured on the bare pass, the condition the policy will face at evaluation.
    # NOT a group statistic, and no longer named as though it were. This pass runs
    # val_kwargs.n=1, so there is exactly one candidate per task and "all of them failed" means
    # "it failed". Reporting it as a group statistic told the Teacher that 79-84% of its training
    # groups had no gradient when the true share was 21-35% — a 2.4x to 3.7x overstatement of the
    # single concept the prompt leans on hardest. The real group statistic now comes from the
    # training rollouts (zero_gradient_groups_from_log), where groups actually exist.
    zero_reward = {}
    for cat, b in bare_cat.items():
        n = b.get("n", 0)
        if n:
            zero_reward[cat] = {"n_zero": n - b.get("n_correct", 0), "n": n,
                                "rate": round((n - b.get("n_correct", 0)) / n, 4),
                                "unit": "individual bare candidates scoring exactly 0"}

    return {"per_task_gap": per_task_gap, "zero_reward_rate": zero_reward,
            "failures": failures, "successes": [],
            "per_instance": per_instance,
            # verl's own per-data_source means, kept for the record. NOT used for per_task_gap:
            # this dataset has two data_sources (CudaForge = all scratch, CudaForgeImprovement =
            # l1+l2+l3 merged), which share no key with the scaffold's scopes.
            "train_curve": {"bare_mean": bare_avg, "injected_mean": inj_avg,
                            "bare_by_data_source": bare_src, "injected_by_data_source": inj_src}}


# Separator folding the A/B arm into the category label so all three arms can share one pass.
# Chosen because no category or task_name contains it.
# Optional tail the reward appends AFTER the fields REWARD_LINE captures: why a candidate
# scored 0 ("fail:compile_error, err:...") . Parsed separately so REWARD_LINE and everything
# holding offsets into its groups stay byte-compatible with logs from before the field existed.
_FAIL_TAIL = re.compile(r"fail:([^,\s]+)(?:,\s*err:([^\n]*))?")


def _fail_of(txt, match):
    """(kind, err) from the reward line `match` sits on, or (None, None)."""
    end = txt.find("\n", match.end())
    tail = txt[match.end():end if end != -1 else len(txt)]
    ft = _FAIL_TAIL.search(tail)
    if not ft:
        return None, None
    return ft.group(1), (ft.group(2) or "").strip() or None


_ARM_SEP = "@@"

# Marks a task_name as belonging to ONE PARQUET ROW of a measurement pass ("name#r17@@arm").
# ray_trainer.py:596 pads every validation batch to a multiple of rollout.agent.num_workers by
# repeating rows; the repeats are generated AND scored (their reward lines hit the worker logs)
# and only their tensors are dropped by unpad. With 8 workers and 36-row waves that was 4 phantom
# lines per wave — 60 of 540 in the first smoke A/B, n reported as 199/200/201 per 180-row arm.
# A row id makes the phantom detectable at parse time: the same id seen twice is one row scored
# twice, whereas legitimate repeat copies (ab_repeats_max) are distinct rows with distinct ids.
_ROW_MARK = "#r"


def measure_ab_adapter(checkpoint, current, candidate, tasks, cfg, seed):
    """Three-way A/B — bare vs the live scaffold vs the proposal — in ONE generation pass.

    Scored per CATEGORY from the reward's own prints, for the same reason signals_adapter is:
    verl aggregates validation by `data_source`, and neither dataset's data_sources line up with
    the scaffold's categories — the Triton set has exactly one. An earlier version asked
    `per.get(task, overall_avg)` and, finding no matching key, silently handed ab_gate the OVERALL
    mean for every touched category, so a proposal editing one of six categories was judged on a
    mean five sixths of which could not have changed.

    The arms share ONE pass. Measured at step 40 with an EMPTY scaffold, where `bare` and
    `current` are the same condition run twice on byte-identical prompts (md5 a2bbbaa613dc for
    both files):

        conv 1/24 vs 6/24 · loss 10/24 vs 4/24 · reduce 4/24 vs 1/24 — 32 of 144 tasks flipped,
        and of the 112 tasks whose generated code could be recovered from both passes, ZERO
        produced the same code.

    Two separate things are going on there, and it is worth not confusing them.

    The identical CODE never reappearing is vLLM's batching. Generation is greedy — agent_loop.py
    overrides temperature to val_kwargs.temperature=0, metrics confirm mean@1, and the benchmark
    timed out zero times — so what is left is that identical prompts scheduled into differently
    composed batches reduce in a different order, the argmax flips on a near-tie, and one divergent
    token cascades through a 7-9k-token response. Inherent to batched inference; a rerun does not
    fix it. Merging the arms removes the part of this that IS controllable: one process instead of
    three, one model load, one GPU state, one batch composition.

    The alarming per-category NUMBERS are something much more boring: n=24. Those two passes agree
    to 4.2 points in aggregate (30/144 vs 24/144), the six categories' swings cancel in direction,
    and a chi-square test of homogeneity gives 10.58 on 6 df against a 12.59 critical value — the
    spread is inside what plain binomial sampling predicts. '1/24 vs 6/24' is a ratio read at a low
    base rate; in absolute terms it is 5 tasks, or 2.2 sigma. Different programs mostly meet the
    same fate, which is why total generation nondeterminism (112/112) moves only 32/144 verdicts.

    So the fix that matters is not the merge, it is the sample size — hence the row budget below.
    """
    from . import splice as SP
    import pandas as pd

    step = int(str(checkpoint).rstrip("/").split("global_step_")[-1])
    base = os.path.join(cfg["work"], f"ab_s{step}_base.parquet")

    # Measured on the HELD-OUT file, restricted to the touched categories. All three arms run on
    # the same problems at the same checkpoint: paired comparison, one policy, no problem the
    # model has just been updated on.
    #
    # Two alternatives were rejected. Sampling fresh rows from the training file scores problems
    # the policy was trained on during this very cycle, and the `current` arm's problems were
    # trained on WITH that text in the prompt, which is the condition being measured. Taking
    # `bare` and `current` from the training rollouts costs nothing but spreads them over the ten
    # policies of the cycle while `candidate` sits at the last one.
    #
    # Consequence to keep in view: the gate now selects on the same file valid_seen reports, so
    # valid_seen becomes an optimistic estimate of standalone performance. This dataset ships no
    # further held-out split.
    #
    # Copies per problem rather than val_kwargs.n, because vLLM returns n identical sequences for
    # one greedy request — separate rows place the copies at different batch positions, which is
    # where the run-to-run variation comes from.
    touched = sorted(set(tasks or cfg["domain"].categories))
    n_q = _count_in_categories(cfg["val_file"], touched)
    budget = int(cfg.get("ab_row_budget", os.environ.get("ARM_AB_ROWS", "540")))
    reps_max = int(cfg.get("ab_repeats_max", os.environ.get("ARM_AB_REPEATS_MAX", "6")))
    reps = max(1, min(reps_max, budget // max(1, 3 * n_q)))
    _subset(cfg["val_file"], base, None, seed, only=touched, repeats=reps)
    n_rows = n_q * reps
    cfg.get("log", lambda *a: None)(
        f"[ab] held-out: {n_q} problems in {touched} x {reps} copies x 3 arms "
        f"= {3 * n_rows} generations in one pass")

    # Splice each arm separately (splice reads extra_info["category"], so the label must still be
    # the real one at this point), then concatenate into ONE file with the arm folded into the
    # category so the single pass can be taken apart afterwards.
    frames = []
    for arm, scaf, mode in (("bare", current, "none"),
                            ("current", current, "force"),
                            ("candidate", candidate, "force")):
        pq = os.path.join(cfg["work"], f"ab_s{step}_{arm}.parquet")
        SP.build(base, pq, scaf, seed=seed, mode=mode)
        df = pd.read_parquet(pq)
        rows = []
        for x in df.extra_info:
            e = dict(x) if isinstance(x, dict) else json.loads(x)
            cat = e.get("category")
            if not cat:
                lv = e.get("level")
                cat = "scratch" if lv is None else f"improve_l{int(float(lv))}"
            e["category"] = f"{cat}{_ARM_SEP}{arm}"
            # task_name is what per_instance_from_log aggregates on; keep the arms apart there too
            # so a task appearing three times is not read as one instance tried three times.
            # The row id (see _ROW_MARK) additionally keeps verl's batch padding apart from
            # ab_repeats_max copies: a padded repeat reprints an EXISTING id, a copy gets its own.
            # Rows without a task_name can't be tagged and keep legacy line counting — the Triton
            # set names every row; only the CudaForge scratch half would be affected.
            if e.get("task_name"):
                e["task_name"] = f"{e['task_name']}{_ROW_MARK}{len(rows)}{_ARM_SEP}{arm}"
            rows.append(e)
        df["extra_info"] = rows
        frames.append(df)

    # INTERLEAVE the arms. Concatenating them in order puts every bare row before every current
    # row, and rows are served in file order — so each arm's rows would systematically share
    # batches with rows of the same arm. That is not a fair comparison, because batch composition
    # CHANGES THE OUTPUT: measured against this repo's own rubric-judge server (vLLM 0.11.0, same
    # host, temperature=0, identical prompts), 16 concurrent copies of one request produced 5
    # distinct completions and 4 concurrent copies produced 2, while five STRICTLY SERIAL repeats
    # produced 5 identical ones. The nondeterminism is entirely batch composition — floating-point
    # reduction order depends on who shares the batch and where — so blocking by arm turns it from
    # noise into a per-arm systematic offset. Shuffling makes each arm's batch neighbours a random
    # mix of all three, which is the condition under which the arm difference is unbiased.
    # Seeded, so a rerun is reproducible.
    merged = os.path.join(cfg["work"], f"ab_s{step}_merged.parquet")
    (pd.concat(frames, ignore_index=True)
       .sample(frac=1.0, random_state=seed)
       .reset_index(drop=True)
       .to_parquet(merged, index=False))

    off = worker_log_offsets(cfg)
    _measure_pass(checkpoint, merged, "ab", cfg, seed)
    by_cat = per_category_from_log(off)

    out = {"bare": {}, "current": {}, "candidate": {}}
    for key, v in by_cat.items():
        if _ARM_SEP not in key:
            continue
        cat, arm = key.rsplit(_ARM_SEP, 1)
        if arm in out:
            # ab_gate consumes {task: (rate, n)} and aggregates n-weighted over the touched tasks,
            # so a category with no rows must be ABSENT rather than present with a made-up rate:
            # absent contributes nothing, present at 0.0 counts as a measured failure.
            if not tasks or cat in tasks:
                out[arm][cat] = (v["correct_rate"], v["n"])
    return out


def persist_scaffold(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path
