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
TORCH_CUDA_ARCH_LIST = os.environ.get("ARM_TORCH_CUDA_ARCH_LIST", "9.0")
VENV_BIN = os.path.dirname(VENV_PY)

ANSI = re.compile(r"\x1b\[[0-9;]*m")

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


def existing_ckpt_step(cfg):
    import glob
    best = 0
    for d in glob.glob(os.path.join(cfg["ckpt_root"], "global_step_*")):
        m = re.search(r"global_step_(\d+)$", os.path.basename(d))
        if m and ckpt_is_usable(d):
            best = max(best, int(m.group(1)))
    return best


def _train_cmd(cfg, train_file, to_step, val_before=False, test_freq=999999):
    """The verl invocation. Derived from experiments/Qwen3-32B_KL003_1e-6_Rubric.sh, with the
    model, GPU count and batch sizes moved into cfg so the arm is one config away from a
    different size."""
    return " ".join([
        f"{VENV_PY} -m verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files={cfg['val_file']}",
        f"data.train_batch_size={cfg['train_batch_size']}",
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
        f"trainer.test_freq={test_freq}",
        f"trainer.total_training_steps={to_step}",
        "trainer.resume_mode=auto",
        "trainer.logger='[\"console\"]'",
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
    draws, pers = [], []
    for d in range(val_n):
        elog = os.path.join(cfg["log_dir"], f"{cfg['exp']}_eval_s{step}_d{d}.log")
        env = base_env(cfg)
        env["VLLM_SEED"] = str(d)
        cmd = _train_cmd(cfg, cfg["val_file"], step, val_before=True, test_freq=1)
        _run(cmd + " trainer.val_only=True", elog, env)
        sr, per = parse_val(elog)
        if sr is not None:
            draws.append(sr)
            pers.append(per)
    if not draws:
        raise StepFailed(
            f"eval at step {step} parsed no validation reward from any of {val_n} draws; "
            f"see {cfg['log_dir']}/{cfg['exp']}_eval_s{step}_d*.log")
    avg = round(sum(draws) / len(draws), 4)
    keys = sorted({k for p in pers for k in p})
    per_task = {k: round(sum(p.get(k, 0.0) for p in pers) / len(pers), 4) for k in keys}
    return {"avg": avg, "per_task": per_task, "draws": [round(x, 4) for x in draws],
            "n_draws": len(draws), "n_draws_requested": val_n}


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
            e["task_name"] = f"{e.get('task_name')}{_ARM_SEP}d{d}"
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
    cfg.get("log", lambda *a: None)(
        f"[eval] step {step}: {val_n} draws in ONE pass ({len(draws)} parsed) draws={draws}")
    return {"avg": avg, "per_task": per_task, "draws": draws,
            "n_draws": len(draws), "n_draws_requested": val_n}


# `category:` is optional so logs written before it was echoed still parse.
REWARD_LINE = re.compile(
    r"correctness:\s*(\d+),\s*speedup:\s*([0-9.eE+-]+),\s*data_source:(\S+?),\s*"
    r"task_name:(\S+?),\s*level:([^,\s]+)(?:,\s*category:(\S+))?")


def worker_log_offsets(cfg):
    """Byte offset of every Ray worker stdout right now.

    The reward runs inside Ray workers, and Ray captures their stdout into its own per-worker
    files rather than the trainer's log — so the `correctness: ... task_name: ...` lines never
    reach train.log. Snapshotting offsets before a measurement pass and reading only what is
    appended after it is what separates that pass's candidates from the training rollouts
    interleaved in the same files.
    """
    import glob
    out = {_ROOT_KEY: cfg["ray_tmp"]}
    for f in glob.glob(os.path.join(cfg["ray_tmp"], "ray", "session_*", "logs", "worker-*.out")):
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
        files.update(glob.glob(os.path.join(root, "ray", "session_*", "logs", "worker-*.out")))
    else:                                    # older snapshot without a root: fall back
        for d in {os.path.dirname(os.path.dirname(f)) for f in offsets}:
            files.update(glob.glob(os.path.join(d, "logs", "worker-*.out")))
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
        correct, speedup, _ds, task, level, _category = m.groups()
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
    cat_of = {}
    for m in REWARD_LINE.finditer(txt):
        correct, speedup, _ds, task, level, category = m.groups()
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
        g = gap.setdefault(cat, {"bare": [0, 0], "injected": [0, 0]})
        g[arm][0] += sum(c for c, _ in results)
        g[arm][1] += len(results)
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
    failures = sorted(
        ({"instance": t, "correct_rate": round(sum(c for c, _ in v) / len(v), 4),
          "n": len(v), "category": cat_of[t]}
         for t, v in per_task.items()
         if len(v) == want and not fired.get(t, False)
         and sum(c for c, _ in v) < len(v)),
        key=lambda x: (x["correct_rate"], x["instance"]))

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


# Weight applied to a cycle's per_task_gap when accumulating across cycles.
#
# Within one cycle the observations come from ten consecutive policies, and there the optimal
# weighting is uniform: measured drift is 0.0005 per step against a sampling standard deviation of
# 0.036, so the bias a ten-step lag introduces is 6% of the noise and its square is 260x smaller
# than the variance it would reduce. Across cycles the ratio reverses, and the estimate has to
# forget.
#
# The value follows from the local-level model theta_t = theta_{t-1} + w, y_t = theta_t + v. Both
# variances are measurable here rather than fitted: r comes directly from the three eval draws of
# ONE checkpoint (0.001915), and q from the cycle-to-cycle differences minus 2r/3 (0.000368). The
# steady-state Kalman gain for that ratio is 0.524, so the retained weight is 0.476.
#
# The construction is the one GAE uses: a family of estimators indexed by how far back they reach,
# combined by a single exponential weight that trades bias against variance. Policy staleness here
# occupies the position value-function error occupies there.
# 0.476 is what the measured q and r give. 0.8 is the value in use: it keeps roughly five cycles
# of history instead of two, at the cost of a longer lag behind the current policy. The measured
# drift is small enough that the extra lag stays well under the sampling noise.
GAP_DECAY = float(os.environ.get("AUTOSCAFFOLD_GAP_DECAY", "0.8"))


def accumulate_gap(history, decay=None):
    """Exponentially weighted per-category bare/injected rates over past cycles.

    `history` is oldest-first: [{cat: {"bare_s","bare_n","inj_s","inj_n"}}, ...]. The most recent
    cycle carries weight 1, the one before it `decay`, and so on.
    """
    d = GAP_DECAY if decay is None else decay
    acc = {}
    for age, cyc in enumerate(reversed(history or [])):
        w = d ** age
        for cat, v in (cyc or {}).items():
            a = acc.setdefault(cat, {"bare_s": 0.0, "bare_n": 0.0, "inj_s": 0.0, "inj_n": 0.0})
            for k in a:
                a[k] += w * float(v.get(k, 0))
    out = {}
    for cat, a in acc.items():
        b = a["bare_s"] / a["bare_n"] if a["bare_n"] else None
        i = a["inj_s"] / a["inj_n"] if a["inj_n"] else None
        out[cat] = {
            "bare": None if b is None else round(b, 4),
            "injected": None if i is None else round(i, 4),
            "gap": None if (b is None or i is None) else round(i - b, 4),
            "n_bare_eff": round(a["bare_n"], 1), "n_injected_eff": round(a["inj_n"], 1),
            "decay": d, "n_cycles": len(history or []),
        }
    return out


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
        correct, speedup, _ds, task, level, category = m.groups()
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
    for m in REWARD_LINE.finditer(txt):
        correct, speedup, _ds, _task, level, category = m.groups()
        cat = cat_of_level(level, category)
        if cat is None:
            continue
        a = agg.setdefault(cat, {"n": 0, "n_correct": 0, "speedups": []})
        a["n"] += 1
        if correct == "1":
            a["n_correct"] += 1
            try:
                a["speedups"].append(float(speedup))
            except ValueError:
                pass
    out = {}
    for cat, a in agg.items():
        sp = sorted(a["speedups"])
        out[cat] = {"n": a["n"], "n_correct": a["n_correct"],
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
_ARM_SEP = "@@"


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
            if e.get("task_name"):
                e["task_name"] = f"{e['task_name']}{_ARM_SEP}{arm}"
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
