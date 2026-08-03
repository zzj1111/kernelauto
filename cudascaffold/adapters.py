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

ROOT = "/mnt/data1/zha00175/StitchCUDA"
VENV_PY = ("/dev/shm/verl_env/bin/python"
           if os.path.exists("/dev/shm/verl_env/bin/python")
           else "/mnt/data1/zha00175/miniconda/envs/verl/bin/python")

# Toolchain facts established by measurement, not assumption (see exp_cudaforge/DECISIONS.md):
#   - /usr/bin/nvcc is CUDA 11.5 and cannot target sm_90; H200 needs 12.x.
#   - torch looks for the ninja EXECUTABLE on PATH, so the venv's bin must be on it.
CUDA_BIN = "/usr/local/cuda-12.9/bin"
VENV_BIN = os.path.dirname(VENV_PY)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


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
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["TORCH_CUDA_ARCH_LIST"] = "9.0"
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
    except Exception as e:                      # never let telemetry break a completed training run
        cfg["_last_train_rollouts"] = {}
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
    """Standalone eval on the held-out file, averaged over `val_n` seeds.

    test.parquet is never scaffolded, so this is bare by construction. Each draw is a separate
    verl validation pass; they differ only by sampling temperature, which is what makes the
    spread between them a usable read on measurement noise rather than a formality.
    """
    step = int(str(checkpoint).rstrip("/").split("global_step_")[-1])
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
    out = {}
    for f in glob.glob(os.path.join(cfg["ray_tmp"], "ray", "session_*", "logs", "worker-*.out")):
        try:
            out[f] = os.path.getsize(f)
        except OSError:
            pass
    return out


def _read_since(offsets):
    import glob
    chunks = []
    seen = set(offsets or {})
    pats = {os.path.dirname(os.path.dirname(f)) for f in seen} or set()
    files = set(seen)
    for d in pats:
        files.update(glob.glob(os.path.join(d, "logs", "worker-*.out")))
    for f in files:
        try:
            with open(f, errors="ignore") as fh:
                fh.seek((offsets or {}).get(f, 0))
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


def _subset(src, dst, n_per_cat, seed):
    """A fixed-size, category-balanced slice of the training set for measurement.

    Balanced because the categories are lopsided (200 scratch vs 41 improve_l3): a uniform sample
    would measure the scaffold's effect on `scratch` and call it the whole picture.
    """
    import pandas as pd
    from . import splice as SP

    df = pd.read_parquet(src)
    cats = [SP.level_of(df["extra_info"].iloc[i],
                        df["data_source"].iloc[i] if "data_source" in df.columns else None)
            for i in range(len(df))]
    df = df.assign(_cat=cats)
    parts = [g.sample(n=min(n_per_cat, len(g)), random_state=seed) for _, g in df.groupby("_cat")]
    out = pd.concat(parts).drop(columns=["_cat"])
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
    all_fail = {}
    for cat, b in bare_cat.items():
        n = b.get("n", 0)
        if n:
            all_fail[cat] = {"all_fail": n - b.get("n_correct", 0), "total": n,
                             "unit": "candidates scoring exactly 0 in the bare pass"}

    return {"per_task_gap": per_task_gap, "all_fail_groups": all_fail,
            "failures": failures, "successes": [],
            "per_instance": per_instance,
            # verl's own per-data_source means, kept for the record. NOT used for per_task_gap:
            # this dataset has two data_sources (CudaForge = all scratch, CudaForgeImprovement =
            # l1+l2+l3 merged), which share no key with the scaffold's scopes.
            "train_curve": {"bare_mean": bare_avg, "injected_mean": inj_avg,
                            "bare_by_data_source": bare_src, "injected_by_data_source": inj_src}}


def measure_ab_adapter(checkpoint, current, candidate, tasks, cfg, seed):
    """Paired three-way A/B: bare vs the live scaffold vs the proposal, same rows, same seed.

    Scored per CATEGORY from the reward's own prints, for the same reason signals_adapter is:
    verl aggregates validation by `data_source`, and neither dataset's data_sources line up with
    the scaffold's categories — the Triton set has exactly one. The previous version asked
    `per.get(task, overall_avg)` and, finding no matching key, silently handed ab_gate the OVERALL
    mean for every touched category. A proposal editing one of six categories was then judged on
    a mean where five sixths of the rows could not have changed, using the shaped reward, whose
    speedup term is heavy-tailed enough to have produced a -0.72 "gap" between two byte-identical
    prompt sets. Both the dilution and the metric are fixed here.
    """
    from . import splice as SP

    step = int(str(checkpoint).rstrip("/").split("global_step_")[-1])
    base = os.path.join(cfg["work"], f"ab_s{step}_base.parquet")
    _subset(cfg["train_file"], base, cfg["n_per_task"], seed)
    out = {}
    for tag, scaf, mode in (("bare", current, "none"),
                            ("current", current, "force"),
                            ("candidate", candidate, "force")):
        pq = os.path.join(cfg["work"], f"ab_s{step}_{tag}.parquet")
        SP.build(base, pq, scaf, seed=seed, mode=mode)
        off = worker_log_offsets(cfg)
        _measure_pass(checkpoint, pq, f"ab_{tag}", cfg, seed)
        by_cat = per_category_from_log(off)
        # ab_gate consumes {task: (rate, n)} and aggregates n-weighted over the touched tasks, so
        # a category with no rows in this pass must be ABSENT rather than present with a made-up
        # rate: absent contributes nothing, present at 0.0 would count as a measured failure.
        keys = tasks or list(by_cat)
        out[tag] = {t: (by_cat[t]["correct_rate"], by_cat[t]["n"]) for t in keys if t in by_cat}
    return out


def persist_scaffold(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path
