#!/usr/bin/env python3
"""Keep an unattended arm alive without being able to hurt the node it shares.

Every INTERVAL seconds: relaunch run_arm if it died short of its target step, relaunch the
rubric judge if its port went dark, prune regenerable build caches, prune orphan checkpoints,
and watch for the two failure modes that have actually taken a machine down (runaway
kernel_runner concurrency, D-state pileup from a wedged GPU driver).

Every rule here was earned on a real incident during the 2026-08-10 overnight smoke:
  - crash-loop guard: 3 relaunches in 30 min writes HALT and stops — a process that dies at
    import, relaunched forever, is how a log fills a disk.
  - disk guard + cache pruning: unbounded checkpoints filled an 868G root disk to 100% and
    killed a save mid-write.
  - orphan checkpoint pruning: verl's max_actor_ckpt_to_keep only prunes saves its own process
    made, so every crash-relaunch strands the previous process's checkpoints. Guards: the
    latest pointer only advances on a COMPLETED save, a live save always targets a step newer
    than latest, so a dir >= 2 save-periods behind latest and untouched for 30 min cannot be a
    live target. (Deleting a dir a live saver was about to use caused one of the crashes.)
  - D-state circuit breaker: sustained pileup SIGTERMs the TRAINING (our load), never other
    users' jobs, and leaves HALT for a human.

Zero hardcoded paths. Configuration comes from the arm's own env file:

    python scripts/watchdog.py --env-file /path/to/env.sh [--judge-gpu 7] [--interval 300]

where env.sh is the same `export ARM_*=...` file the arm itself is launched with. Everything
the watchdog needs (repo root, python, exp name, exp/ckpt roots, target step, judge model) is
read from it; flags only override.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

D_LIMIT = 40           # sustained D-state processes that mean the driver is wedging
RUNNER_LIMIT = 16      # the flock slot pool caps at 12; above this the cap is broken
RELAUNCH_WINDOW = 1800
RELAUNCH_MAX = 3
DISK_MIN_FREE_GB = 25
CACHE_MAX_AGE_SEC = 3600
ORPHAN_MIN_AGE_SEC = 1800


def parse_env_file(path):
    """{K: V} from `export K=V` lines. Values may be quoted; $VARS are NOT expanded except
    via os.path.expandvars against the current environment, which covers $HOME-style usage."""
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln.startswith("export ") or "=" not in ln:
            continue
        k, v = ln[len("export "):].split("=", 1)
        out[k.strip()] = os.path.expandvars(shlex.split(v)[0] if v else "")
    return out


def build_cfg():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env-file", required=True, help="the arm's env.sh (export ARM_*=... lines)")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--judge-gpu", default=None,
                    help="GPU id for relaunching the rubric judge; omit to not manage the judge")
    ap.add_argument("--judge-port", type=int, default=8210)
    ap.add_argument("--judge-model", default=None,
                    help="model dir for the judge (default: JUDGE_MODEL from the env file)")
    a = ap.parse_args()

    env = parse_env_file(a.env_file)

    def need(key):
        if not env.get(key):
            sys.exit(f"env file {a.env_file} does not export {key}, which the watchdog needs")
        return env[key]

    root = need("ARM_ROOT")
    exp = need("ARM_EXP")
    base = os.path.dirname(os.path.abspath(a.env_file))
    return {
        "env_file": os.path.abspath(a.env_file), "base": base, "interval": a.interval,
        "root": root, "py": env.get("ARM_PYTHON") or sys.executable, "exp": exp,
        "exp_root": env.get("ARM_EXP_ROOT") or os.path.join(base, "exp"),
        "ckpt_root": env.get("ARM_CKPT_ROOT") or os.path.join(base, "ckpts"),
        "target": int(env.get("ARM_TARGET_STEP") or 0) or None,
        "judge_gpu": a.judge_gpu, "judge_port": a.judge_port,
        "judge_model": a.judge_model or env.get("JUDGE_MODEL"),
        "log": os.path.join(base, "watchdog.log"),
        "halt": os.path.join(base, "HALT"),
        "run_log": os.path.join(base, "run.log"),
    }


CFG = None
_relaunches = []
_d_high_streak = 0


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n"
    with open(CFG["log"], "a") as f:
        f.write(line)
    if os.path.getsize(CFG["log"]) > 2_000_000:
        os.rename(CFG["log"], CFG["log"] + ".1")


def procs():
    """[(pid, argv)] argv-parsed from /proc, so a string MENTIONING a name never matches."""
    out = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            argv = [a for a in open(p, "rb").read().decode(errors="ignore").split("\0") if a]
        except OSError:
            continue
        if argv:
            out.append((int(p.split("/")[2]), argv))
    return out


def is_py(argv, needle):
    return argv[0].rsplit("/", 1)[-1].startswith("python") and any(needle in a for a in argv[1:])


def d_count():
    n = 0
    for p in glob.glob("/proc/[0-9]*/stat"):
        try:
            s = open(p).read()
            if s[s.rindex(")") + 2] == "D":
                n += 1
        except (OSError, ValueError):
            continue
    return n


def state_step():
    try:
        st = json.load(open(os.path.join(CFG["exp_root"], CFG["exp"], "state.json")))
        return st["step"]
    except (OSError, ValueError, KeyError):
        return None


def relaunch_arm():
    now = time.time()
    _relaunches[:] = [t for t in _relaunches if now - t < RELAUNCH_WINDOW]
    if len(_relaunches) >= RELAUNCH_MAX:
        log(f"CRASH-LOOP: {RELAUNCH_MAX} relaunches in {RELAUNCH_WINDOW}s — halting")
        open(CFG["halt"], "w").write("crash-loop\n")
        return
    _relaunches.append(now)
    with open(CFG["run_log"], "a") as out:
        subprocess.Popen(
            ["bash", "-c",
             f"source {shlex.quote(CFG['env_file'])} && exec \"$ARM_PYTHON\" -m cudascaffold.run_arm"],
            cwd=CFG["root"], stdout=out, stderr=out, start_new_session=True)
    log("relaunched run_arm")


def judge_up():
    r = subprocess.run(["ss", "-tln"], capture_output=True, text=True)
    return f":{CFG['judge_port']}" in r.stdout


def relaunch_judge():
    if not (CFG["judge_gpu"] is not None and CFG["judge_model"]):
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(CFG["judge_gpu"]))
    with open(os.path.join(CFG["base"], "judge.log"), "a") as out:
        subprocess.Popen(
            [CFG["py"], "-m", "vllm.entrypoints.openai.api_server",
             "--model", CFG["judge_model"], "--served-model-name", "rubric-judge",
             "--host", "127.0.0.1", "--port", str(CFG["judge_port"]),
             "--gpu-memory-utilization", "0.40", "--max-model-len", "16384",
             "--disable-log-requests"],
            env=env, stdout=out, stderr=out, start_new_session=True)
    log(f"relaunched judge on GPU {CFG['judge_gpu']}")


def stop_training(reason):
    open(CFG["halt"], "w").write(reason + "\n")
    for pid, argv in procs():
        if is_py(argv, "cudascaffold.run_arm") or is_py(argv, "verl.trainer.main_ppo"):
            try:
                os.kill(pid, signal.SIGTERM)
                log(f"SIGTERM {pid} ({reason})")
            except OSError:
                pass


def prune_disk():
    try:
        st = os.statvfs(CFG["ckpt_root"] if os.path.isdir(CFG["ckpt_root"]) else "/")
        free_gb = st.f_bavail * st.f_frsize / 1e9
        if free_gb < DISK_MIN_FREE_GB:
            log(f"DISK LOW: {free_gb:.0f}G free under {CFG['ckpt_root']}")
        for root in glob.glob("/dev/shm/torch_ext_cache_reward_cuda*") + \
                    glob.glob("/tmp/torch_ext_cache_reward_cuda*"):
            for cache_dir in glob.glob(root + "/*"):
                try:
                    if time.time() - os.path.getmtime(cache_dir) > CACHE_MAX_AGE_SEC:
                        shutil.rmtree(cache_dir, ignore_errors=True)
                except OSError:
                    pass
        ck = os.path.join(CFG["ckpt_root"], CFG["exp"])
        latest_f = os.path.join(ck, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_f):
            latest = int(open(latest_f).read().strip())
            for cd in glob.glob(os.path.join(ck, "global_step_*")):
                n = int(cd.rsplit("_", 1)[-1])
                if n <= latest - 4 and time.time() - os.path.getmtime(cd) > ORPHAN_MIN_AGE_SEC:
                    shutil.rmtree(cd, ignore_errors=True)
                    log(f"pruned orphan checkpoint global_step_{n} (latest={latest})")
    except (OSError, ValueError):
        pass


def main():
    global _d_high_streak
    log(f"watchdog up: exp={CFG['exp']} root={CFG['root']} interval={CFG['interval']}s")
    while True:
        try:
            d = d_count()
            runners = sum(1 for _, a in procs() if is_py(a, "kernel_runner.py"))
            arm_alive = any(is_py(a, "cudascaffold.run_arm") for _, a in procs())
            step, tgt = state_step(), CFG["target"]

            if d > D_LIMIT:
                _d_high_streak += 1
            else:
                _d_high_streak = 0
            if _d_high_streak >= 2:
                log(f"D-STATE PILEUP: {d} twice running — stopping training to protect the node")
                stop_training(f"d_state={d}")
                _d_high_streak = 0
            elif runners > RUNNER_LIMIT:
                log(f"RUNNER CAP BROKEN: {runners} concurrent kernel_runner (pool caps at 12)")

            prune_disk()

            if os.path.exists(CFG["halt"]):
                log(f"halted ({open(CFG['halt']).read().strip()}); arm_alive={arm_alive} d={d}")
            else:
                if not arm_alive and (step is None or tgt is None or step < tgt):
                    log(f"arm dead at step={step} target={tgt} — relaunching")
                    relaunch_arm()
                if CFG["judge_gpu"] is not None and not judge_up():
                    log(f"judge port {CFG['judge_port']} down — relaunching")
                    relaunch_judge()
                log(f"ok arm={'up' if arm_alive else 'down'} step={step}/{tgt} d={d} "
                    f"runners={runners} load={open('/proc/loadavg').read().split()[0]}")
        except Exception as e:            # the watchdog itself must never die
            try:
                log(f"watchdog error: {e!r}")
            except OSError:
                pass
        time.sleep(CFG["interval"])


if __name__ == "__main__":
    CFG = build_cfg()
    main()
