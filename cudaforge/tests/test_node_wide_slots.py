"""The runner cap must hold across PROCESSES, not just within one.

This is the defect that wedged a node twice. The cap was a threading.Semaphore, which counts
only inside the process that owns it — and the reward does not run in one process: verl builds
`reward_model.num_workers` RewardLoopWorker actors (default 8, see
verl/trainer/config/_generated_ppo_trainer.yaml:247), each of which loads this module by path
and so got its own counter. A nominal 12 was really 8 x 12 = 96.

Measured 2026-08-10: the A/B pass (540 rows chunked over 8 workers, one asyncio task per row)
put 85 kernel_runner processes on one GPU in 49 seconds. 113 processes ended in uninterruptible
sleep on os_acquire_rwlock_read, load 124, nvidia-smi itself hanging. Training with the same
code was fine because a step is only 48 candidates, six per worker.

Every test here therefore uses real subprocesses. A single-process test would have passed
against the broken implementation.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

CUDAFORGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child(slot_dir, n, hold_s, out_path):
    """Source for a process that takes one slot, records the moment, and holds it."""
    return textwrap.dedent(f"""
        import importlib.util, os, time
        os.environ["CUDAFORGE_MAX_CONCURRENT_RUNNERS"] = "{n}"
        os.environ["CUDAFORGE_SLOT_DIR"] = {slot_dir!r}
        os.environ["CUDAFORGE_SLOT_POLL_SEC"] = "0.02"
        # These exercise slot semantics, not node health; the watchdog has its own tests and
        # would otherwise refuse to run at all on a machine that is currently wedged.
        os.environ["CUDAFORGE_MAX_D_STATE"] = "100000"
        spec = importlib.util.spec_from_file_location(
            "rbr", os.path.join({CUDAFORGE!r}, "reward_bench_rubric.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with m._RUNNER_SLOTS.acquire(timeout=30):
            with open({out_path!r}, "a") as f:
                f.write(f"{{time.time():.6f}} in\\n"); f.flush()
            time.sleep({hold_s})
            with open({out_path!r}, "a") as f:
                f.write(f"{{time.time():.6f}} out\\n"); f.flush()
    """)


def _run_children(tmp_path, n_slots, n_children, hold_s=0.6):
    slot_dir = str(tmp_path / "slots")
    out = str(tmp_path / "events.txt")
    procs = [subprocess.Popen([sys.executable, "-c", _child(slot_dir, n_slots, hold_s, out)])
             for _ in range(n_children)]
    for p in procs:
        assert p.wait(timeout=180) == 0, "a child failed to acquire a slot"
    events = []
    for line in open(out):
        t, kind = line.split()
        events.append((float(t), kind))
    events.sort()
    peak = cur = 0
    for _, kind in events:
        cur += 1 if kind == "in" else -1
        peak = max(peak, cur)
    return peak, len(events) // 2


def test_the_cap_holds_across_processes(tmp_path):
    """Twelve processes, three slots: never more than three inside at once."""
    peak, completed = _run_children(tmp_path, n_slots=3, n_children=12)
    assert completed == 12, "not every process got through"
    assert peak <= 3, f"{peak} processes held a slot at once against a cap of 3"


def test_every_process_eventually_gets_a_slot(tmp_path):
    """A cap must throttle, not starve — the batch still has to finish."""
    peak, completed = _run_children(tmp_path, n_slots=2, n_children=6, hold_s=0.25)
    assert completed == 6
    assert peak <= 2


def test_a_slot_is_released_when_its_holder_dies(tmp_path):
    """flock is kernel-owned, so a crashed scorer cannot leak a slot.

    The counterpart property — a process stuck in D state keeps its slot because it has not
    died — is the one that makes this correct accounting for a wedged GPU, and is why the cap
    is not a userspace counter.
    """
    slot_dir = str(tmp_path / "slots")
    src = textwrap.dedent(f"""
        import importlib.util, os, time
        os.environ["CUDAFORGE_MAX_CONCURRENT_RUNNERS"] = "1"
        os.environ["CUDAFORGE_SLOT_DIR"] = {slot_dir!r}
        os.environ["CUDAFORGE_MAX_D_STATE"] = "100000"
        spec = importlib.util.spec_from_file_location(
            "rbr", os.path.join({CUDAFORGE!r}, "reward_bench_rubric.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with m._RUNNER_SLOTS.acquire(timeout=30):
            print("HELD", flush=True); time.sleep(120)
    """)
    holder = subprocess.Popen([sys.executable, "-c", src], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "HELD"
    holder.kill()
    holder.wait(timeout=30)

    t0 = time.time()
    peak, completed = _run_children(tmp_path, n_slots=1, n_children=1, hold_s=0.05)
    assert completed == 1, "the dead holder leaked its slot"
    assert time.time() - t0 < 30


def test_it_fails_loudly_rather_than_adding_to_a_stuck_pile(tmp_path):
    """When no slot frees up, the answer is to stop — not to proceed uncapped.

    Slots stay held only when their holders never finish, which on this workload means scorers
    in uninterruptible sleep on a wedged GPU driver. Adding more processes there is how a slow
    node becomes a rebooted one, so exceeding the wait must end the run, loudly, leaving the
    machine recoverable.
    """
    import importlib.util
    os.environ["CUDAFORGE_MAX_CONCURRENT_RUNNERS"] = "1"
    os.environ["CUDAFORGE_SLOT_DIR"] = str(tmp_path / "slots")
    os.environ["CUDAFORGE_SLOT_POLL_SEC"] = "0.02"
    os.environ["CUDAFORGE_MAX_D_STATE"] = "100000"
    spec = importlib.util.spec_from_file_location(
        "rbr", os.path.join(CUDAFORGE, "reward_bench_rubric.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with m._RUNNER_SLOTS.acquire(timeout=30):
        with pytest.raises(RuntimeError, match="no kernel_runner slot"):
            with m._RUNNER_SLOTS.acquire(timeout=0.2):
                pass


def test_the_two_rewards_cannot_cap_differently(tmp_path):
    """The ablation imports the pool rather than declaring its own — a divergent copy is how it
    missed the cap the first time."""
    import importlib.util
    mods = {}
    for name in ("reward_bench_rubric.py", "reward_rubric_ablation.py"):
        spec = importlib.util.spec_from_file_location(name[:-3],
                                                      os.path.join(CUDAFORGE, name))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mods[name] = mod
    a, b = mods["reward_bench_rubric.py"], mods["reward_rubric_ablation.py"]
    assert a._MAX_CONCURRENT == b._MAX_CONCURRENT
    assert a._RUNNER_SLOTS.dir == b._RUNNER_SLOTS.dir, \
        "the two rewards would queue on different slot directories, so neither caps the other"


# ---- the run must also stop itself when the node starts wedging ------------------------------
#
# The launch gate only looks once. Today's 85 runners appeared inside 49 seconds, so a run can
# start healthy and destroy the node before it finishes. Processes in uninterruptible sleep are
# the signature: they are queued on the GPU driver's global rwsem, they do not take SIGKILL, and
# every additional CUDA context lengthens that queue.

def _reward(monkeypatch, **env):
    import importlib.util
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    spec = importlib.util.spec_from_file_location(
        "rbr", os.path.join(CUDAFORGE, "reward_bench_rubric.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_d_state_count_matches_the_kernel(monkeypatch):
    """Counted from /proc, deliberately: nvidia-smi hangs once the driver is wedged, so the
    watchdog must not depend on anything that talks to the driver."""
    m = _reward(monkeypatch)
    ours = m._count_uninterruptible()
    ps = int(subprocess.run("ps -eo state= | grep -c D || true", shell=True,
                            capture_output=True, text=True).stdout.strip() or 0)
    assert abs(ours - ps) <= 3, f"our count {ours} vs ps {ps}"


def test_a_backing_up_node_stops_the_batch(monkeypatch, tmp_path):
    m = _reward(monkeypatch, CUDAFORGE_MAX_D_STATE=0, CUDAFORGE_SLOT_DIR=str(tmp_path / "s"),
                CUDAFORGE_MAX_CONCURRENT_RUNNERS=4)
    monkeypatch.setattr(m, "_count_uninterruptible", lambda: 999)
    with pytest.raises(RuntimeError, match="uninterruptible sleep"):
        with m._RUNNER_SLOTS.acquire(timeout=5):
            pass


def test_a_healthy_node_is_not_blocked(monkeypatch, tmp_path):
    m = _reward(monkeypatch, CUDAFORGE_MAX_D_STATE=40, CUDAFORGE_SLOT_DIR=str(tmp_path / "s2"),
                CUDAFORGE_MAX_CONCURRENT_RUNNERS=4)
    monkeypatch.setattr(m, "_count_uninterruptible", lambda: 3)
    with m._RUNNER_SLOTS.acquire(timeout=5) as fd:
        assert fd is not None


def test_the_check_is_before_the_slot_not_after(monkeypatch, tmp_path):
    """Stopping matters only if it happens before another context is created."""
    src = open(os.path.join(CUDAFORGE, "reward_bench_rubric.py"), encoding="utf-8").read()
    body = src[src.index("    def acquire(self, timeout=None):"):]
    assert body.index("_abort_if_node_is_wedging()") < body.index("fcntl.flock"), \
        "the watchdog fires after a slot is taken, which is one context too late"


def test_the_log_can_tell_queued_from_spawned(monkeypatch, tmp_path):
    """A call that logged only `start` is ambiguous, and that ambiguity blocked an investigation.

    The 2026-08-10 A/B left 429 calls with a start record and nothing else. Two very different
    states produce that: still queued for a slot, where no child exists and nothing is at risk;
    or spawned and then abandoned, leaving an orphaned runner holding a CUDA context — the exact
    shape that deadlocks the driver. `spawned` is written the moment a slot is held, which is the
    moment a child becomes possible, so the two are now distinguishable after the fact.
    """
    import importlib.util, json, subprocess as _sp
    monkeypatch.setenv("CUDAFORGE_MAX_D_STATE", "100000")
    monkeypatch.setenv("CUDAFORGE_SLOT_DIR", str(tmp_path / "slots"))
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "rbr", os.path.join(CUDAFORGE, "reward_bench_rubric.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class _P:
        returncode = 0
        stderr = b""
        stdout = json.dumps({"ok": True, "correct": True, "speedup": 1.0}).encode()

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _P())
    m.bench("class ModelNew: pass", "class Model: pass")

    def phases():
        return [r.get("phase") for f in
                __import__("glob").glob(
                    str(tmp_path / "cudaforge_logs" / "bench" / "*" / "*.jsonl"))
                for l in open(f) if l.strip() for r in [json.loads(l)]]

    # A SUCCESSFUL score writes start then spawned and stops: `finish` is only written when the
    # runner failed, or when log_on_success is set. Learning that corrected a misreading of the
    # 2026-08-10 logs, where 429 calls looked abandoned and were in fact the ones that worked.
    got = phases()
    assert got == ["start", "spawned"], f"success path should be start -> spawned, got {got}"

    class _Bad:
        returncode = 1
        stderr = b"boom"
        stdout = b""
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Bad())
    m.bench("class ModelNew: pass", "class Model: pass")
    got = phases()
    assert got[-1] == "finish", f"a failed score must record why, got {got}"
    assert got.count("spawned") == 2, "every attempt that reaches a slot must say so"


def test_a_multi_gpu_pool_pins_each_holder_to_one_card(tmp_path):
    """Three gpus x cap 2 = six concurrent, and every acquire names the card it granted."""
    import importlib.util
    os.environ["CUDAFORGE_MAX_CONCURRENT_RUNNERS"] = "2"
    os.environ["CUDAFORGE_SLOT_DIR"] = str(tmp_path / "slots")
    os.environ["CUDAFORGE_MAX_D_STATE"] = "100000"
    os.environ["REWARD_CUDA_VISIBLE_DEVICES"] = "4,5,6"
    try:
        spec = importlib.util.spec_from_file_location(
            "rbr_multi", os.path.join(CUDAFORGE, "reward_bench_rubric.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        import contextlib as _ctx
        held, gpus = [], set()
        with _ctx.ExitStack() as stack:
            for _ in range(6):
                fd, gpu = stack.enter_context(m._RUNNER_SLOTS.acquire(timeout=5))
                held.append(fd)
                gpus.add(gpu)
            assert gpus == {"4", "5", "6"}, f"pool did not span every card: {gpus}"
            with pytest.raises(RuntimeError, match="no kernel_runner slot"):
                with m._RUNNER_SLOTS.acquire(timeout=0.3):
                    pass
    finally:
        for k in ("CUDAFORGE_MAX_CONCURRENT_RUNNERS", "CUDAFORGE_SLOT_DIR",
                  "CUDAFORGE_MAX_D_STATE", "REWARD_CUDA_VISIBLE_DEVICES"):
            os.environ.pop(k, None)
