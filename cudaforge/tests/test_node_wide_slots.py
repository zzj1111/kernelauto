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
