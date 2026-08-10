"""Every path that spawns kernel_runner must be bounded.

Each runner creates its own CUDA context. On 2026-08-04, 243 concurrent ones wedged the GPU
driver for every user of the node — this box is shared, so the blast radius is other people's
jobs, not just ours. reward_bench_rubric.py got a semaphore that day; reward_rubric_ablation.py
is a copy taken before it existed and did not, and its bench() is deliberately kept for offline
debugging, so the unbounded path stayed one call away from being live.

These tests read the source rather than importing it: the reward modules pull in torch and are
loaded by verl through a path, not as a package, and the property being checked is structural.
"""
from __future__ import annotations

import os
import re

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARDS = ("reward_bench_rubric.py", "reward_rubric_ablation.py")


def _src(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("name", REWARDS)
def test_the_runner_subprocess_is_inside_the_semaphore(name):
    src = _src(name)
    spawns = [m.start() for m in re.finditer(r"subprocess\.run\(\s*\n\s*cmd,", src)]
    if not spawns:
        pytest.skip(f"{name} spawns no runner subprocess")
    for pos in spawns:
        window = src[max(0, pos - 400):pos]
        assert "_RUNNER_SLOTS" in window, (
            f"{name}: a kernel_runner subprocess is spawned without holding _RUNNER_SLOTS. "
            f"243 unbounded runners took the GPU driver down for the whole node once.")


def test_both_rewards_share_one_slot_pool():
    """Stronger than "both mention the same env var", which is what this asserted before.

    A divergent copy is how the ablation missed the cap in the first place, so the ablation now
    imports the pool from its sibling instead of declaring one. Identity of the object is the
    property that matters: two pools would queue on different slot directories and neither would
    cap the other.
    """
    import importlib.util
    mods = {}
    for name in REWARDS:
        spec = importlib.util.spec_from_file_location(name[:-3], os.path.join(HERE, name))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        mods[name] = m
    a, b = (mods[n] for n in REWARDS)
    assert a._MAX_CONCURRENT == b._MAX_CONCURRENT, "the two rewards cap differently"
    assert a._RUNNER_SLOTS.dir == b._RUNNER_SLOTS.dir, \
        "different slot directories: neither pool constrains the other"


def test_the_cap_is_configurable_from_one_place():
    src = _src("reward_bench_rubric.py")
    assert "CUDAFORGE_MAX_CONCURRENT_RUNNERS" in src, \
        "the cap must stay settable without editing code"
    # Checked on the parsed code, not the text: the comment above the pool explains what a
    # threading.Semaphore did wrong, and must not itself trip this.
    import ast
    names = {f"{getattr(n.value,'id','')}.{n.attr}" for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Attribute)}
    assert "threading.Semaphore" not in names, (
        "a threading.Semaphore counts inside ONE process, and verl runs this module in "
        "reward_model.num_workers separate actors — that is what made a nominal 12 into 96")


def test_the_ablation_reward_never_benches():
    """It is documented as rubric-only; if that stops being true the cap is not the only concern
    (the ablation would silently become a benchmark run costing GPU time per rollout)."""
    import ast
    tree = ast.parse(_src("reward_rubric_ablation.py"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "compute_score")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "bench" not in called, \
        "compute_score now calls bench() — the ablation is no longer rubric-only"
