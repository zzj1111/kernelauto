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


@pytest.mark.parametrize("name", REWARDS)
def test_the_cap_is_declared_and_configurable(name):
    src = _src(name)
    if "subprocess.run(" not in src:
        pytest.skip(f"{name} spawns nothing")
    assert "CUDAFORGE_MAX_CONCURRENT_RUNNERS" in src, (
        f"{name} must read the cap from the same env var as its sibling, so raising it on one "
        f"path cannot silently leave the other unbounded")


def test_the_two_rewards_agree_on_the_cap():
    """A divergent default is how the first copy went stale — same knob, same number."""
    pat = re.compile(r'CUDAFORGE_MAX_CONCURRENT_RUNNERS"\s*,\s*"(\d+)"')
    caps = {name: pat.search(_src(name)) for name in REWARDS}
    found = {name: m.group(1) for name, m in caps.items() if m}
    assert len(found) == len(REWARDS), f"cap not found in: {set(REWARDS) - set(found)}"
    assert len(set(found.values())) == 1, f"the two rewards cap differently: {found}"


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
