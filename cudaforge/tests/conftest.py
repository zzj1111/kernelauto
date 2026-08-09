"""Load the reward modules and stub everything that needs a GPU or a network.

verl loads these modules by PATH (custom_reward_function.path), not as a package, so they are
not importable as `cudaforge.reward_bench_rubric` from an installed tree. They do import
cleanly without torch — only `requests` is pulled in at module level — which is what makes
behavioural tests possible at all on a machine with no free GPU.

Two seams are stubbed, and they are the only two the reward reaches out through:
  * bench(...)             -> the nvcc compile + correctness check + timing subprocess
  * _run_rubric_judge(...) -> the HTTP call to the vLLM rubric server
Stubbing them leaves the part under test — the composition arithmetic, the degradation paths,
and the attribution prints the controller scrapes — running for real.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

CUDAFORGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(filename):
    path = os.path.join(CUDAFORGE, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def reward_src():
    """The main reward module, imported once."""
    return _load("reward_bench_rubric.py")


@pytest.fixture(scope="session")
def ablation_src():
    return _load("reward_rubric_ablation.py")


@pytest.fixture
def reward(reward_src, monkeypatch):
    """The reward module with its two outward seams stubbed.

    Yields the module plus a `stub` handle:
        stub.bench = (correctness, speedup)   what the benchmark subprocess "returns"
        stub.rubric = {...}                   what the judge "returns"
        stub.bench_calls / stub.rubric_calls  what it was asked
    """
    class Stub:
        bench = (1, 2.0)
        rubric = None
        bench_calls = []
        rubric_calls = []

    stub = Stub()

    def fake_bench(solution_str, reference_str, *a, **k):
        stub.bench_calls.append((solution_str, reference_str))
        out = stub.bench
        if isinstance(out, BaseException):
            raise out
        return out

    def fake_judge(data_source, *, reference_code, candidate_code, extra_info=None, **k):
        stub.rubric_calls.append(
            {"data_source": data_source, "candidate_code": candidate_code,
             "extra_info": extra_info})
        out = stub.rubric
        if isinstance(out, BaseException):
            raise out
        if out is None and hasattr(reward_src, "_default_neutral_rubric"):
            # The module's own neutral rubric, so an unset stub means "the judge answered
            # neutrally" rather than a shape this code never produces.
            return reward_src._default_neutral_rubric(data_source)
        return out

    monkeypatch.setattr(reward_src, "bench", fake_bench)
    monkeypatch.setattr(reward_src, "_run_rubric_judge", fake_judge)
    reward_src._stub = stub
    yield reward_src
    delattr(reward_src, "_stub")


@pytest.fixture
def stub(reward):
    return reward._stub
