"""Defects found by an adversarial audit of the reward path, pinned.

Every one of these was rated "corrupts the training signal" and survived a refutation pass.
They share a shape worth naming: a value that cannot be compared (NaN) or cannot be trusted
(free-form JSON from an LLM) is fed to a comparison that quietly does the wrong thing, and the
result is a WRONG NUMBER rather than an error.
"""
from __future__ import annotations

import math

import pytest


# ---- NaN defeats the correctness gate --------------------------------------------------------
#
# kernel_runner's gate is `if max_err_mean > tol or mean_err_mean > tol: -> incorrect`. A NaN
# anywhere in the candidate output makes both aggregates NaN; every comparison with NaN is
# False, so the failure branch is SKIPPED and an all-NaN kernel is reported correct with a real
# speedup. The direction is the worst possible: skipping the numerical work makes a kernel both
# more NaN-prone AND faster, so RL selects for it. (Inf is already caught: inf > tol is True.)

def test_the_gate_comparison_alone_cannot_reject_a_nan():
    """The premise, stated as arithmetic so the fix below has a reason to exist."""
    tol = 1e-3
    assert not (float("nan") > tol), "if this ever changes, the guard can go"
    assert float("inf") > tol, "Inf was always caught; only NaN needed the new branch"


def test_a_non_finite_output_error_is_decided_before_the_tolerance_gate():
    src = open("cudaforge/kernel_runner.py", encoding="utf-8").read()
    gate = src.index("if max_err_mean > tol or mean_err_mean > tol:")
    guard = src.index("if nonfinite_output is not None or shape_mismatch is not None:")
    assert guard < gate, (
        "the uncomparable-output branch must run BEFORE the tolerance gate; after it, a NaN has "
        "already been read as 'within tolerance' and the kernel is marked correct")


def test_a_nan_error_is_detected_where_it_is_computed():
    src = open("cudaforge/kernel_runner.py", encoding="utf-8").read()
    assert "if not (math.isfinite(max_err) and math.isfinite(mean_err)):" in src, \
        "the per-input NaN check is gone; NaN flows into the aggregate and defeats the gate"


def test_outputs_of_the_wrong_shape_are_not_broadcast_into_a_comparison():
    src = open("cudaforge/kernel_runner.py", encoding="utf-8").read()
    sub = src.index("diff = (tst_t - ref_t).abs()")
    assert "if tuple(tst_t.shape) != tuple(ref_t.shape):" in src[:sub], \
        "without a shape check, a wrong-shaped output broadcasts and is compared to a different "\
        "tensor instead of failing"


# ---- an LLM's boolean is not a Python bool ---------------------------------------------------
#
# major_hacking hard-zeroes the reward and arrives as free-form JSON. bool() makes every
# non-empty string truthy, so a judge rendering the field as the string "false" would zero every
# correct, fast kernel it looked at — and the debug log would call each one reward hacking.

@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False),
    ("true", True), ("false", False), ("False", False), ('"false"', False),
    ("yes", True), ("no", False), (1, True), (0, False), (None, False),
])
def test_the_hacking_flag_is_parsed_not_coerced(reward_src, raw, expected):
    assert reward_src._as_flag(raw) is expected, (
        f"{raw!r} parsed as {reward_src._as_flag(raw)!r}; bool({raw!r}) would be {bool(raw)!r}")


def test_an_unparseable_flag_does_not_destroy_the_measured_reward(reward_src):
    """None means 'the judge did not say'. The safe direction is to leave the reward alone:
    a wrongly-applied zero is unrecoverable, a wrongly-skipped one costs only vigilance."""
    assert reward_src._as_flag("maybe") is None
    assert reward_src._as_flag({}) is None


def test_a_string_false_no_longer_zeroes_a_good_kernel(reward, stub):
    """End to end through compute_score: correct, fast, and the judge writes "false"."""
    stub.bench = (1, 4.0)
    stub.rubric = {"anti_hacking": 5, "bottleneck_coverage": 5, "cuda_perf_quality": 5,
                   "multi_component_focus": 5, "major_hacking": "false", "total": 20}
    r = reward.compute_score("CudaForge", "class ModelNew: pass", "gt",
                             {"answer": "class Model: pass"})
    assert r > 0.0, "a correct, fast kernel was zeroed because the judge quoted its boolean"
    assert math.isfinite(r) and r <= 3.0


def test_a_real_flag_still_zeroes(reward, stub):
    """The guard must not have disarmed the gate it protects."""
    stub.bench = (1, 4.0)
    stub.rubric = {"anti_hacking": 1, "bottleneck_coverage": 1, "cuda_perf_quality": 1,
                   "multi_component_focus": 1, "major_hacking": True, "total": 0}
    assert reward.compute_score("CudaForge", "class ModelNew: pass", "gt",
                                {"answer": "class Model: pass"}) == 0.0
