"""Contracts the reward owes the rest of the system, checked without a GPU.

These assert only properties the code documents about ITSELF plus the cross-module contract
with the controller. They deliberately do not pin the composition arithmetic — an audit of that
is in flight, and a characterisation test written first would enshrine whatever it finds.

The contract that matters most is the printed line: cudascaffold does not read this function's
return value for its per-category and per-instance signals, it scrapes stdout. If that format
drifts by one character the Teacher goes blind and nothing fails loudly.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def _score(reward, *, correctness=1, speedup=2.0, rubric=None, extra=None, stub=None):
    stub.bench = (correctness, speedup)
    stub.rubric = rubric
    info = {"answer": "class Model: pass"}
    info.update(extra or {})
    return reward.compute_score("CudaForge", "class ModelNew: pass", "gt", info)


def test_a_missing_reference_scores_zero_and_never_benches(reward, stub):
    assert reward.compute_score("CudaForge", "code", "gt", None) == 0.0
    assert reward.compute_score("CudaForge", "code", "gt", {}) == 0.0
    assert stub.bench_calls == [], "benched a candidate with no reference to check it against"


def test_incorrect_candidates_score_zero_and_skip_the_judge(reward, stub):
    assert _score(reward, correctness=0, speedup=9.0, stub=stub) == 0.0
    assert stub.rubric_calls == [], "paid for a rubric judgement on a candidate already scored 0"


def test_the_reward_stays_inside_its_documented_cap(reward, stub):
    """min(reward, 3.0) is the documented ceiling; nothing may exceed it, at any speedup."""
    for sp in (0.0, 1.0, 4.9, 5.0, 5.1, 1e6):
        r = _score(reward, speedup=sp, stub=stub)
        assert 0.0 <= r <= 3.0, f"speedup={sp} produced {r}, outside [0, 3]"


def test_speedup_is_clipped_so_an_outlier_cannot_dominate(reward, stub):
    """Above the clip, more speedup buys nothing — the documented reason the cap exists."""
    assert _score(reward, speedup=5.0, stub=stub) == _score(reward, speedup=50.0, stub=stub)


def test_more_speedup_is_never_worth_less(reward, stub):
    """Monotone below the clip. A dip would mean a better kernel is punished."""
    rs = [_score(reward, speedup=s, stub=stub) for s in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)]
    assert rs == sorted(rs), f"reward is not monotone in speedup: {rs}"


# ---- the stdout contract with cudascaffold ---------------------------------------------------

def test_every_scored_candidate_prints_a_line_the_controller_can_parse(reward, stub, capsys):
    from cudascaffold.adapters import REWARD_LINE
    for correctness, sp in ((1, 2.0), (0, 0.0)):
        capsys.readouterr()
        _score(reward, correctness=correctness, speedup=sp, stub=stub,
               extra={"task_name": "conv_fwd", "level": 0, "category": "conv"})
        out = capsys.readouterr().out
        m = REWARD_LINE.search(out)
        assert m, (f"correctness={correctness}: nothing the controller's REWARD_LINE regex "
                   f"matches was printed.\n--- stdout ---\n{out}")


def test_the_printed_fields_carry_the_instance_and_its_category(reward, stub, capsys):
    from cudascaffold.adapters import REWARD_LINE, cat_of_level
    capsys.readouterr()
    _score(reward, stub=stub, extra={"task_name": "matmul_tiled", "level": 0,
                                     "category": "matmul"})
    m = REWARD_LINE.search(capsys.readouterr().out)
    assert m, "no parseable reward line"
    correct, speedup, ds, task, level, category = m.groups()
    assert task == "matmul_tiled"
    assert cat_of_level(level, category) == "matmul", (
        f"the controller would file this candidate under {cat_of_level(level, category)!r}")


def test_a_candidate_without_a_task_name_still_reports_its_category(reward, stub, capsys):
    """The CudaForge half carries no task_name — it must still be attributable by category,
    or a whole dataset disappears from the Teacher's breakdown."""
    from cudascaffold.adapters import REWARD_LINE, cat_of_level
    capsys.readouterr()
    _score(reward, stub=stub, extra={"level": 2})
    m = REWARD_LINE.search(capsys.readouterr().out)
    assert m, "no parseable reward line for a task_name-less candidate"
    _c, _s, _ds, _task, level, category = m.groups()
    assert cat_of_level(level, category) == "improve_l2"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_a_degenerate_speedup_cannot_produce_a_degenerate_reward(reward, stub, bad):
    """The timing subprocess is the least trustworthy input in the system; whatever it returns,
    the reward handed to the optimiser has to be a finite number inside the cap."""
    import math
    r = _score(reward, speedup=bad, stub=stub)
    assert isinstance(r, float) and math.isfinite(r), f"speedup={bad} produced {r!r}"
    assert 0.0 <= r <= 3.0


# ---- the NaN guards, pinned explicitly ------------------------------------------------------
#
# max(0.0, min(nan, 3.0)) happens to be 0.0 because every comparison with NaN is False, so the
# clamp swallows a NaN by accident of Python's semantics rather than by intent. That accident
# does not survive reordering: min(3.0, max(nan, 0.0)) is 3.0, which would hand a failed
# measurement the MAXIMUM reward. So the explicit guards are pinned on their own diagnostics,
# not through the clamp.

def test_the_composition_reports_a_non_finite_shaped_reward(reward, stub, capsys):
    capsys.readouterr()
    _score(reward, speedup=float("nan"), stub=stub)
    assert "NON-FINITE" in capsys.readouterr().out, (
        "a NaN reached the final composition and was clamped silently — the explicit isfinite "
        "branch is gone, and only argument order is standing between NaN and a full-marks reward")


def test_bench_never_lets_a_non_finite_speedup_out(reward_src, monkeypatch, capsys):
    """The real bench(), with only the runner subprocess stubbed.

    json.loads accepts NaN and Infinity verbatim, so a bad timing crosses the process boundary
    intact; this is the seam where it has to be stopped.
    """
    import json as _json
    import math
    import subprocess as _sp

    class _P:
        returncode = 0
        stderr = b""

        def __init__(self, payload):
            self.stdout = _json.dumps(payload).encode()

    for raw in (float("nan"), float("inf"), float("-inf")):
        payload = {"ok": True, "correct": True, "speedup": raw}
        monkeypatch.setattr(_sp, "run", lambda *a, **k: _P(payload))
        monkeypatch.setattr(reward_src.subprocess, "run", lambda *a, **k: _P(payload))
        correctness, speedup = reward_src.bench("class ModelNew: pass", "class Model: pass")
        assert math.isfinite(speedup), f"bench returned speedup={speedup!r} for raw={raw!r}"
        assert correctness == 1, (
            "a broken TIMING was turned into a wrong-answer verdict; correctness was already "
            "decided by the runner and a measurement failure must not overturn it")
