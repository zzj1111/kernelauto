"""Defects found by an adversarial audit of the reward path, pinned.

Every one of these was rated "corrupts the training signal" and survived a refutation pass.
They share a shape worth naming: a value that cannot be compared (NaN) or cannot be trusted
(free-form JSON from an LLM) is fed to a comparison that quietly does the wrong thing, and the
result is a WRONG NUMBER rather than an error.
"""
from __future__ import annotations

import math
import os

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


# ---- attribution: one line per candidate, emitted after the reward is known -------------------
#
# The line was printed right after bench() and before the rubric gate, so a kernel zeroed for
# major_hacking was reported to the Teacher as a success with a positive speedup — correct and
# fast on the record, worth nothing in the gradient. And the missing-reference path returned
# without printing at all, so those candidates did not exist as far as the controller was
# concerned.

def _lines(out):
    from cudascaffold.adapters import REWARD_LINE
    return [m.groups() for m in REWARD_LINE.finditer(out)]


def test_exactly_one_attribution_line_per_candidate(reward, stub, capsys):
    cases = [
        dict(bench=(1, 2.0), rubric=None),                                   # ordinary success
        dict(bench=(0, 0.0), rubric=None),                                   # incorrect
        dict(bench=(1, 4.0), rubric={"major_hacking": True, "total": 0}),    # zeroed by rubric
    ]
    for c in cases:
        stub.bench, stub.rubric = c["bench"], c["rubric"]
        capsys.readouterr()
        reward.compute_score("CudaForge", "class ModelNew: pass", "gt",
                             {"answer": "class Model: pass", "task_name": "t", "level": 0,
                              "category": "conv"})
        got = _lines(capsys.readouterr().out)
        assert len(got) == 1, f"{c}: emitted {len(got)} attribution lines, expected exactly 1"


def test_a_candidate_with_no_reference_is_still_attributed(reward, stub, capsys):
    capsys.readouterr()
    reward.compute_score("CudaForge", "code", "gt", {"task_name": "t", "level": 0})
    assert len(_lines(capsys.readouterr().out)) == 1, \
        "returning silently made these candidates invisible to the Teacher"


def test_a_kernel_zeroed_for_hacking_is_not_logged_as_a_success(reward, stub, capsys):
    stub.bench = (1, 4.0)
    stub.rubric = {"anti_hacking": 1, "bottleneck_coverage": 1, "cuda_perf_quality": 1,
                   "multi_component_focus": 1, "major_hacking": True, "total": 0}
    capsys.readouterr()
    r = reward.compute_score("CudaForge", "class ModelNew: pass", "gt",
                             {"answer": "class Model: pass", "task_name": "t", "level": 0})
    out = capsys.readouterr().out
    assert r == 0.0
    (correct, speedup, _ds, _task, _level, _cat, rew), = _lines(out)
    assert rew is not None and float(rew) == 0.0, (
        f"the line reports reward={rew}; the Teacher would credit the text that produced a "
        f"kernel worth nothing")
    assert correct == "1", "the bench verdict is still reported truthfully"


def test_the_controller_still_parses_lines_written_before_the_field_existed():
    """Old Ray logs must keep working; the field is additive."""
    from cudascaffold.adapters import REWARD_LINE
    legacy = ("correctness: 1, speedup: 2.5, data_source:CudaForge, task_name:conv_a, "
              "level:0, category:conv")
    m = REWARD_LINE.search(legacy)
    assert m, "the regex no longer matches a pre-reward log line"
    assert m.groups()[-1] is None and m.group(4) == "conv_a"


# ---- build directories must not collide ------------------------------------------------------
#
# "shared" pointed every concurrent runner at ONE directory per GPU, and torch's build directory
# inside it is keyed only on the load_inline name= that the MODEL chooses — models reuse names
# like "fused_op" constantly. Two candidates compiling under the same name at once overwrite each
# other's sources in one directory, so a candidate can be graded on another candidate's .so.

def _ext_dir_for(reward_src, monkeypatch, solution, visible="3"):
    """Run bench() far enough to capture the env it would hand the runner."""
    import json as _json
    import subprocess as _sp
    seen = {}

    class _P:
        returncode = 0
        stderr = b""
        stdout = _json.dumps({"ok": True, "correct": True, "speedup": 1.0}).encode()

    def fake_run(*a, **k):
        seen["env"] = dict(k.get("env") or {})
        return _P()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    # These assert build-directory isolation, not node health. The watchdog has its own tests
    # and would otherwise stop this one on a machine whose driver is currently wedged.
    monkeypatch.setattr(reward_src, "_abort_if_node_is_wedging", lambda: None)
    monkeypatch.setattr(reward_src.subprocess, "run", fake_run)
    reward_src.bench(solution, "class Model: pass")
    return seen["env"]["TORCH_EXTENSIONS_DIR"]


def test_two_different_candidates_never_share_a_build_directory(reward_src, monkeypatch):
    a = _ext_dir_for(reward_src, monkeypatch, "class ModelNew:  # variant A\n    pass")
    b = _ext_dir_for(reward_src, monkeypatch, "class ModelNew:  # variant B\n    pass")
    assert a != b, (
        "both candidates compile in the same directory; whichever .so wins the race is the one "
        "both get graded on")


def test_identical_candidates_still_share_the_cache(reward_src, monkeypatch):
    """The one legitimate cache hit — greedy decoding does produce duplicates — is kept."""
    src = "class ModelNew:\n    pass"
    assert _ext_dir_for(reward_src, monkeypatch, src) == _ext_dir_for(reward_src, monkeypatch, src)


def test_the_directory_is_still_scoped_to_the_reward_gpu(reward_src, monkeypatch):
    src = "class ModelNew:\n    pass"
    assert _ext_dir_for(reward_src, monkeypatch, src, visible="3") != \
           _ext_dir_for(reward_src, monkeypatch, src, visible="5")


def test_a_timeout_removes_the_build_directory_it_abandoned(reward_src, monkeypatch, tmp_path):
    """SIGKILL cannot be caught, so torch's lock file is left behind; nothing else removes it."""
    import subprocess as _sp
    root = tmp_path / "extroot"
    monkeypatch.setenv("CUDAFORGE_EXT_CACHE_ROOT", str(root))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    created = {}

    def fake_run(*a, **k):
        d = (k.get("env") or {})["TORCH_EXTENSIONS_DIR"]
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "lock"), "w").close()      # what torch leaves behind
        created["dir"] = d
        raise _sp.TimeoutExpired(cmd="runner", timeout=1)

    monkeypatch.setattr(reward_src, "_abort_if_node_is_wedging", lambda: None)
    monkeypatch.setattr(reward_src.subprocess, "run", fake_run)
    correctness, speedup = reward_src.bench("class ModelNew:\n    pass", "class Model: pass")
    assert (correctness, speedup) == (0, 0.0)
    assert not os.path.exists(created["dir"]), (
        "the abandoned build directory survived; the identical-source retry would wait on a "
        "lock nobody will drop")


# ---- the stop-token set is a property of the checkpoint, not of the data ----------------------
#
# generation_config decides where a generation stops. This checkpoint ships an "eosfix" that
# removes <|endoftext|> from the stop set, keeping only <|im_end|>. That is correct for the
# current corpus — 1177 scored candidates carried no control token and nothing hit the length cap
# — but whether a model emits a given terminator depends on the prompt format, so the same set
# can be wrong for the next dataset, and the failure is silent: the compiler is simply handed
# whatever the model wrote after its own terminator.

@pytest.mark.parametrize("tok", ["<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|eot_id|>"])
def test_a_control_token_truncates_the_candidate(reward_src, tok, capsys):
    raw = f"import torch\ndef f():\n    return 1\n{tok}\nGARBAGE THAT IS NOT CODE"
    out = reward_src._extract_python_code(raw)
    assert "GARBAGE" not in out, (
        f"text after {tok} reached the compiler as part of the kernel")
    assert "def f()" in out, "the real candidate was lost along with the trailing text"


def test_the_mismatch_is_reported_not_swallowed(reward_src, capsys):
    capsys.readouterr()
    reward_src._extract_python_code("import torch<|endoftext|>trailing")
    assert "control token" in capsys.readouterr().out, (
        "a stop-token set that does not match the data is invisible unless it says so")


def test_ordinary_generations_are_untouched(reward_src, capsys):
    """The guard must not alter the normal path — every candidate goes through it."""
    raw = "```python\nimport torch\n\nclass ModelNew(torch.nn.Module):\n    pass\n```"
    capsys.readouterr()
    out = reward_src._extract_python_code(raw)
    assert out == "import torch\n\nclass ModelNew(torch.nn.Module):\n    pass"
    assert "control token" not in capsys.readouterr().out


def test_the_first_control_token_wins(reward_src):
    """Whichever terminator appears first is where the generation actually ended."""
    out = reward_src._extract_python_code("a=1\n<|im_end|>\nb=2\n<|endoftext|>\nc=3")
    assert "b=2" not in out and "c=3" not in out
