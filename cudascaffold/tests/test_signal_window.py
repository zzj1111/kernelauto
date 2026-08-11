"""The Teacher's per-category signal must come from training rollouts and nothing else.

train_adapter snapshots the worker-log offsets, runs the training subprocess, and reads
everything the reward printed in between. That window is only the training rollouts if the
training subprocess scores nothing else — and for the whole run up to 2026-08-10 it did score
something else. verl validates when

    test_freq > 0 and (is_last_step or global_steps % test_freq == 0)     ray_trainer.py:1377

so `test_freq=999999` defeated the modulo and nothing else: every pass ends on its last step,
so every cycle ran a full held-out validation before returning. Cycle 1 of smoke4 printed 280
reward lines for 96 training rollouts — 16 task_names seen 6 times each (8 prompts x 2 steps
x rollout.n=6) plus 180 held-out problems seen once. Two thirds of what the Teacher read to
decide which category was weak was the held-out set the A/B gate exists to keep it away from.

Guarding this needs both halves: the override we send, and the condition in verl that reads it.
A test that only pinned `test_freq=-1` would have passed just as happily on 999999.
"""
from __future__ import annotations

import ast
import inspect
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
RAY_TRAINER = os.path.join(REPO, "verl", "trainer", "ppo", "ray_trainer.py")


def _default(param):
    from cudascaffold import adapters as A
    return inspect.signature(A._train_cmd).parameters[param].default


def test_a_training_pass_asks_for_no_validation_at_all():
    assert _default("test_freq") <= 0, (
        f"_train_cmd defaults to test_freq={_default('test_freq')!r}. A positive value validates "
        "on the last step no matter how large it is, and those rows land in the Teacher's window.")
    assert _default("val_before") is False, \
        "a training pass that validates first spends the whole held-out set before step 1"


def test_verl_still_gates_validation_on_test_freq_being_positive():
    """The reason -1 works. If verl ever drops the `test_freq > 0` conjunct, -1 stops disabling
    anything and the contamination comes back silently — this is the assertion that notices."""
    if not os.path.exists(RAY_TRAINER):
        pytest.skip("vendored verl not present")
    src = open(RAY_TRAINER, encoding="utf-8").read()
    guard = re.search(r"# validate\n\s*if \((.*?)\):", src, re.S)
    assert guard, "the validation guard in ray_trainer.py no longer looks the way this pins it"
    cond = " ".join(guard.group(1).split())
    assert "self.config.trainer.test_freq > 0" in cond, (
        f"validation is no longer gated on test_freq > 0 (guard is {cond!r}); a non-positive "
        "test_freq no longer disables it and _train_cmd's default is now inert")
    assert "is_last_step" in cond, \
        "the last-step trigger is gone; the comment on _train_cmd's default is out of date"


def test_the_eval_and_ab_passes_still_validate():
    """The same default must not switch off the passes whose entire job is to validate. They go
    through val_only, which ray_trainer.py:1128 gates on val_before_train, not on test_freq —
    but they pass test_freq=1 explicitly and that is what this pins."""
    from cudascaffold import adapters as A
    src = open(A.__file__, encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_train_cmd"]
    assert calls, "no _train_cmd call sites found"
    kw = [{k.arg: k.value for k in c.keywords} for c in calls]
    validating = [k for k in kw if "test_freq" in k]
    assert validating, "no call site asks for validation any more — eval and the A/B gate are blind"
    for k in validating:
        got = ast.literal_eval(k["test_freq"])
        assert got > 0, f"a call site passes test_freq={got!r}, which validates nothing"


def test_the_signal_window_closes_before_the_checkpoint_is_evaluated():
    """Ordering, in case the offsets ever move: the snapshot is taken before the training
    subprocess and read after it, with eval_fn running later in the cycle. If signals were
    computed after eval instead, disabling verl's internal validation would fix nothing."""
    from cudascaffold import adapters as A
    src = open(A.__file__, encoding="utf-8").read()
    body = src[src.index("def train_adapter("):]
    body = body[:body.index("\ndef ")]
    snap = body.index("worker_log_offsets(")
    run = body.index("_run(_train_cmd(")
    read = body.index("signals_from_training(")
    assert snap < run < read, (
        "train_adapter no longer snapshots offsets before the training subprocess and reads "
        "them after it; the Teacher's signal window is not the training pass any more")
