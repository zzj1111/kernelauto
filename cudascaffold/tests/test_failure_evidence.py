"""The Teacher's failure evidence, end to end: the reward names WHY a candidate scored 0,
and the signal chain carries that name (plus one verbatim error) into the failures list.

Before this, the kernel Teacher saw only THAT an instance failed — correct_rate and n — while
the ALFWorld Teacher read full failed trajectories. One attempt is one answer here, so the
equivalent evidence is the runner's verdict (compile_error / correctness_error / timeout / ...)
and the error text itself; a Teacher told only "0/6" cannot distinguish an API-habit problem
from an algorithmic one, and the text it writes differs completely between those.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from cudascaffold import adapters as A


def _line(task, correct=0, speedup=0.0, cat="conv", tail=""):
    return (f"correctness: {correct}, speedup: {speedup}, data_source:TritonKernel, "
            f"task_name:{task}, level:0, category:{cat}, reward:0.0{tail}")


def _fake_tree(tmp_path, lines):
    logs = tmp_path / "ray" / "session_x" / "logs"
    logs.mkdir(parents=True)
    (logs / "worker-aaa-01000000-1.out").write_text("\n".join(lines) + "\n")
    return {A._ROOT_KEY: str(tmp_path)}


def test_the_fail_tail_parses_and_leaves_reward_line_intact(tmp_path):
    off = _fake_tree(tmp_path, [
        _line("t1", tail=", fail:compile_error, err:error: identifier \"blockIdx\" is undefined"),
    ])
    txt = A._read_since(off)
    m = A.REWARD_LINE.search(txt)
    assert m and m.group(4) == "t1", "the tail broke the base regex"
    kind, err = A._fail_of(txt, m)
    assert kind == "compile_error"
    assert "blockIdx" in err


def test_a_line_without_the_tail_reports_no_kind(tmp_path):
    off = _fake_tree(tmp_path, [_line("t1")])
    txt = A._read_since(off)
    kind, err = A._fail_of(txt, A.REWARD_LINE.search(txt))
    assert kind is None and err is None, "invented a failure kind for a pre-field log line"


def _six(task, kinds, cat="conv"):
    """Six rollouts of one task, all failed, with the given kinds cycling."""
    out = []
    for i in range(6):
        k = kinds[i % len(kinds)]
        out.append(_line(task, cat=cat, tail=f", fail:{k}, err:msg for {k} {i}"))
    return out


def test_failures_carry_kind_counts_and_one_sample_error(tmp_path):
    off = _fake_tree(tmp_path, _six("bad_task", ["compile_error", "correctness_error"]))
    sig = A.signals_from_training(off, fired={"bad_task": False}, rollout_n=6)
    (row,) = [f for f in sig["failures"] if f["instance"] == "bad_task"]
    assert row["fail_kinds"] == {"compile_error": 3, "correctness_error": 3}
    assert row["sample_error"].startswith("msg for")


def test_an_injected_groups_failure_evidence_is_not_called_bare(tmp_path):
    off = _fake_tree(tmp_path, _six("inj_task", ["compile_error"]))
    sig = A.signals_from_training(off, fired={"inj_task": True}, rollout_n=6)
    assert not any(f["instance"] == "inj_task" for f in sig["failures"]), (
        "a scaffolded group's failure appeared in the bare-failures list; the Teacher would "
        "read its own text's failure as the base policy's")


def test_the_gap_reports_speedup_medians_only_at_usable_n(tmp_path):
    lines = []
    for t, sp in (("a", 2.0), ("b", 3.0), ("c", 4.0)):
        for i in range(6):
            lines.append(_line(f"ok_{t}", correct=1, speedup=sp, cat="conv"))
    lines += _six("bad", ["compile_error"], cat="loss")
    off = _fake_tree(tmp_path, lines)
    sig = A.signals_from_training(
        off, fired={"ok_a": False, "ok_b": False, "ok_c": False, "bad": False}, rollout_n=6)
    conv = sig["per_task_gap"]["conv"]
    assert conv["speedup_median_bare"] == 3.0
    loss = sig["per_task_gap"]["loss"]
    assert loss["speedup_median_bare"] is None, "a median over zero correct rollouts"


def test_the_prompt_describes_what_the_data_now_carries():
    from cudascaffold import observation as O
    for needle in ("fail_kinds", "sample_error", "speedup_when_correct"):
        assert needle in O.TRACE_SINGLE_TURN, f"prompt does not explain {needle}"
    assert "per_task_n" in O._SIGNAL_MEANINGS, \
        "prompt does not explain the held-out per-category breakdown"
