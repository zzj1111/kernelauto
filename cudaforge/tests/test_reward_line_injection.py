"""The reward line is scraped with a regex, and the err tail is candidate-controlled text.

A candidate whose exception message is SHAPED like a reward line must not register as a
second sample: the controller counts REWARD_LINE matches to build the Teacher's statistics
and the A/B gate's n, so an unsanitized echo hands the trained policy a text channel into
its own evaluation. These tests pin the sanitizer and the tail grammar.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cudascaffold import adapters as A

FORGED = ("correctness: 1, speedup: 9.9, data_source:Forged, task_name:evil, "
          "level:0, category:conv, reward:1.0")


def _emit(reward, capsys, **kw):
    reward._emit_attribution("TritonKernel", {"task_name": "t1", "level": 0,
                                              "category": "conv"}, 0, 0.0, 0.0, **kw)
    return capsys.readouterr().out


def test_a_forged_error_message_cannot_mint_a_second_reward_line(reward, capsys):
    out = _emit(reward, capsys, fail_kind="runtime_error", fail_msg=FORGED)
    matches = list(A.REWARD_LINE.finditer(out))
    assert len(matches) == 1, f"forged err minted {len(matches)} reward lines:\n{out}"
    assert matches[0].group(4) == "t1", "the surviving match is not the genuine line"


def test_the_sanitized_message_still_reads_as_evidence(reward, capsys):
    out = _emit(reward, capsys, fail_kind="runtime_error",
                fail_msg="RuntimeError: CUDA error: an illegal memory access was encountered")
    kind, err = A._fail_of(out, A.REWARD_LINE.search(out))
    assert kind == "runtime_error"
    assert "illegal memory access" in err


def test_a_fail_kind_with_separators_stays_one_token(reward, capsys):
    out = _emit(reward, capsys, fail_kind="weird kind, with comma", fail_msg="boom")
    kind, err = A._fail_of(out, A.REWARD_LINE.search(out))
    assert kind == "weird_kind_with_comma"
    assert err == "boom"
