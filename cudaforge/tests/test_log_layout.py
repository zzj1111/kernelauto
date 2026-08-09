"""The forensic log must not grow one file per rollout.

The flat `bench_<ts>_pid<n>.jsonl` layout left ~30 000 files in one directory on this
filesystem, where listing it takes over two minutes, and nothing prunes it — it grows for the
life of the project. Sharding by day bounds the directory and makes old logs deletable as a
unit; one file per process keeps concurrent runners off each other's appends.

Read from source rather than imported: the module pulls in torch and is loaded by verl through
a path, not as a package.
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def test_bench_logs_are_sharded_by_day_and_process():
    src = _src("reward_bench_rubric.py")
    assert 'f"bench_{ts}_pid{pid}.jsonl"' not in src, \
        "back to one file per rollout — that is the layout that made the directory unlistable"
    m = re.search(r'log_path = os\.path\.join\((.+?)\)\n', src, re.S)
    assert m, "could not find the bench log path construction"
    expr = m.group(1)
    assert "ts[:8]" in expr, "the day shard is gone; the directory is unbounded again"
    assert "pid" in expr, "concurrent runners would append to one file"


def test_the_record_still_carries_its_own_timestamp():
    """The per-rollout timestamp moved out of the filename, so it has to be in the record."""
    src = _src("reward_bench_rubric.py")
    assert 'record.setdefault("ts", ts)' in src, \
        "timestamps were only in the filename and the filename no longer has them"
    assert 'record.setdefault("pid", pid)' in src


def test_the_rubric_log_stays_a_single_appended_file():
    """It was already right — one file, appended. Pin it so it is not 'fixed' into per-call
    files by someone copying the old bench pattern."""
    src = _src("reward_bench_rubric.py")
    assert '"rubric_logs", "rubric_judge.jsonl"' in src
