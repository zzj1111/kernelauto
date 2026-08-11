"""verl scores rows that do not exist, and the parse must not count them.

ray_trainer.py:596 pads every validation batch to a multiple of rollout.agent.num_workers by
REPEATING rows. The repeats go through generation and reward — their lines land in the worker
logs — and only their tensors are dropped by unpad. So verl's own metrics are clean while
anything scraped from the logs is inflated: the first smoke A/B printed 600 lines for 540 rows
(36-row waves, 8 workers, 4 phantoms per wave x 15), and the gate reported n=199/200/201 for
180-row arms. 47 of the 60 phantoms reprinted identical results; 13 re-benched the same greedy
code and got different timings, so the phantom is not even a harmless copy of its row.

The defence is a row id stamped into task_name when the measurement parquet is built
("name#r17@@arm"): the same id twice is one row scored twice and collapses to one sample;
legitimate ab_repeats_max copies are distinct rows with distinct ids and never collapse.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from cudascaffold import adapters as A


def _fake_ray_tmp(tmp_path, lines):
    logs = tmp_path / "ray" / "session_x" / "logs"
    logs.mkdir(parents=True)
    (logs / "worker-aaa-01000000-1.out").write_text("\n".join(lines) + "\n")
    return {A._ROOT_KEY: str(tmp_path)}


def _line(task, correct=1, speedup=2.0, cat="conv@@candidate"):
    return (f"correctness: {correct}, speedup: {speedup}, data_source:TritonKernel, "
            f"task_name:{task}, level:0, category:{cat}, reward:1.0")


def test_a_padded_repeat_counts_as_one_row(tmp_path):
    off = _fake_ray_tmp(tmp_path, [
        _line("t1#r0@@candidate"),
        _line("t1#r0@@candidate"),          # verl's padding scored the same row again
        _line("t2#r1@@candidate", correct=0, speedup=0.0),
    ])
    got = A.per_category_from_log(off)["conv@@candidate"]
    assert got["n"] == 2, f"phantom inflated n to {got['n']}"
    assert got["correct_rate"] == 0.5


def test_disagreeing_phantom_observations_split_the_difference(tmp_path):
    """13 of 60 real phantoms re-benched the same code to a different result. Neither
    observation is more real than the other, so the row carries their mean."""
    off = _fake_ray_tmp(tmp_path, [
        _line("t1#r0@@candidate", correct=1, speedup=2.0),
        _line("t1#r0@@candidate", correct=0, speedup=0.0),
    ])
    got = A.per_category_from_log(off)["conv@@candidate"]
    assert got["n"] == 1
    assert got["correct_rate"] == 0.5


def test_legitimate_copies_stay_separate_samples(tmp_path):
    """ab_repeats_max deliberately scores a problem several times at different batch positions.
    Those are distinct rows with distinct ids and must all count — a collapse keyed on the bare
    task name would undo the repeat design."""
    off = _fake_ray_tmp(tmp_path, [
        _line("t1#r0@@candidate"),
        _line("t1#r3@@candidate", correct=0, speedup=0.0),  # same problem, second copy
    ])
    got = A.per_category_from_log(off)["conv@@candidate"]
    assert got["n"] == 2
    assert got["correct_rate"] == 0.5


def test_untagged_lines_keep_legacy_counting(tmp_path):
    """Training and eval windows print bare task names (their multiplicity is real: 6 rollouts
    of one prompt are 6 samples). Only row-tagged lines may collapse."""
    off = _fake_ray_tmp(tmp_path, [
        _line("plain_task", cat="conv"),
        _line("plain_task", cat="conv"),
    ])
    assert A.per_category_from_log(off)["conv"]["n"] == 2


def test_the_built_parquet_tags_every_named_row_uniquely():
    """Source-level check on measure_ab_adapter: the tag must be row-unique (an index, not just
    the arm), and placed BEFORE the arm suffix so rsplit on _ARM_SEP still yields the arm."""
    src = open(A.__file__, encoding="utf-8").read()
    body = src[src.index("def measure_ab_adapter("):src.index("def persist_scaffold(")]
    assert "_ROW_MARK" in body, "the row id is no longer stamped into task_name"
    stamp = next(ln for ln in body.splitlines() if "_ROW_MARK" in ln and "task_name" in ln
                 and "=" in ln and "#" != ln.strip()[0])
    assert stamp.index("_ROW_MARK") < stamp.index("_ARM_SEP"), (
        "row id must precede the arm suffix, or rsplit(_ARM_SEP) returns 'arm#r17' and every "
        "arm bucket goes empty")
