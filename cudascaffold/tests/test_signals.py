"""Regression tests for the signals the Teacher actually reads.

These cover the defect found at cycle 10 (2026-08-02): `per_task_gap` was keyed by verl's
`data_source` (`CudaForge` / `CudaForgeImprovement`) instead of by the scaffold's categories,
so it shared no key with anything the Teacher could edit, and `all_fail_groups` — built by
matching those same data_source strings against `improve_l{level}` / `"scratch"` — was silently
always empty. Ten consecutive cycles produced no scaffold.

Run: /dev/shm/verl_env/bin/python -m pytest cudascaffold/tests/test_signals.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/mnt/data1/zha00175/StitchCUDA")

from cudascaffold.adapters import cat_of_level, per_category_from_log  # noqa: E402

SCAFFOLD_SCOPES = {"scratch", "improve_l1", "improve_l2", "improve_l3"}


def _line(correct, speedup, ds, task, level):
    return (f"correctness: {correct}, speedup: {speedup}, data_source:{ds}, "
            f"task_name:{task}, level:{level}\n")


def _worker_logs(tmp_path, text, name="worker-aaa.out"):
    """Lay out a fake Ray tree and return offsets that select everything written after."""
    d = tmp_path / "ray" / "session_2026-08-02_00-00-00_0_1" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text("preamble that must not be counted\n")
    offsets = {str(f): f.stat().st_size}
    with open(f, "a") as fh:
        fh.write(text)
    return offsets


def test_cat_of_level_maps_reward_field_to_scaffold_scope():
    assert cat_of_level("None") == "scratch"       # scratch rows carry no level
    assert cat_of_level("") == "scratch"
    assert cat_of_level("nan") == "scratch"
    assert cat_of_level("1.0") == "improve_l1"
    assert cat_of_level("2.0") == "improve_l2"
    assert cat_of_level("3") == "improve_l3"
    assert cat_of_level("junk") is None            # unknown -> dropped, never mislabelled


def test_keys_are_scaffold_scopes_not_data_sources(tmp_path):
    """The regression itself: two data_sources must not become the breakdown."""
    text = (_line(1, 1.5, "CudaForge", "None", "None")
            + _line(0, 0.0, "CudaForge", "None", "None")
            + _line(1, 2.0, "CudaForgeImprovement", "12_Foo", "1.0")
            + _line(0, 0.0, "CudaForgeImprovement", "13_Bar", "2.0")
            + _line(0, 0.0, "CudaForgeImprovement", "14_Baz", "3.0"))
    out = per_category_from_log(_worker_logs(tmp_path, text))
    assert set(out) <= SCAFFOLD_SCOPES, f"keys must be scaffold scopes, got {sorted(out)}"
    assert "CudaForge" not in out and "CudaForgeImprovement" not in out
    assert set(out) == {"scratch", "improve_l1", "improve_l2", "improve_l3"}


def test_offsets_exclude_earlier_content(tmp_path):
    """Only the window after the snapshot counts, or a pass inherits the previous pass's rows."""
    out = per_category_from_log(_worker_logs(tmp_path, _line(1, 1.0, "CudaForge", "None", "None")))
    assert out["scratch"]["n"] == 1


def test_correct_rate_and_speedup_are_reported_separately(tmp_path):
    """correct_rate drives the gap; speedup stays visible so correct-but-not-faster shows up."""
    text = (_line(1, 0.99, "CudaForge", "None", "None")
            + _line(1, 1.01, "CudaForge", "None", "None")
            + _line(1, 0.98, "CudaForge", "None", "None")
            + _line(0, 0.0, "CudaForge", "None", "None"))
    s = per_category_from_log(_worker_logs(tmp_path, text))["scratch"]
    assert s["n"] == 4 and s["n_correct"] == 3
    assert s["correct_rate"] == 0.75
    assert s["speedup_median"] == 0.99          # median of the CORRECT ones only
    assert s["n_faster_than_torch"] == 1        # only 1.01 beats the reference


def test_shaped_reward_outlier_cannot_swing_the_signal(tmp_path):
    """Why correct_rate, not mean reward.

    At n=8 on the shaped reward, one kernel hitting the 5.0 speedup clip moved a pass mean by
    ~0.6 — that is how two byte-identical prompt sets produced a measured gap of -0.72. The same
    outlier must not move correct_rate by more than 1/n.
    """
    base = [_line(1, 1.0, "CudaForge", "None", "None") for _ in range(7)]
    without = per_category_from_log(_worker_logs(tmp_path, "".join(base + [
        _line(1, 1.0, "CudaForge", "None", "None")]), name="w1.out"))
    with_spike = per_category_from_log(_worker_logs(tmp_path, "".join(base + [
        _line(1, 92.6, "CudaForge", "None", "None")]), name="w2.out"))
    assert without["scratch"]["correct_rate"] == with_spike["scratch"]["correct_rate"] == 1.0
    assert with_spike["scratch"]["speedup_median"] == 1.0   # outlier visible but not dominant


def test_all_fail_is_derivable_and_nonempty_when_a_category_is_stuck(tmp_path):
    """all_fail_groups used to be unreachable; a fully-failing category must now surface."""
    text = "".join(_line(0, 0.0, "CudaForgeImprovement", f"{i}_Task", "3.0") for i in range(6))
    out = per_category_from_log(_worker_logs(tmp_path, text))
    cat = out["improve_l3"]
    assert cat["n"] == 6 and cat["n_correct"] == 0
    assert cat["n"] - cat["n_correct"] == 6, "zero-reward count must equal the whole category"
    assert cat["speedup_median"] is None


def test_empty_window_is_empty_not_crash(tmp_path):
    assert per_category_from_log(_worker_logs(tmp_path, "")) == {}
    assert per_category_from_log({}) == {}


if __name__ == "__main__":
    import pathlib
    import tempfile

    fails = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print(f"  PASS {name}")
        except AssertionError as e:
            fails.append(name)
            print(f"  FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
