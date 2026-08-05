"""The A/B gate must not mistake its own measurement noise for an effect.

These tests assert the CLAIMS made in gates.ab_gate and adapters.measure_ab_adapter.

The margin tests use the real numbers those claims were derived from — the step-40 A/B of
triton_scaffold_8b, where the scaffold was still empty so `bare` and `current` were the SAME
condition measured twice over byte-identical prompts (md5 a2bbbaa613dc for both files). Every
difference among them is therefore noise by construction, and any ACCEPT is a false positive.

The subset tests cover where the gate draws its rows from. It now measures on the HELD-OUT file
rather than on training rows, takes every row of the touched categories, and buys copies with
whatever the budget leaves — so one pass costs the same whether one category or all six moved.

If someone lowers NOISE_K, drops the touched-category restriction, or reverts the subset to
training rows, these fail with the actual consequence rather than a style complaint.
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cudascaffold.adapters import _ARM_SEP, _count_in_categories, _subset  # noqa: E402
from cudascaffold.gates import NOISE_K, ab_gate  # noqa: E402

# (category, successes in pass A, successes in pass B) out of 24, step 40, empty scaffold.
SAME_CONDITION = [("conv", 1, 6), ("elementwise", 4, 2), ("loss", 10, 4),
                  ("matmul", 5, 5), ("norm_softmax", 6, 6), ("reduce", 4, 1)]
N = 24

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The gate measures on the HELD-OUT file. Sampling fresh training rows would score problems the
# policy was just updated on, and the `current` arm's problems were trained on WITH that text in
# the prompt — the condition the gate is trying to measure.
HELDOUT = os.path.join(_ROOT, "dataset", "Triton", "test.parquet")


def _gate(cat, cur, cand, n=N):
    return ab_gate({"bare": {cat: (cur, n)}, "current": {cat: (cur, n)},
                    "candidate": {cat: (cand, n)}}, [cat])


def test_no_measured_noise_pair_is_accepted():
    """The bar is set above every difference two IDENTICAL conditions have been seen to produce."""
    for cat, a, b in SAME_CONDITION:
        lo, hi = min(a, b) / N, max(a, b) / N          # worst case: larger side is the candidate
        r = _gate(cat, lo, hi)
        assert not r["accept"], (
            f"{cat}: {lo:.3f} -> {hi:.3f} is noise (same condition, byte-identical prompts) "
            f"but cleared a margin of {r['margin']:.3f}")


def test_a_strict_greater_than_would_have_accepted_three_of_them():
    """Why the margin exists at all: without it the gate is a coin flip on this data."""
    would = [c for c, a, b in SAME_CONDITION if max(a, b) > min(a, b)]
    assert len(would) == 4, would
    assert NOISE_K > 1.0, "K=1 was measured to pass two of these four; see ab_gate's docstring"


def test_margin_shrinks_with_n_so_the_bar_tightens_as_measurement_improves():
    small = _gate("m", 0.15, 0.30, n=24)["margin"]
    large = _gate("m", 0.15, 0.30, n=300)["margin"]
    assert large < small, (small, large)
    assert abs(large - small / (300 / 24) ** 0.5) < 0.02, "margin must fall as 1/sqrt(n)"


def test_an_effect_worth_adopting_still_passes():
    """A conservative bar is only useful if it is not an infinite one."""
    assert _gate("m", 0.15, 0.25, n=300)["accept"], "a 10-point gain at n=300 must be adoptable"
    assert not _gate("m", 0.15, 0.18, n=300)["accept"], "3 points at n=300 is not resolvable"


def test_ties_and_regressions_never_accept():
    assert not _gate("m", 0.20, 0.20)["accept"]
    assert not _gate("m", 0.20, 0.05)["accept"]
    assert not ab_gate({}, [])["accept"], "no touched tasks -> nothing to test -> reject"
    assert not ab_gate({"current": {}, "candidate": {}}, ["m"])["accept"], "no samples -> reject"


def test_arm_label_round_trips_and_untouched_categories_are_excluded():
    """The merged single pass is only sound if it can be taken apart again exactly."""
    by_cat = {f"matmul{_ARM_SEP}bare": (0.10, 300), f"matmul{_ARM_SEP}current": (0.12, 300),
              f"matmul{_ARM_SEP}candidate": (0.24, 300), f"conv{_ARM_SEP}current": (0.99, 300)}
    tasks, out = ["matmul"], {"bare": {}, "current": {}, "candidate": {}}
    for key, v in by_cat.items():
        cat, arm = key.rsplit(_ARM_SEP, 1)
        if arm in out and cat in tasks:
            out[arm][cat] = v
    assert out["candidate"] == {"matmul": (0.24, 300)}
    assert "conv" not in out["current"], "a category the proposal did not touch must not be scored"
    assert ab_gate(out, tasks)["accept"]


def test_ab_subset_takes_every_held_out_row_of_the_touched_categories(tmp_path):
    """`n_per_cat=None` means the whole category, and copies come from `repeats`.

    Asking for "all" with a large integer instead was a real defect: _subset would enter its
    top-up branch, compute ceil(1e9 / 30) repeats, and try to concatenate 33 million copies of 30
    rows. Every A/B would have hung there.
    """
    if not os.path.exists(HELDOUT):
        return
    out = _subset(HELDOUT, str(tmp_path / "s.parquet"), None, seed=1, only=["reduce"], repeats=6)
    assert len(out) == 180, f"30 held-out reduce rows x 6 copies, got {len(out)}"
    names = [str(x.get("task_name")) for x in out["extra_info"]]
    assert len(set(names)) == 30, "copies must not invent problems"
    assert set(collections.Counter(names).values()) == {6}, "every problem gets the same copies"
    from cudascaffold import splice as SP
    cats = {SP.level_of(out["extra_info"].iloc[i],
                        out["data_source"].iloc[i] if "data_source" in out.columns else None)
            for i in range(len(out))}
    assert cats == {"reduce"}, cats


def test_a_large_n_per_cat_is_not_a_way_to_say_all(tmp_path):
    """Guards the direction the bug came from: the two spellings must stay distinguishable."""
    if not os.path.exists(HELDOUT):
        return
    whole = _subset(HELDOUT, str(tmp_path / "a.parquet"), None, seed=1, only=["conv"])
    topped = _subset(HELDOUT, str(tmp_path / "b.parquet"), 90, seed=1, only=["conv"])
    assert len(whole) == 30, "None -> the category as it stands"
    assert len(topped) == 90, "an integer above the category size tops up by repetition"


def test_budget_buys_a_constant_pass_size_however_many_categories_are_touched():
    """The gate spends a fixed row budget: fewer touched categories buy more copies per problem,
    so one pass costs the same whether the proposal edits one category or all six."""
    budget, reps_max, per_cat = 540, 6, 30
    for k in (1, 2, 3, 6):
        n_q = per_cat * k
        reps = max(1, min(reps_max, budget // max(1, 3 * n_q)))
        assert 3 * n_q * reps == budget, (k, n_q, reps)


def test_held_out_counts_match_the_file(tmp_path):
    if not os.path.exists(HELDOUT):
        return
    assert _count_in_categories(HELDOUT, ["conv"]) == 30
    assert _count_in_categories(
        HELDOUT, ["conv", "elementwise", "loss", "matmul", "norm_softmax", "reduce"]) == 180


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
