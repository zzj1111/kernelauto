"""The A/B's unit of independence is the problem, not the row.

The measurement buys `reps` copies of each held-out problem in one pass, and copies of a
problem mostly meet the same fate. Counting rows as independent samples understates the noise
by up to sqrt(1+(reps-1)*rho) — the step-40 same-condition pass shows it directly: its 0.074
gap is 3.3 row-level SEs, which independent draws essentially never produce, but 1.6 SEs once
copies are treated as one cluster. All three arms already run the same problems, so pairing on
the problem is free and removes problem difficulty, the largest term, from the comparison.
"""
import os

import pytest

from cudascaffold import gates as G

TASKS = ["elementwise"]


def _measure(cur_rates, cand_rates, reps=6, cat="elementwise"):
    """Build a measurement whose row-level aggregate and problem table agree."""
    per = {"current": {}, "candidate": {}, "bare": {}}
    cats = {}
    for i, (c, k) in enumerate(zip(cur_rates, cand_rates)):
        key = f"row{i}"
        per["current"][key] = (round(c * reps), reps)
        per["candidate"][key] = (round(k * reps), reps)
        per["bare"][key] = (round(c * reps), reps)
        cats[key] = cat
    agg = lambda rates: {cat: (sum(rates) / len(rates), len(rates) * reps)}
    return {"current": agg(cur_rates), "candidate": agg(cand_rates), "bare": agg(cur_rates),
            "per_problem": per, "pair_category": cats, "reps": reps}


def test_the_problem_is_the_unit_not_the_row(monkeypatch):
    """Same numbers, 6 copies each: the reported n must be problems, not rows."""
    monkeypatch.setattr(G, "NOISE_K", 0.0)
    m = _measure([0.0] * 5 + [1.0] * 5, [0.0] * 4 + [1.0] * 6, reps=6)
    out = G.ab_gate(m, TASKS)
    assert out["unit"] == "problem"
    assert out["n"] == 10, f"reported n={out['n']} — rows counted as samples again"


def test_repeats_do_not_shrink_the_error_bar(monkeypatch):
    """Ten problems measured 2x and 12x must give the SAME standard error.

    This is the whole bug: more copies of the same problems buy resolution on each problem, not
    more problems, so they must not make the gate more confident about the difference.
    """
    monkeypatch.setattr(G, "NOISE_K", 2.0)
    cur = [0.0] * 5 + [1.0] * 5
    cand = [0.0] * 4 + [1.0] * 6
    se2 = G.ab_gate(_measure(cur, cand, reps=2), TASKS)["paired_se"]
    se12 = G.ab_gate(_measure(cur, cand, reps=12), TASKS)["paired_se"]
    assert se2 == pytest.approx(se12, abs=1e-9), (
        f"SE shrank from {se2} to {se12} purely by taking more copies of the same problems")


def test_pairing_ignores_problem_difficulty(monkeypatch):
    """A uniform +1/6 improvement on wildly different problems is detected as exactly that.

    Unpaired, these arms differ by 0.167 against a spread of rates from 0 to 1, and the
    difficulty variance swamps it. Paired, every difference is identical, so the SE is zero and
    the effect is unmissable.
    """
    monkeypatch.setattr(G, "NOISE_K", 2.0)
    cur = [0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
    cand = [c + 1 / 6 for c in cur]
    out = G.ab_gate(_measure(cur, cand, reps=6), TASKS)
    assert out["paired_diff"] == pytest.approx(1 / 6, abs=1e-4)   # reported rounded, for journals
    assert out["paired_se"] == pytest.approx(0.0, abs=1e-9)
    assert out["accept"] is True


def test_a_same_condition_measurement_is_not_accepted(monkeypatch):
    """current and candidate identical per problem -> difference exactly 0, never accepted."""
    monkeypatch.setattr(G, "NOISE_K", 0.0)
    rates = [0.0, 0.5, 1.0, 0.5, 0.0, 1.0]
    out = G.ab_gate(_measure(rates, rates, reps=6), TASKS)
    assert out["paired_diff"] == 0.0
    assert out["accept"] is False


def test_noise_that_cancels_across_problems_is_rejected_with_a_margin(monkeypatch):
    """Half the problems better, half worse, mean ~0 — the margin must hold it back."""
    monkeypatch.setattr(G, "NOISE_K", 2.0)
    cur = [0.5] * 8
    cand = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    out = G.ab_gate(_measure(cur, cand, reps=6), TASKS)
    assert out["accept"] is False, out["reason"]


def test_it_falls_back_when_the_problem_table_is_absent(monkeypatch):
    """An older measurement (or one that resolved no shared problems) must still decide."""
    monkeypatch.setattr(G, "NOISE_K", 0.0)
    legacy = {"current": {"elementwise": (0.2, 120)},
              "candidate": {"elementwise": (0.4, 120)},
              "bare": {"elementwise": (0.2, 120)}}
    out = G.ab_gate(legacy, TASKS)
    assert out["accept"] is True and "unit" not in out


def test_one_shared_problem_is_not_enough_to_compute_a_spread(monkeypatch):
    """With a single problem there is no spread to estimate, so fall back rather than invent one."""
    monkeypatch.setattr(G, "NOISE_K", 2.0)
    m = _measure([0.0], [1.0], reps=6)
    out = G.ab_gate(m, TASKS)
    assert "unit" not in out, "a one-problem SE is not a measurement"


def test_only_touched_categories_enter_the_comparison(monkeypatch):
    monkeypatch.setattr(G, "NOISE_K", 0.0)
    m = _measure([0.0] * 4, [1.0] * 4, reps=6)
    m["per_problem"]["current"]["rowX"] = (6, 6)
    m["per_problem"]["candidate"]["rowX"] = (0, 6)
    m["pair_category"]["rowX"] = "matmul"          # not touched
    out = G.ab_gate(m, TASKS)
    assert out["n"] == 4, "an untouched category leaked into the paired comparison"
    assert out["paired_diff"] == pytest.approx(1.0)
