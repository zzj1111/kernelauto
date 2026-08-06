"""The two harness gates — pure functions on measurements. No I/O, no GPU, no API.

These are the ONLY places the harness (as opposed to the Teacher) makes a decision:

  ab_gate      -- accept/reject a proposed TEXT change, judged by a frozen-policy A/B
                  on train (candidate scaffold vs current scaffold, same weights, same
                  games). The clean way to attribute a scaffold edit: anything measured
                  AFTER a training step cannot be told apart from the training step.
                  Text changes only; p/withdrawal are not A/B-testable in-context, so a p
                  edit is instead voted on by the A/B verdict of the text it arrives with.

  update_best  -- track the high-water mark of the held-out curve. Reporting only.

There used to be a revert_gate here. It was removed 2026-07-29: measured cycle volatility
(~+-0.08) exceeded any usable margin, so it anchored on a lucky cycle and reverted forever.
Nothing rewinds a regression now.

Everything else (what to change, based on what) is the Teacher's job.
"""
from __future__ import annotations

import os

# How many standard errors of the difference a candidate must clear.
#
# DEFAULT 0 — a strict `>` with no margin, which is the rule this project originally locked.
# The margin was added on 2026-08-03 after measuring that two passes over BYTE-IDENTICAL
# prompts differ (see ab_gate's docstring for the numbers), and removed again on 2026-08-05
# at the user's decision. What the margin cost in practice: the Teacher narrowed from 6
# scopes to 2 after its first rejection, which is the behaviour the design wants, and that
# narrowing cut the A/B's n from 180 to 60 and raised the bar it had to clear from 6 points
# to 17 — so the gate punished the Teacher for learning. Removing the margin removes that
# coupling; the cost is that same-condition noise can now be accepted.
#
# Set ARM_AB_NOISE_K=2.2 to restore the calibrated margin.
NOISE_K = float(os.environ.get("ARM_AB_NOISE_K", "0"))


def _weighted_mean(per_task, tasks):
    """n-weighted mean success over `tasks`; returns (mean, total_n)."""
    num = den = 0
    for t in tasks:
        if t in per_task:
            s, n = per_task[t]
            num += s * n
            den += n
    return (num / den if den else 0.0), den


def ab_gate(measure, tasks):
    """Decide whether to ACCEPT a proposed text change.

    measure: {"bare": {task:(succ,n)}, "current": {task:(succ,n)}, "candidate": {...}}
             success rates of the frozen policy on train, three ways, on the touched tasks
             (SAME games across the three arms — a paired comparison).
    tasks:   the touched tasks (from scaffold.touched_tasks); the comparison is aggregated
             over exactly these (a 'general' edit touches all tasks).

    ACCEPT iff the candidate beats the current text. With NOISE_K=0 (the default) that is a
    strict `>`; a positive NOISE_K additionally requires the difference to clear that many
    standard errors of the comparison.

    What a strict `>` gives up, measured rather than assumed. Generation is not reproducible
    run to run (see measure_ab_adapter): at n=24 per
    category, two passes over BYTE-IDENTICAL prompts differed by up to 6 of 24 tasks, so a bare
    `>` accepted or rejected on differences the measurement cannot resolve. Since both arms are
    Bernoulli over the same rows, the standard error of the DIFFERENCE is

        sqrt( p_c(1-p_c)/n_c + p_k(1-p_k)/n_k )

    NOISE_K multiplies it. Replaying the six real same-condition category pairs of step 40 as if
    they were candidate-vs-current gives the ratio of an observed same-condition difference to its
    own binomial SE:

        matmul 0.00 · norm_softmax 0.00 · elementwise 0.87 · reduce 1.42 · loss 1.91 · conv 2.04

    K=1 passes three of the six as improvements; K=1.5 still passes two; K=2.2 passes none.

    Those numbers are ordinary sampling noise, not a special property of this stack. A chi-square
    test of homogeneity over the six pairs gives 10.58 on 6 df against a 12.59 critical value:
    the spread is within what plain binomial sampling at n=24 predicts. The two passes agree to
    4.2 points in aggregate (30/144 vs 24/144) and the per-category swings cancel in direction —
    the alarming '1/24 vs 6/24, six times worse!' is a ratio read at a low base rate, and is 5
    tasks, or 2.2 sigma. Generation nondeterminism is real (112 of 112 recoverable responses
    differed) but mostly does not change a verdict: a different program usually meets the same
    fate. So K is not correcting for an exotic noise source. It is just 2 sigma, and the actual
    disease was always n=24.

    K sets the false-positive rate ALONE — raising n shrinks the noise and the margin by the same
    sqrt(n), leaving the ratio above unchanged. What n buys is resolution: the smallest real effect
    that can clear the bar. So the two are tuned separately — K here against the observed noise
    distribution, n in measure_ab_adapter by spending the row budget only on categories the
    proposal actually touched. Together they put the bar near 6 points, against the ~23 that a
    strict `>` at n=24 could honestly resolve.

    The errors are not symmetric, which is why a margin was tried at all. A rejected proposal
    costs one cycle — the Teacher re-proposes next cycle with the same evidence. An accepted one
    is permanent: the revert_gate was removed 2026-07-29, so nothing rewinds a scaffold that
    turns out to be noise, and it keeps being injected for the rest of the run. With the margin
    off, the held-out curve is the only thing that will show such a scaffold was noise, and it
    will show it slowly.

    No retries: a rejected proposal is discarded and the current scaffold trains on unchanged; the
    Teacher re-proposes next cycle.
    Returns {accept, reason, cand_mean, cur_mean, bare_mean, margin, n}.
    """
    if not tasks:
        return {"accept": False, "reason": "no touched tasks (nothing to A/B)", "n": 0}
    cur_mean, cur_n = _weighted_mean(measure.get("current", {}), tasks)
    cand_mean, cand_n = _weighted_mean(measure.get("candidate", {}), tasks)
    bare_mean, _ = _weighted_mean(measure.get("bare", {}), tasks)
    if not cur_n or not cand_n:
        return {"accept": False, "reason": "missing A/B samples -> reject (keep current)",
                "cand_mean": cand_mean, "cur_mean": cur_mean, "bare_mean": bare_mean, "n": 0}
    margin = NOISE_K * ((cur_mean * (1 - cur_mean) / cur_n)
                        + (cand_mean * (1 - cand_mean) / cand_n)) ** 0.5
    accept = cand_mean > cur_mean + margin
    reason = (f"candidate {cand_mean:.3f} vs current {cur_mean:.3f} "
              f"(bare {bare_mean:.3f}, margin {margin:.3f}, n {cur_n}+{cand_n}) "
              f"-> {'ACCEPT' if accept else 'reject'}")
    return {"accept": accept, "reason": reason, "cand_mean": round(cand_mean, 3),
            "cur_mean": round(cur_mean, 3), "bare_mean": round(bare_mean, 3),
            "margin": round(margin, 4), "n": cur_n + cand_n}


def update_best(best, best_step, sr, step):
    """Raw best-so-far (per the locked decision: no denoising — VAL_N averaging already
    tames the max-bias). Returns (best, best_step) possibly updated."""
    if sr is not None and (best is None or sr > best):
        return sr, step
    return best, best_step


