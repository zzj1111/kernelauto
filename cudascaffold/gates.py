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

    ACCEPT iff candidate simply beats current on the touched tasks (strict >, no margin —
    per the locked decision). No retries: a rejected proposal is discarded and the current
    scaffold trains on unchanged; the Teacher re-proposes next cycle.
    Returns {accept, reason, cand_mean, cur_mean, bare_mean, n}.
    """
    if not tasks:
        return {"accept": False, "reason": "no touched tasks (nothing to A/B)", "n": 0}
    cur_mean, cur_n = _weighted_mean(measure.get("current", {}), tasks)
    cand_mean, cand_n = _weighted_mean(measure.get("candidate", {}), tasks)
    bare_mean, _ = _weighted_mean(measure.get("bare", {}), tasks)
    if not cur_n or not cand_n:
        return {"accept": False, "reason": "missing A/B samples -> reject (keep current)",
                "cand_mean": cand_mean, "cur_mean": cur_mean, "bare_mean": bare_mean, "n": 0}
    accept = cand_mean > cur_mean
    reason = (f"candidate {cand_mean:.3f} {'>' if accept else '<='} current {cur_mean:.3f} "
              f"(bare {bare_mean:.3f}) -> {'ACCEPT' if accept else 'reject'}")
    return {"accept": accept, "reason": reason, "cand_mean": round(cand_mean, 3),
            "cur_mean": round(cur_mean, 3), "bare_mean": round(bare_mean, 3), "n": cur_n + cand_n}


def update_best(best, best_step, sr, step):
    """Raw best-so-far (per the locked decision: no denoising — VAL_N averaging already
    tames the max-bias). Returns (best, best_step) possibly updated."""
    if sr is not None and (best is None or sr > best):
        return sr, step
    return best, best_step


