"""The auto-scaffold control loop.

One cycle = train K steps with the current scaffold -> eval standalone on valid_seen
(VAL_N draws) -> gather train signals -> Teacher proposes -> A/B gate on text ->
apply accepted text + any p change -> write scaffold.
There is no revert: a regression stays in the curve (see run_cycle).

An optional triage pre-check can sit before the signals, asking whether the cycle is worth
measuring at all. It is OFF by default — the measurement it skipped stopped being expensive
when signals became a read of the training rollouts. See the note in decide().

All heavy/side-effecting work is injected as functions so this module is pure control
flow and unit-testable with mocks:
  train_fn(scaffold, from_step, to_step) -> checkpoint_id
  eval_fn(checkpoint, val_n)             -> {"avg", "per_task", "draws"}   (valid_seen, standalone)
  signals_fn(checkpoint, scaffold)       -> signals dict (train-side; see observation.assemble)
  measure_ab_fn(checkpoint, current, candidate, tasks) -> {"bare","current","candidate"}
  teacher_fn(obs)                        -> (action, note)
  triage_fn(obs)                         -> (intervene: bool, why: str)
  persist_fn(scaffold)                   -> None   (write scaffold JSON so training hot-reloads)
"""
from __future__ import annotations

import copy
import os

# Ask the Teacher whether a cycle is worth measuring before paying for the measurement.
# Default OFF: the measurement it skipped is free now, so a decline only costs proposals.
# See the note at the call site in decide(). Set to 1 to restore it.
TRIAGE_ENABLED = os.environ.get("AUTOSCAFFOLD_TRIAGE", "0") not in ("0", "false", "False")

from . import scaffold as S
from . import gates
from . import observation
from .observation import assemble_observation, p_gated_by_ab
from .teacher import propose as default_teacher
from .teacher import UNREACHABLE_NOTE


def _mark_unreachable(state, note, log, cyc):
    """Count consecutive cycles in which the Teacher could not be reached, and say so loudly.

    A dead key, an exhausted quota or a severed network leaves exactly the trace a Teacher that
    read the signals and declined leaves: an empty scaffold and a run that finishes at full
    length. One of those is a result; the other is an outage. Nothing here can fix the outage, and
    stopping would discard training that is still the plain-RL control — so the loop continues and
    refuses to let the condition pass quietly, in the log and in the state that the journal and
    any later report read from.
    """
    n = state.get("teacher_unreachable_cycles", 0) + 1 if UNREACHABLE_NOTE in (note or "") else 0
    state["teacher_unreachable_cycles"] = n
    if n:
        bar = "=" * 72
        log(f"[c{cyc}] {bar}\n"
            f"[c{cyc}] TEACHER UNREACHABLE for {n} consecutive cycle(s): {note}\n"
            f"[c{cyc}] Training and eval continue, but NO scaffold can be proposed while this "
            f"lasts. A run that finishes this way is a plain-RL control, NOT evidence that text "
            f"does not help. Check OPENAI_API_KEY / quota / network.\n"
            f"[c{cyc}] {bar}")
    return n


def new_state(step0=0, scaffold=None):
    sc = scaffold if scaffold is not None else S.empty_scaffold()
    return {
        "cycle": 0,
        "step": step0,
        "scaffold": sc,
        "sr_history": [],           # per-cycle averaged valid_seen success (recent last)
        "best": None,
        "best_step": step0,
        "decision_history": [],     # [{cycle, step, summary, sr_before, sr_after, verdict}]
        "last_eval": None,          # the eval behind sr_history[-1]; lets a restart prime (see run)
    }


def _backfill_outcome(history, sr_after):
    """Attach this cycle's valid_seen result to the most recent unresolved decision."""
    for entry in reversed(history):
        if entry.get("sr_after") is None:
            entry["sr_after"] = sr_after
            break


def _summary(action):
    """Record of what the Teacher proposed. Keeps the ACTUAL TEXT, not just the target names:
    the wording is the thing the A/B gate judged, so it is what the Teacher must see next cycle
    to learn which phrasings won or lost (otherwise it re-proposes the same failed text)."""
    items = list(action.get("item_ops") or [])
    txt = [f"{op.get('op')}:{op.get('scope') or op.get('id')}" for op in items]
    # A LIST, not a dict. This is the Teacher's memory of its own proposals — compact_history
    # feeds it back so it can see which exact wording lost an A/B and not re-propose it. Keyed by
    # "op:scope" it collided: three rubrics added to `loss` in one cycle share that key, so two of
    # the three vanished from the record and the Teacher would re-propose them as if new.
    proposed = [{k: v for k, v in
                 (("op", op.get("op")), ("scope", op.get("scope")), ("id", op.get("id")),
                  ("kind", op.get("kind")), ("text", op.get("text")),
                  ("alpha", op.get("alpha"))) if v is not None}
                for op in items]
    txt += [op["target"] for op in action.get("text_ops", [])]
    proposed += [{"op": "set", "scope": op["target"], "text": op["text"]}
                 for op in action.get("text_ops", [])]
    pch = {op["task"]: op["p"] for op in action.get("p_ops", [])}
    alpha = {op["uid"]: op["alpha"] for op in action.get("prefix_ops", [])}
    return {"text_edits": txt, "p_edits": pch, "text_proposed": proposed,
            "alpha_edits": alpha, "diagnosis": action.get("diagnosis", "")}


def _save(state, fns):
    """Persist the FULL loop state (step, scaffold, best anchor, history) so a restart resumes
    where it stopped. Called right after the eval — so a measured valid_seen survives a crash
    later in the same cycle — and again at every cycle exit. Without this a restart cold-starts
    at step 0 with an empty scaffold while stale checkpoints still exist on disk, and the loop
    silently evaluates a checkpoint from several cycles ago."""
    try:
        fns.get("state_fn", lambda *_: None)(state)
    except Exception as e:
        # Never crash a run over bookkeeping — but never fail SILENTLY either: a broken
        # state_fn disables resume, which is exactly what this function exists to prevent.
        fns.get("log", lambda *a: None)(f"[warn] state save failed: {type(e).__name__}: {e}")
    return state


def _journal(state, fns):
    """Persist the decision history (proposals + A/B verdicts + outcomes). This is both the
    audit trail AND the Teacher's memory: it is replayed into the next observation, and reloaded
    on restart. Written after every cycle so a backfilled sr_after is captured."""
    try:
        fns.get("journal_fn", lambda *_: None)(state["decision_history"])
    except Exception as e:
        fns.get("log", lambda *a: None)(f"[warn] journal write failed: {type(e).__name__}: {e}")
    _save(state, fns)
    return state


def run_cycle(state, fns, cfg):
    """Advance the loop by one cycle. Mutates `state` (the loop owns it) and returns it.
    Scaffold transitions use the immutable scaffold.apply_* helpers."""
    K = cfg.get("steps_per_cycle", 10)
    VAL_N = cfg.get("val_n", 3)
    log = fns.get("log", lambda *a: None)

    state["cycle"] += 1
    cyc = state["cycle"]

    # 1) train K steps with the current scaffold injected -> checkpoint
    to_step = state["step"] + K
    ckpt = fns["train_fn"](state["scaffold"], state["step"], to_step)
    state["step"] = to_step

    # 1b) Per-category outcomes of the rollouts that cycle just ran. Free — they were scored
    # anyway — and orders of magnitude better sampled than anything the paid measurement pass
    # produces. Kept as a series so "stuck" is visible as a trend rather than a single reading.
    rollouts = (cfg.get("_last_train_rollouts") or {})
    if rollouts:
        state.setdefault("train_rollouts", []).append(
            {"cycle": cyc, "step": to_step, "by_category": rollouts})
        state["train_rollouts"] = state["train_rollouts"][-12:]
        log(f"[c{cyc}] train rollouts " + " ".join(
            f"{k}:{v['n_correct']}/{v['n']}" for k, v in sorted(rollouts.items())))

    # 2) standalone eval on the validation anchor (valid_seen), averaged over VAL_N draws
    ev = fns["eval_fn"](ckpt, VAL_N)
    sr = ev.get("avg")
    state["sr_history"].append(sr)
    state["last_eval"] = ev
    _backfill_outcome(state["decision_history"], sr)
    log(f"[c{cyc}] step={state['step']} valid_seen avg={sr} draws={ev.get('draws')}")
    _save(state, fns)                    # measurement is expensive: never lose it to a later crash

    # 3) Track the best held-out point. REPORTING ONLY — there is no revert.
    #
    # The revert gate was removed 2026-07-29 at the user's decision. It had already been disabled
    # in every arm for a measured reason: cycle-to-cycle volatility on this benchmark is about
    # +-0.08, wider than any usable margin, so the gate anchored on whichever cycle got lucky and
    # then reverted forever — it destroyed runs instead of protecting them. Removing it rather
    # than leaving it behind a default-ON flag means a future arm cannot get it back by
    # forgetting an environment variable.
    #
    # Consequence to be aware of: nothing now rewinds a regression. The run keeps whatever the
    # last cycle produced, and a bad stretch shows up as a dip in the curve rather than being
    # undone. `best` below is the high-water mark for the record, not a checkpoint to return to.
    prev_best = state["best"]
    state["best"], state["best_step"] = gates.update_best(state["best"], state["best_step"], sr, state["step"])
    if state["best"] != prev_best:
        log(f"[c{cyc}] new best {state['best']} @ step {state['best_step']}")

    return decide(state, fns, cfg, ckpt, sr, ev, cyc)


def decide(state, fns, cfg, ckpt, sr, ev, cyc):
    """signals -> Teacher -> A/B door -> validate -> persist. Split out of run_cycle so it can
    also run BEFORE the first training stretch (see `run`), where an eval already exists but the
    Teacher has never spoken."""
    log = fns.get("log", lambda *a: None)
    ev = ev or {}

    # 4b) OFF by default since 2026-08-06 — the cost it existed to avoid is gone.
    #
    # The pre-check was worth its own GPT call when signals_fn meant a full bare+injected sweep on
    # a frozen checkpoint: ~85 min/cycle against ~44 min of training (alf_scratch150_pcap). Being
    # told "no change" after paying that was pure waste, so it was cheaper to ask first.
    #
    # signals_fn now reads the rollouts training already wrote, between two byte offsets. It costs
    # milliseconds. The only expensive thing left downstream is the A/B, and the A/B runs ONLY if
    # the Teacher proposes text — so what a decline still buys is exactly the cycles where the
    # Teacher WOULD have proposed something. That is not a saving, it is the experiment being
    # suppressed: the run exists to find out whether Teacher-written text helps, and a decline
    # removes a chance to find out while the measurement it avoids is free.
    #
    # It is also redundant. propose() can already decline, records its diagnosis, and decides with
    # strictly more information — triage never sees per_task_gap, contrastive_traces or
    # failure_patterns, which are the signals the question turns on.
    #
    # The record: alf_scratch200_stdloss declined 17 of 20 cycles and finished with an empty
    # scaffold, having measured nothing about the thing it was run to measure. Every decline was
    # defensible in isolation; the failure was that "not yet" costs nothing until the run is over.
    # `intervene_floor_cycles` was added to bound that, and is kept because it still applies when
    # the pre-check is switched back on.
    #
    # AUTOSCAFFOLD_TRIAGE=1 restores it, for a run where the A/B budget matters more than
    # proposal coverage. Fails OPEN when on: teacher.triage returns True on any error, so a
    # broken pre-check degrades to measuring rather than to silently freezing the scaffold.
    floor = cfg.get("intervene_floor_cycles")
    scaffold_empty = not S.has_text(state["scaffold"])
    forced = bool(floor) and scaffold_empty and cyc > floor

    triage_fn = fns.get("triage_fn")
    if TRIAGE_ENABLED and triage_fn is not None and not forced:
        traj = observation.eval_trajectory(state["decision_history"], state["step"], sr,
                                           (ev or {}).get("draws"))
        go, why = triage_fn(observation.assemble_triage_observation(
            state["scaffold"], traj, state["decision_history"], state["step"],
            per_task=(ev or {}).get("per_task"), per_task_n=(ev or {}).get("per_task_n"),
            train_rollouts=state.get("train_rollouts"),
            cycle=cyc, n_cycles=cfg.get("n_cycles"),
            floor_cycles=cfg.get("intervene_floor_cycles"),
            zero_gradient=cfg.get("_last_zero_gradient")))
        log(f"[c{cyc}] triage: {'measure' if go else 'SKIP'} — {why}")
        if not go:
            # triage fails CLOSED on an unreachable Teacher, so its decline is the other way an
            # outage can end a cycle. Same counter, same banner — the condition must not depend
            # on which of the two calls happened to be the one that could not get through.
            _mark_unreachable(state, why, log, cyc)
            state["decision_history"].append({
                "cycle": cyc, "step": state["step"],
                "summary": {"noop": True, "triage_declined": True, "diagnosis": why},
                "sr_before": sr, "draws_before": ev.get("draws"),
                "sr_after": None, "verdict": "noop"})
            return _journal(state, fns)
    elif forced:
        log(f"[c{cyc}] triage: FORCED — scaffold still empty after {floor} cycles; "
            f"measuring regardless of the pre-check")

    # 5) gather train-side signals for the Teacher (bare+injected on train, all-fail groups, failures)
    signals = fns["signals_fn"](ckpt, state["scaffold"])
    signals.setdefault("valid_seen", {"avg": sr, "per_task": ev.get("per_task", {}), "draws": ev.get("draws")})

    # 6) Teacher proposes (or declines)
    obs = assemble_observation(state["scaffold"], signals, state["decision_history"],
                               state["step"], cfg.get("domain"))
    try:
        action, note = fns.get("teacher_fn", default_teacher)(obs, state["scaffold"])
    except TypeError:                      # a test double that predates the scaffold argument
        action, note = fns.get("teacher_fn", default_teacher)(obs)
    log(f"[c{cyc}] teacher: {note}; edits={_summary(action)['text_edits']} p={_summary(action)['p_edits']}")

    _mark_unreachable(state, note, log, cyc)

    if S.is_noop(action):
        # Record WHY. The signals are free now, but the Teacher's reading of them is not: this is
        # the only place a cycle's diagnosis is written down, and dropping it left cycle 10's
        # decision to do nothing with no trace of what it saw. `note` is carried too — it
        # distinguishes a Teacher that deliberately declined ("ok") from one whose output failed
        # validation, and from one that could not be reached at all (below).
        log(f"[c{cyc}] teacher declined: {(action.get('diagnosis') or '(no diagnosis)')[:300]}")
        state["decision_history"].append({"cycle": cyc, "step": state["step"],
                                          "summary": {"noop": True, "note": note,
                                                      "diagnosis": action.get("diagnosis", "")},
                                          "sr_before": sr,
                                          "draws_before": ev.get("draws"),
                                          "sr_after": None, "verdict": "noop"})
        return _journal(state, fns)

    # A teacher_fn that answers in the older whole-scope form reaches here WITHOUT having gone
    # through teacher.normalize (a harness that calls the model itself, a replayed journal entry).
    # is_noop() counts text_ops as an intervention, so such an action clears the decline check and
    # is then dropped by the item-ops branch below: no A/B, no edit, and a "rejected" verdict
    # recorded against a proposal nothing ever measured. Translate it instead — the mapping is
    # exact ("this scope's text is now X" = delete what is there, add one entry).
    if action.get("text_ops") and not action.get("item_ops"):
        from . import teacher as _teacher
        action = {**action,
                  "item_ops": _teacher._as_item_ops(action, S.migrate_items(state["scaffold"],
                                                                            cfg.get("domain")),
                                                    cfg.get("domain"))}

    # 7) text changes go through the frozen-policy A/B door; p changes apply directly
    scaf = state["scaffold"]
    ab_result = None
    accepted_text = False
    p_applied = False
    if action.get("item_ops"):
        # The A/B still compares the WHOLE set, not one item: with the measured noise floor
        # (same-condition arms differed by 7.4 points at n=300 on 2026-08-03), a per-item contrast
        # would need a randomized-subset design and the user chose to keep the whole-set
        # comparison. What items buy here is a bounded, itemised diff — the budget caps how much
        # of the set can move behind a single verdict, so one ACCEPT covers at most a few entries
        # rather than a wholesale rewrite.
        candidate, budget_notes = S.apply_item_ops(scaf, action["item_ops"],
                                                   cfg.get("domain"), step=state["step"])
        for n in budget_notes:
            log(f"[c{cyc}] budget: {n}")
        tasks = S.touched_scopes(action["item_ops"], scaf, cfg.get("domain"))
        if candidate.get("items") == S.migrate_items(scaf, cfg.get("domain")).get("items"):
            # Both sides are migrated so the comparison is between two normalised shapes; comparing
            # a migrated candidate against a raw scaffold would read "changed" on every cycle.
            log(f"[c{cyc}] every edit was dropped by the budget — nothing to A/B")
        else:
            measure = fns["measure_ab_fn"](ckpt, scaf, candidate, sorted(tasks))
            ab_result = gates.ab_gate(measure, sorted(tasks))
            if ab_result["accept"]:
                scaf = candidate
                accepted_text = True
            log(f"[c{cyc}] A/B: {ab_result['reason']}")

    # A rejected text verdict also throws away the p edits that rode with it. The Teacher
    # proposes text and p as ONE action justified by one diagnosis; p is the unmeasured half
    # (text must clear the paired A/B, p never does). Letting p through after the measurement
    # refused its accompanying text applies the risky half of a proposal that just failed.
    # Observed in cycle 1 of alf_scratch150_pcap: text scored 0.078 vs bare 0.128 and was
    # rejected, yet p still went 0 -> 0.35/0.5. Harmless only because that text was empty.
    # A p-only action (no text_ops, so no A/B verdict exists) is unaffected.
    p_vetoed = bool(action.get("p_ops")) and ab_result is not None and not ab_result["accept"]
    if p_vetoed and p_gated_by_ab():
        log(f"[c{cyc}] p edits dropped with the rejected text ({len(action['p_ops'])} ops)")
    elif action.get("p_ops"):
        scaf = S.apply_p_ops(scaf, action["p_ops"])
        p_applied = True

    # Per-instance disclosure levels apply directly, like p, and deliberately do NOT go
    # through the A/B door: the A/B gate adjudicates WORDING the Teacher authored, and a
    # prefix is verbatim ground truth. What needs judging is the disclosure LEVEL, which
    # shows up only in held-out success across cycles.
    prefix_applied = False
    if action.get("prefix_ops"):
        scaf = S.apply_prefix_ops(scaf, action["prefix_ops"])
        prefix_applied = True

    # 8) physical validation before writing (fixed rule); on failure keep the pre-action scaffold
    ok, reason = S.validate_scaffold(scaf, cfg.get("domain"))
    if not ok:
        log(f"[c{cyc}] assembled scaffold invalid ({reason}); keeping previous scaffold")
        # The rollback discards EVERY edit in this action, so all three flags have to come back
        # down with it. Clearing only accepted_text left the record claiming p was applied on a
        # cycle where the scaffold never changed — and that record is both the audit trail and
        # the Teacher's memory, so it would learn from an edit that never happened.
        scaf = state["scaffold"]
        accepted_text = p_applied = prefix_applied = False

    state["scaffold"] = scaf
    fns.get("persist_fn", lambda *_: None)(scaf)
    state["decision_history"].append({
        "cycle": cyc, "step": state["step"], "summary": _summary(action),
        "ab": ab_result, "accepted_text": accepted_text,
        "p_applied": p_applied, "p_vetoed_with_text": bool(p_vetoed and p_gated_by_ab()),
        "prefix_applied": prefix_applied, "sr_before": sr,
        "draws_before": ev.get("draws"), "sr_after": None,
        "verdict": "accepted" if (accepted_text or p_applied or prefix_applied) else "rejected"})
    return _journal(state, fns)


def prime(state, fns, cfg):
    """Let the Teacher write before the first training stretch.

    A cycle is train -> eval -> Teacher, so the scaffold a cycle trains with is the one written
    from the PREVIOUS cycle's eval. On a seeded or resumed start that leaves cycle 1 training
    against whatever scaffold we happened to begin with — an empty one on a cold start, so the
    first K steps carry no signal at all and are pure-RL steps wearing an experiment's name.
    When an eval for the weights already on disk exists and the Teacher has never spoken, run
    the decision first so training starts with a real scaffold.

    No-op once any decision exists, so a mid-run restart does not double-decide.
    """
    if not state.get("sr_history") or state.get("decision_history"):
        return state
    ev = state.get("last_eval") or {}
    sr = state["sr_history"][-1]
    # The weights on disk are the ones at state["step"], full stop — nothing rewinds them any
    # more. This used to prefer state["best_checkpoint"], which paired the LATEST eval (sr, from
    # sr_history[-1]) with the BEST-SCORING checkpoint: whenever those differed, the Teacher was
    # shown signals measured on one set of weights and a held-out number from another. That was
    # only ever consistent while the revert gate was rewinding weights to best_step, and the gate
    # has been off in every arm and is now gone. A resumed state.json can still carry the old key
    # from before the removal, so read the step instead of trusting whatever is left in there.
    ckpt = f"{cfg['ckpt_root']}/global_step_{state['step']}"
    fns.get("log", lambda *a: None)(
        f"[prime] Teacher writes before training: step={state['step']} valid_seen={sr} ckpt={ckpt}")
    return decide(state, fns, cfg, ckpt, sr, ev, cyc=state.get("cycle", 0))


def run(state, fns, cfg, n_cycles):
    """Run up to n_cycles, or until a caller-supplied stop condition in cfg['stop_fn']."""
    stop_fn = cfg.get("stop_fn", lambda st: False)
    state = prime(state, fns, cfg)
    for _ in range(n_cycles):
        if stop_fn(state):
            break
        state = run_cycle(state, fns, cfg)
    return state
