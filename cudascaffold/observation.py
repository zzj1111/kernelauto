"""Assemble the Teacher's per-cycle observation and render the descriptive prompt.

The whole design intent: the harness DESCRIBES the world, the tools, and what each
signal MEANS; it never tells the Teacher what works or what to do. So the prompt is
purely descriptive ("this number measures X"), never prescriptive ("do Y when X").
The Teacher reads the complete training information and decides for itself.
"""
from __future__ import annotations

import json
import os

from . import scaffold as S


def bare_prompt_loss():
    """Whether the update re-conditions on the unscaffolded prompt.

    Read from ARM_BARE_LOSS — the SAME variable run_arm.py passes to the trainer as
    algorithm.bare_prompt_loss.enable — so the mechanism the Teacher is told about cannot drift
    away from the one the trainer is running. Deliberately a function, not a module constant:
    tests and a supervisor that sets the variable late still get the right answer.
    """
    return os.environ.get("ARM_BARE_LOSS", "True") not in ("0", "false", "False", "no", "off")


def p_gated_by_ab():
    """Whether a failed text A/B also voids the p edits submitted with it.

    Defined HERE, not in loop.py, so the rule and the sentence describing it to the Teacher read
    the same variable — loop.py imports this rather than testing the environment itself. Stating
    a rule the run is not enforcing is the same class of bug as describing the wrong loss.
    """
    return os.environ.get("AUTOSCAFFOLD_P_GATED_BY_AB", "1") not in ("0", "false", "False")


_FAIL_CHARS = 120000     # cap on serialized failure trajectories fed to the Teacher


FULL_TEXT_RECENT = 4      # keep the proposed wording verbatim for this many recent cycles


def compact_history(decision_history, recent=FULL_TEXT_RECENT):
    """Memory fed back to the Teacher. Recent cycles keep the PROPOSED TEXT verbatim (so it can
    see which exact wording won or lost the A/B and not re-propose a loser); older cycles keep
    only the outcome summary, so the prompt does not grow without bound over a long run."""
    hist = list(decision_history or [])
    out = []
    for i, e in enumerate(hist):
        keep_text = i >= len(hist) - recent
        s = dict(e.get("summary") or {})
        if not keep_text:
            s.pop("text_proposed", None)
            if isinstance(s.get("diagnosis"), str):
                s["diagnosis"] = s["diagnosis"][:200]
        out.append({**e, "summary": s})
    return out


def eval_trajectory(decision_history, current_step, current_sr, current_draws=None):
    """The held-out curve so far, as an ordered series plus its per-cycle deltas.

    Everything here is already implicit in decision_history (each entry carries the sr measured
    before it acted), but only as scattered fields. Whether the curve is still climbing is the
    single most decision-relevant fact for timing an intervention, so it is surfaced as one
    object rather than left for the Teacher to reassemble.

    Each point carries its individual `draws` where they were recorded. The averages alone cannot
    say whether a small delta means anything; the spread between draws of the SAME weights is the
    only measurement of that, and handing it over is what lets the Teacher judge noise instead of
    being handed a threshold someone picked. Measurement only — no verdict about what the shape
    means.
    """
    pts = {}
    for e in (decision_history or []):
        s, v = e.get("step"), e.get("sr_before")
        if s is not None and v is not None:
            pts[int(s)] = (round(float(v), 4), e.get("draws_before"))
    if current_step is not None and current_sr is not None:
        pts.setdefault(int(current_step), (round(float(current_sr), 4), current_draws))
    order = sorted(pts)
    series = []
    for s in order:
        v, dr = pts[s]
        row = {"step": s, "valid_seen": v}
        if dr:
            row["draws"] = [round(float(x), 4) for x in dr]
        series.append(row)
    deltas = [{"from_step": order[i - 1], "to_step": order[i],
               "delta": round(pts[order[i]][0] - pts[order[i - 1]][0], 4)}
              for i in range(1, len(order))]
    return {"series": series,
            "deltas": deltas,
            "last_delta": deltas[-1]["delta"] if deltas else None,
            "best": max((pts[s][0] for s in order), default=None)}


def assemble_observation(current_scaffold, signals, decision_history, step, domain=None):
    """Build the observation packet from measured signals only (no advice).

    signals is a dict the loop fills from train-side measurement, e.g.:
      per_task_gap:    {task: {"bare": float, "injected": float, "n": int}}   (frozen policy, train)
      all_fail_groups: {task: {"all_fail": int, "total": int}}               (train rollout groups)
      failures:        [ {task, transcript, opens, takes, held, ...} ]        (train, mode=none)
      successes:       [ ... few contrasting successes ... ]
      valid_seen:      {"avg": float, "per_task": {...}, "draws": [...]}      (validation anchor, read-only)
      train_curve:     {"episode_success": float, ...}
    """
    domain = domain or S.CUDA_DOMAIN
    return {
        "step": step,
        "objective": "Maximize STANDALONE (no-scaffold) success on held-out eval. The "
                     "scaffold is injected ONLY during training; the model is always "
                     "evaluated with NO scaffold. So the scaffold is a training aid, not "
                     "something the model keeps at test time.",
        "scopes": domain.scopes(),
        "current_scaffold": {
            "general_skill": current_scaffold.get("general_skill", ""),
            "skills": current_scaffold.get("skills", {}),
            "p_task": current_scaffold.get("p_task", {}),
            "version": current_scaffold.get("version", 0),
        },
        "signals": {
            "per_task_gap": signals.get("per_task_gap", {}),
            "all_fail_groups": signals.get("all_fail_groups", {}),
            "valid_seen": signals.get("valid_seen", {}),
            "eval_trajectory": eval_trajectory(
                decision_history, step, (signals.get("valid_seen") or {}).get("avg"),
                (signals.get("valid_seen") or {}).get("draws")),
            "train_curve": signals.get("train_curve", {}),
            "per_instance": signals.get("per_instance", {}),
        },
        "failure_trajectories": signals.get("failures", []),
        "successes_for_contrast": signals.get("successes", []),
        "decision_history": compact_history(decision_history),
    }


_SIGNAL_MEANINGS = (
    "WHAT THE SIGNALS MEAN (descriptions, NOT instructions):\n"
    "- signals.per_task_gap[task] = success of the CURRENT frozen policy on train, "
    "'bare' (no scaffold) vs 'injected' (current scaffold). The gap (injected - bare) is "
    "how much the current scaffold helps this policy in-context right now.\n"
    "- signals.all_fail_groups[task] = how many training rollout groups had EVERY rollout "
    "fail. A group where all rollouts get the same outcome gives the RL update no gradient, "
    "so all-fail groups are places the policy currently gets no learning signal.\n"
    "- signals.valid_seen = the held-out standalone success (the objective). Read-only: you "
    "cannot see or optimize the final test set (unseen); this is the validation signal.\n"
    "- signals.eval_trajectory = that same held-out number across ALL cycles so far: `series` "
    "(step, value, and the individual `draws` behind that average), `deltas` (cycle-over-cycle "
    "change), `last_delta`, and `best`. The draws are repeats on the SAME weights, so their "
    "spread measures how much a delta can move by chance. All of it is measurement; what the "
    "shape implies is your call.\n"
    "- signals.per_instance[instance_id] = how the CURRENT policy does on ONE instance, bare: "
    "how many attempts, what fraction were correct, and mean speedup when correct. "
    "failure_trajectories is the same set sorted worst-first. These are what a per-instance "
    "target has to act on — a category mean cannot say WHICH instance is stuck.\n"
    "- failure_trajectories = the worst instances under the bare prompt, worst first.\n"
    "- decision_history = YOUR OWN MEMORY across cycles. Each entry records what you proposed "
    "(summary.text_proposed = the exact wording, kept verbatim for recent cycles; "
    "summary.diagnosis = your reasoning at the time), whether the A/B gate accepted it "
    "(verdict accepted/rejected/noop/reverted, with ab.cand_mean vs ab.cur_mean vs ab.bare_mean), "
    "and what happened to held-out success afterwards (sr_before -> sr_after). A 'rejected' entry "
    "means that exact wording measured no better than what it would have replaced.\n"
)

_MECHANISM = (
    "THE INJECTION MECHANISM (what the system physically does with your text):\n"
    "- You attach TEXT to a SCOPE. During TRAINING ONLY, the text attached to an episode's "
    "scope is spliced into that episode's prompt with probability p (decided once per rollout "
    "group). At EVALUATION nothing is ever spliced in.\n"
    "- Scopes available in this domain: {scopes}.\n"
    "  'general' text is spliced into every episode; text on a category label is spliced only "
    "into episodes carrying that label. Text on both is spliced together.\n"
    "- p is per category, in [0, {p_max}]. p={p_max} injects into that fraction of groups, p=0 "
    "nothing (fully withdrawn). {p_max} is a HARD CAP enforced by the system, not a preference: "
    "anything higher is clamped down to it, so at least half of every category's rollout groups "
    "always see the bare prompt. Proposing a larger number does not get you a larger number.\n"
    "- p also moves at most {p_max_delta} per cycle in either direction, enforced the same way: "
    "a larger jump is trimmed to that step, not rejected. Both the cap and the step limit report "
    "what they trimmed, so you can see when a request was larger than what took effect.\n"
    "- p starts at 0 for every category, i.e. NOTHING is injected until you raise it. Text you "
    "attach is inert while its category's p is 0: to actually train against it you must set a "
    "p_op for that category in the same action. Withdrawing is setting p back toward 0.\n"
    "- Where the domain says text can be attached to an individual instance, a text_op target of "
    "`instance:<instance_id>` attaches text to THAT instance only. It is spliced after the "
    "general and category text, so an instance receives all three. The instance's own injection "
    "still rides its category's p — attaching text to an instance whose category has p=0 changes "
    "nothing. At most 40 instance entries are kept; the oldest are dropped.\n"
    "- You may also intervene on nothing this cycle.\n"
)

_WHEN_TO_INTERVENE = (
    "WHEN TO INTERVENE (this section IS a directive, unlike the mechanism sections above):\n"
    "\n"
    "What injecting costs, every time it happens:\n"
    "- Context. The spliced block occupies prompt budget the policy would otherwise spend on "
    "the observation and the admissible-action list.\n"
    "- Exploration. An injected group is a group the policy did NOT explore on its own. You are "
    "spending its rollouts to see what YOUR text elicits instead of what the policy would have "
    "found. When the policy is already finding successes unaided, that trade is a loss.\n"
    "- Transfer risk. The loss is computed on the bare prompt, so the gradient pushes the "
    "student toward behaviour that was elicited under text it will never see at evaluation. "
    "Behaviour the bare policy cannot reproduce is gradient spent on something unreachable.\n"
    "\n"
    "The benefit exists only where the policy is STUCK. Text buys something when it converts "
    "all-fail groups (which contribute zero gradient) into mixed ones. Where the policy is "
    "already improving, there is little to buy and the three costs above are still paid in "
    "full.\n"
    "\n"
    "Therefore:\n"
    "- While signals.eval_trajectory is still climbing steadily, PREFER TO DECLINE. The policy "
    "is generating its own learning signal; let it. Measured in this harness: a broad scaffold "
    "written at the first opportunity scored 0.078 against a bare prompt's 0.128 and was "
    "rejected by the A/B — early, wide intervention actively hurt.\n"
    "- Step in when the curve FLATTENS or REGRESSES, and target the categories whose "
    "all_fail_groups stay high while the total has stopped moving. That is the point where the "
    "policy has stopped producing gradient for itself and text has something to convert.\n"
    "- Read signals.eval_trajectory before acting. Each point carries the individual draws behind "
    "its average — repeated evaluations of the SAME weights — so the spread among them tells you "
    "directly how large a delta has to be before it means anything. Judge that from the numbers "
    "in front of you rather than from any fixed rule.\n"
    "- Declining is a first-class action with zero cost. Injecting unhelpful text costs the A/B "
    "measurement, the training steps that ride on it, and possibly held-out success. When in "
    "doubt, decline and re-read the trajectory next cycle.\n"
    "{p_limits}"
)

def _p_limits_text():
    """The p rules AS CONFIGURED. Both are switchable per arm, so neither may be hard-coded into
    the prompt: an arm that pins the older rules would otherwise be told it is bound by limits it
    is not, and plan around a constraint that does not exist."""
    out = []
    if S.P_MAX_DELTA < S.P_MAX:
        out.append(
            f"- p may move at most {S.P_MAX_DELTA} per cycle, up or down. Reaching a high "
            "injection rate takes several cycles by design, so a category you want to help needs "
            "a decision made early enough to ramp, not a single large jump.\n")
    if p_gated_by_ab():
        out.append(
            "- p edits proposed ALONGSIDE text that then fails its A/B are discarded with that "
            "text. If you want a p change to survive on its own merits, propose it WITHOUT a "
            "text edit.\n")
    if not out:
        return ""
    return "\nLimits on p you should plan around, enforced by the system:\n" + "".join(out)

_CONTENT_VOCAB = (
    "KINDS OF SCAFFOLD CONTENT (a vocabulary, NOT a recommendation — the text is free-form, so "
    "what you write into a scope is your choice). Each acts on a different reason a policy "
    "fails, and each fails in its own way; what follows is what they do and what they cost, not "
    "which to pick:\n"
    "\n"
    "- skills: strategy or procedure guidance for a class of problem. Acts on a policy that has "
    "the capability but not the routine — the failures share a procedural cause. Costs nothing "
    "but prompt budget when it names a routine the policy already follows, which is the common "
    "way skills waste a cycle. What it can leave behind is a habit, so it transfers relatively "
    "well to the unscaffolded evaluation.\n"
    "\n"
    "- rubrics: explicit criteria for what a good answer must satisfy. Acts on a different "
    "failure entirely — a policy producing acceptable work that scores badly, i.e. optimising "
    "something other than what it is graded on. Says what is being judged, not how to do it, so "
    "it is useless when the policy simply cannot do the thing: naming a standard does not supply "
    "the means, and stating criteria invites satisfying them literally. Worth knowing here that "
    "the student is scored against criteria it is never shown.\n"
    "\n"
    "- examples: a worked demonstration of a solved case. Acts on failures of FORM — structure, "
    "API use, boilerplate, output contract — where the policy knows the substance but not the "
    "shape. The most expensive kind in prompt budget, and the one most likely to be copied "
    "specifically rather than generalised. It also narrows how much rollouts differ from each "
    "other, which matters because a group whose rollouts all score the same gives the update no "
    "gradient.\n"
    "\n"
    "- hints: a partial reveal of a known-good solution for a specific instance. Acts on one "
    "instance rather than a class, so its reach is small. Putting spread back into a group where "
    "every rollout fails is not special to hints — it is what ANY of these kinds does when it "
    "works, since a group whose rollouts all score the same gives the update no gradient. What "
    "sets a hint apart is force: revealing part of the answer can make success nearly certain "
    "rather than merely more likely. The side effects scale with that force — the policy is "
    "evaluated without the text, so the more of the answer a hint supplies, the more of what it "
    "learns is that answer rather than the reasoning that reaches it. Only possible where "
    "reference solutions exist AND text can be attached per instance.\n"
    "\n"
    "These are not modes to select, and nothing stops one text from doing several at once.\n"
    "{availability}"
    "Which of the available ones is WORTH using is not stated anywhere — that is what the "
    "SIGNALS are for.\n"
)



_PRIORS = (
    "PRIOR KNOWLEDGE FROM EARLIER RUNS (calibration, not orders — the signals still decide):\n"
    "\n"
    "On p (injection probability):\n"
    "- Holding p=1.0 for the whole run is known to BACKFIRE. In a matched ablation on a "
    "different domain, never withdrawing left standalone success BELOW the no-scaffold "
    "baseline (37.6 vs 42.9), while decaying p to 0.3 over training reached 45.9. The model "
    "must spend training steps on the bare prompt it will face at evaluation, or it learns "
    "behaviour that only works with the text present.\n"
    "- p is the UNGUARDED lever. Text changes must pass a paired A/B before they take effect; "
    "p changes apply immediately with no measurement behind them. Treat a large p move as the "
    "riskier action, not the safer one.\n"
    "- Intermediate values exist and are usually better than 0 or 1. Setting p=0 on a category "
    "whose text is helping discards that help entirely; halving p keeps it while giving the "
    "policy bare-prompt practice on the same category.\n"
    "\n"
    "On measurement noise:\n"
    "- The per-category measurements are small samples. In this harness the SAME quantity has "
    "come back as +0.167 and -0.075 on two consecutive cycles. A per-category difference "
    "under roughly 0.10 is not distinguishable from noise at these sample sizes.\n"
    "- Held-out per-category rates are noisier still (a handful of episodes per category per "
    "draw); the held-out TOTAL is the only number measured on a large enough sample to read "
    "directly. Prefer a decision that a single noisy reading cannot invert, and prefer "
    "reversible moves (halve p) over irreversible ones (p=0) when the evidence is one reading.\n"
    "\n"
    "On where scaffolding can help at all:\n"
    "- The RL update gets NO gradient from a rollout group where every rollout fails. Turning "
    "such a group into a mixed one creates learning signal where there was none. Categories "
    "with many all-fail groups are therefore where injected text has something to buy; a "
    "category already succeeding almost always has little for it to buy, and text there mostly "
    "perturbs a policy that is already working.\n"
    "- A category at or near ceiling on held-out cannot improve. Edits there cost a cycle and "
    "risk a regression.\n"
    "\n"
    "On content:\n"
    "- Text that restates what the policy already does reliably has measured no effect. Text "
    "that has measured an effect named ONE concrete failure mode visible in the trajectories "
    "and gave the rule that avoids it (e.g. an ordering constraint the policy keeps violating). "
    "Prefer specific, checkable rules over general procedure descriptions.\n"
    "- Rewriting text that is currently measuring POSITIVE has repeatedly produced something "
    "worse. When the current text is winning its A/B, the higher-value move is usually "
    "elsewhere.\n"
)



_OPTIMIZATION = (
    "HOW THE STUDENT IS OPTIMIZED, AND WHAT YOUR TEXT ACTUALLY CHANGES (mechanism, not advice):\n"
    "\n"
    "1. RL update. For each training instance the system samples a GROUP of rollouts and scores "
    "them. The advantage is computed WITHIN the group, relative to that group's own mean. So a "
    "group whose rollouts all get the same outcome contributes NOTHING to the update: if every "
    "rollout in a group fails, every advantage in it is zero and no gradient flows from that "
    "instance at all. signals.all_fail_groups counts exactly those.\n"
    "\n"
    "2. Where injected text can change the outcome. Because the update is group-relative, text "
    "that turns an all-fail group into a group with at least one success creates learning signal "
    "where there was none. Text attached to instances that already succeed does not create "
    "signal; it only changes which rollouts are above or below their own group's mean.\n"
    "\n"
    "{loss_section}"
    "\n"
    "4. Evaluation is always bare. signals.valid_seen is measured with nothing injected, on a "
    "held-out split. It is the only number that reflects what the student can do on its own.\n"
)

# Section 3 describes what the update is actually conditioned on, and that depends on a training
# flag. It is derived from the SAME environment variable that configures the trainer (see
# run_arm.py) rather than written out twice, because a Teacher reasoning from the wrong loss is
# worse than one told nothing: every judgment about what text can buy follows from this.
_OPT_LOSS_BARE = (
    "3. THE LOSS IS COMPUTED ON THE BARE PROMPT. This is the part that makes this setup different "
    "from ordinary prompt engineering. The rollout is generated while your text is present, so "
    "your text decides WHICH trajectories get explored. But every log-probability in the update -- "
    "the behaviour policy's, the reference policy's, and the updated policy's -- is recomputed on "
    "the prompt WITHOUT your text. The gradient therefore raises the probability of those "
    "trajectories under the bare prompt.\n"
    "\n"
    "   Consequences that follow from this, which you can verify in the signals:\n"
    "   - The student is never trained to condition on your text. It is trained to produce, "
    "unaided, the behaviour your text elicited. Your text is an exploration device, not a context "
    "the student will keep.\n"
    "   - A success that the bare policy could plausibly have produced transfers; a success that "
    "depends on the text being present has to be reproducible from the bare prompt or the "
    "gradient is pushing toward something the student cannot reach. signals.per_task_gap "
    "(injected minus bare) measures how far apart those two conditions are right now, and "
    "signals.valid_seen measures what actually survived into the weights.\n"
    "   - Injection probability p therefore trades off two things you can see separately: higher p "
    "means more groups explored with help (more rescued all-fail groups), lower p means more "
    "training steps spent on the exact prompt the student is evaluated under.\n"
)

_OPT_LOSS_STANDARD = (
    "3. THE LOSS IS COMPUTED ON THE PROMPT THAT WAS ACTUALLY USED. This is ordinary RL: for an "
    "injected group, the rollout is generated with your text present AND every log-probability in "
    "the update is conditioned on that same prompt, text included. For a group that was not "
    "injected, both are the bare prompt.\n"
    "\n"
    "   Consequences that follow from this, which you can verify in the signals:\n"
    "   - On injected groups the student IS trained to condition on your text. It learns the "
    "behaviour together with the context that prompted it. But it is evaluated with that context "
    "absent, so anything it learned to do only-when-the-text-is-there does not show up in the "
    "score.\n"
    "   - WITHDRAWAL IS THEREFORE THE TRANSFER MECHANISM, not an afterthought. What makes a "
    "behaviour survive to evaluation is training steps taken on the bare prompt: groups where p "
    "did not fire, and every group once p reaches 0. Text that is never withdrawn produces a "
    "student that depends on it. Raising p buys exploration; lowering p is what converts that "
    "exploration into standalone ability. Both are your decision and neither happens on its own.\n"
    "   - signals.per_task_gap (injected minus bare, same frozen policy) is the size of the crutch "
    "right now: a large gap means much of the behaviour is still context-dependent and withdrawal "
    "has not finished. signals.valid_seen measures what actually survived into the weights.\n"
)


def render_domain_facts(domain):
    """Purely structural facts about the training domain. No advice, no scaffold suggestions —
    the Teacher must infer from this what kinds of scaffolding the structure even permits."""
    lines = [f"- Domain: {domain.name}", f"- Episode: {domain.episode_desc}"]
    if domain.categories:
        lines.append(f"- Every instance carries exactly ONE of {len(domain.categories)} category "
                     f"labels. The labels, and what each denotes:")
        for c in domain.categories:
            lines.append(f"    * {c}: {domain.category_info.get(c, '(no description)')}")
    else:
        lines.append("- Instances carry no category labels.")
    lines.append(f"- Reference (known-good) solutions available per instance: "
                 f"{'YES' if domain.has_reference_solutions else 'NO'}.")
    lines.append(f"- Text can be attached to an individual instance: "
                 f"{'YES' if domain.instance_scope else 'NO'}.")
    if domain.action_primitives:
        lines.append(f"- The only valid actions are: {', '.join(domain.action_primitives)}.")
    lines.extend(f"- {f}" for f in domain.extra_facts)
    return "\n".join(lines)


def _content_availability(domain):
    """Which content kinds this domain can physically carry, and why.

    Derived from the same two booleans the Teacher is shown rather than written per domain, so
    it cannot claim a kind the mechanism does not support. Only availability is stated; whether
    a kind is worth using is left to the signals, which is where the evidence is.
    """
    lines = ["Which are POSSIBLE here follows from the domain structure above:\n"]
    lines.append("- skills: YES — any scope holds free-form text.\n")
    lines.append("- rubrics: YES — a rubric is just text stating what a good answer must "
                 "satisfy, so any scope can hold one.\n")
    lines.append("- examples: YES — a worked demonstration is text you compose; it does not "
                 "require the domain to supply one.\n")
    if domain.has_reference_solutions and domain.instance_scope:
        lines.append("- hints: YES — known-good solutions exist AND text can be attached to a "
                     "single instance, so part of a solution can be revealed for that instance.\n")
    else:
        why = []
        if not domain.has_reference_solutions:
            why.append("no known-good solution ships with an instance")
        if not domain.instance_scope:
            why.append("text cannot be attached to a single instance")
        lines.append(f"- hints: NO — {' and '.join(why)}. Anything you wrote as a hint would be "
                     "your own guess at a solution rather than a reveal of a known one.\n")
    return "".join(lines)


def render_system_prompt(domain=None, priors=False):
    """Descriptive system prompt: domain STRUCTURE, injection MECHANISM, content VOCABULARY,
    signal MEANINGS. It never says which scaffold to use or at what granularity.

    `priors=True` additionally supplies calibration distilled from earlier runs (see _PRIORS).
    It is OFF by default on purpose: with it off, anything the Teacher works out — that
    withdrawal is necessary, that all-fail groups are where the leverage is — it worked out
    from the signals alone, and that is a result. With it on, those become instructions and
    the run can no longer support that claim. Keep the flag so the two are comparable.
    """
    domain = domain or S.CUDA_DOMAIN
    return (
        "You are the Teacher in an automated RL training run. A weak language-model agent is "
        "being trained with reinforcement learning (GiGPO). Your job is to shape the text that "
        "is injected into its TRAINING prompts.\n\n"
        "DOMAIN STRUCTURE (facts, not advice):\n"
        + render_domain_facts(domain) + "\n\n"
        + _MECHANISM.format(scopes=domain.scopes(), p_max=S.P_MAX,
                            p_max_delta=S.P_MAX_DELTA) + "\n"
        + _CONTENT_VOCAB.format(availability=_content_availability(domain)) + "\n"
        + _OPTIMIZATION.format(
            loss_section=(_OPT_LOSS_BARE if bare_prompt_loss() else _OPT_LOSS_STANDARD)) + "\n"
        + _SIGNAL_MEANINGS + "\n"
        + _WHEN_TO_INTERVENE.format(p_limits=_p_limits_text()) + "\n"
        + (_PRIORS + "\n" if priors else "")
        + "HOW YOUR OUTPUT IS USED: text changes are accepted only if, on the current frozen "
        "policy, the new text beats the current text in a paired A/B on train. p changes are "
        f"capped at {S.P_MAX}"
        + (f" and rate-limited to {S.P_MAX_DELTA} per cycle" if S.P_MAX_DELTA < S.P_MAX else "")
        + ("; p edits submitted together with text that then FAILS its A/B are discarded along "
           "with that text, so a p change you want judged on its own should be submitted without "
           "a text edit" if p_gated_by_ab() else "")
        + ". Nothing rewinds a regression: there is no revert, so a bad cycle stays in the "
        "curve. You are told this so you understand the loop — decide freely based on the "
        "signals.\n\n"
        "HOW TO WRITE THE TEXT. There is no length target. Write whatever it takes to be "
        "unambiguous, and prefer being explicit over being short.\n"
        "- The reader is a small model that already knows the vocabulary. Restating what it knows "
        "buys nothing; the text is only worth its prompt budget where it says something the "
        "policy is not already doing. Look at what the rollouts actually got wrong and address "
        "that, not the topic in general.\n"
        "- Prefer instructions whose effect you could check by reading one emitted kernel. "
        "'Handle the reduction tail when the length is not a multiple of the block size' changes "
        "what gets written; 'optimize memory access patterns' does not.\n"
        "- Name the concrete thing: the operator, the layout, the launch shape, the failure to "
        "avoid. Vague text is worse than no text — it costs prompt budget and the A/B gate cannot "
        "tell it apart from nothing, so it wastes the cycle it is measured in.\n"
        "- The policy is evaluated with this text ABSENT. Write what teaches a habit it can keep, "
        "not a crutch that only works while the text is there.\n\n"
        "Return ONLY JSON: {\"diagnosis\": \"<your reasoning>\", "
        "\"text_ops\": [{\"target\": \"<scope>\", \"text\": \"<the text to attach>\"}], "
        "\"p_ops\": [{\"task\": \"<category>\", \"p\": <float 0..1>}]}. "
        "Empty text_ops and empty p_ops means you choose not to intervene this cycle."
    )


def _balanced_trim(fails, keep_fn):
    """Drop failures round-robin across task types until `keep_fn` accepts the list.

    Trimming from the tail is what a naive budget cap does, and it is wrong here: the list
    arrives grouped by task type, and a 50-step ALFWorld failure serialises to ~8k chars, so a
    160k budget deletes roughly the last 25 of 40 — i.e. entire task types, chosen by nothing
    but list order. The Teacher then sees failures for the first categories only and cannot
    tell that from "those categories had no failures". Round-robin keeps every type represented
    and spends the budget on breadth instead of on whoever sorted first.
    """
    from collections import OrderedDict
    buckets = OrderedDict()
    for f in fails:
        buckets.setdefault((f or {}).get("task_type", "?"), []).append(f)
    order = []                      # interleave: one per type, then the next of each, ...
    while any(buckets.values()):
        for k in list(buckets):
            if buckets[k]:
                order.append(buckets[k].pop(0))
    kept = list(order)
    while kept and not keep_fn(kept):
        kept = kept[:-1]            # the tail is now the LEAST-represented type's extras
    return kept


def render_user_prompt(obs):
    """The measured observation packet, trimmed to a char budget on the heavy field.

    A trim is recorded in the packet itself (failure_trajectories_dropped): silently shrinking
    the Teacher's only view of real behaviour makes "no failures on this task" and "trimmed
    away" indistinguishable, and the Teacher reasons from that field.
    """
    body = json.dumps(obs, ensure_ascii=False, default=str)
    budget = _FAIL_CHARS + 40000
    if len(body) > budget and obs.get("failure_trajectories"):
        trimmed = dict(obs)
        original = list(obs["failure_trajectories"])
        fails = _balanced_trim(
            original,
            lambda cand: len(json.dumps({**trimmed, "failure_trajectories": cand},
                                        ensure_ascii=False, default=str)) <= budget)
        trimmed["failure_trajectories"] = fails
        if len(fails) < len(original):
            from collections import Counter
            trimmed["failure_trajectories_dropped"] = {
                "kept": len(fails), "of": len(original),
                "note": "trimmed to a char budget, balanced across task types; "
                        "absence here does NOT mean the task had no failures",
                "kept_by_task": dict(Counter((f or {}).get("task_type", "?") for f in fails)),
                "of_by_task": dict(Counter((f or {}).get("task_type", "?") for f in original)),
            }
        body = json.dumps(trimmed, ensure_ascii=False, default=str)
    return "MEASURED STATE (only measurements — infer everything yourself):\n" + body


_TRIAGE_PROMPT = (
    "You are the Teacher in an automated RL training run, being asked ONE cheap question before "
    "any expensive measurement is taken.\n"
    "\n"
    "A weak agent is being trained with RL. You can attach text to its TRAINING prompts (it is "
    "always EVALUATED with no text). Deciding WHAT to attach requires a full measurement pass "
    "over the training set — several bare and injected rollouts per category, plus a paired A/B "
    "if you propose new wording. That pass costs roughly as much wall-clock as the training it "
    "sits between. It is only worth paying when you are actually going to act.\n"
    "\n"
    "So: should this cycle be measured at all?\n"
    "\n"
    "Reasons to say NO (and let training continue untouched):\n"
    "- Held-out success is still climbing on its own. The policy is generating its own learning "
    "signal; injected text costs prompt budget, replaces rollouts the policy would have explored "
    "itself, and risks teaching behaviour that depends on text absent at evaluation.\n"
    "- The current scaffold is already winning its A/B and nothing suggests it has stopped "
    "working. Rewriting text that is measuring positive has repeatedly produced something worse.\n"
    "- The last change has not had a full cycle to show up in held-out success yet.\n"
    "\n"
    "Reasons to say YES:\n"
    "- Held-out success has stopped improving, by whatever reading of the trajectory and its "
    "draws you find convincing. That is where the policy has stopped producing gradient for "
    "itself and text has something to convert.\n"
    "- A category you believe is stuck needs its injection probability moved, and p only moves in "
    "bounded steps per cycle, so a change you want in effect later has to start now.\n"
    "- train_rollouts_by_category shows a category the policy is getting no gradient from — a "
    "near-zero correct_rate means every rollout in those groups scores the same, so the update "
    "has no spread to learn from. That is a LOCAL stall, and it can be true while the overall "
    "held-out number is still climbing. Waiting for the total to flatten before touching a "
    "category that has been at zero for cycles is waiting for the wrong signal.\n"
    "\n"
    "WHAT SAYING YES ACTUALLY RISKS. Text you propose does not go straight into training. It "
    "goes through a paired A/B on the FROZEN policy — same rows, same seed, current scaffold vs "
    "your candidate — and is discarded unless it wins. p edits submitted with rejected text are "
    "discarded with it, and p moves in bounded steps. So a wrong proposal costs one measurement "
    "pass, not a damaged run: the failure mode you are guarding against downstream is already "
    "guarded. Weigh YES against the cost of the measurement, not against the risk of the text.\n"
    "\n"
    "WHAT SAYING NO COSTS. `progress` tells you which cycle this is, how many remain, whether "
    "the scaffold still has no text at all, and how many cycles you have already declined. A run "
    "that reaches its last cycle with an empty scaffold has measured nothing about whether text "
    "helps — the one question it exists to answer. Each individual cycle looks like a reasonable "
    "one to skip; that is exactly how a run ends with twenty declines and no result. If the "
    "scaffold is still empty and the run is well past its first few cycles, the burden shifts: "
    "prefer measuring unless you can say concretely what a later cycle will show that this one "
    "does not.\n"
    "\n"
    "valid_seen_per_task breaks the latest held-out number down by category, each with the "
    "`n_episodes` it was computed from. A total that has stopped moving can mean every category "
    "has run out of room, or that most have while one or two are stuck well below the rest — "
    "only the breakdown separates those, and only the stuck case is worth measuring. Read "
    "`n_episodes` before trusting any category: a rate over a handful of episodes can read 0.000 "
    "or 1.000 and mean nothing.\n"
    "\n"
    "eval_trajectory gives each point's individual draws as well as its average. Those draws "
    "re-evaluate the SAME weights, so their spread is your measurement of how much a delta can "
    "move by chance alone — use it to decide what counts as flat. Note what follows if that "
    "spread is wide: a delta smaller than it can never be established, so 'the curve has not "
    "convincingly flattened' may be unfalsifiable on this run rather than evidence of progress. "
    "When the held-out signal cannot settle the question, decide on train_rollouts_by_category, "
    "which is sampled orders of magnitude better and does not depend on that curve at all.\n"
    "\n"
    "Return ONLY JSON: {\"intervene\": true|false, \"why\": \"<one or two sentences>\"}.\n"
)


def render_triage_prompt():
    """System prompt for the cheap intervene/decline pre-check. Deliberately does NOT include the
    domain facts, injection mechanics or content vocabulary: this decision is about the shape of
    the held-out curve, not about what to write. Keeping it small is the point — it runs before
    the measurement that the answer decides whether to pay for."""
    return _TRIAGE_PROMPT


def assemble_triage_observation(current_scaffold, eval_traj, decision_history, step,
                                per_task=None, per_task_n=None, train_rollouts=None,
                                cycle=None, n_cycles=None, floor_cycles=None):
    """The cheap observation: only what is already computed. No train-side measurement, which is
    exactly the cost this pre-check exists to avoid.

    per_task/per_task_n come from the eval that just ran, so they are free. They matter because
    the TOTAL can sit flat for two very different reasons: every category at its ceiling (nothing
    left to buy) or most at ceiling while one or two are stuck (leverage). Those look identical
    in the aggregate. Counts travel with the rates because a category's rate is a mean over
    however many episodes of that type were sampled, and a rate standing on a few episodes cannot
    be read at all.

    train_rollouts is the addition that makes this question answerable. The pre-check was being
    asked to spot a stuck category while the only breakdown it received came from the held-out
    eval — which in some domains has a single key, and in all of them is far too small to read a
    category from. Meanwhile every training rollout of the cycle was already scored per category
    and thrown away. Feeding those back gives thousands of samples per category, costs nothing,
    and is independent of the noise-dominated held-out curve.

    `progress` exists because a Teacher with no horizon has no reason to ever stop waiting: every
    individual cycle looks like a fine one to skip. Knowing the run ends, and that the scaffold is
    still empty, is what makes "not yet" a claim with a cost attached.
    """
    per_task = per_task or {}
    per_task_n = per_task_n or {}
    has_text = bool((current_scaffold.get("general_skill") or "").strip()) or any(
        (v if isinstance(v, str) else (v or {}).get("text", "")).strip()
        for v in (current_scaffold.get("skills") or {}).values())
    obs = {
        "step": step,
        "eval_trajectory": eval_traj,
        "valid_seen_per_task": {t: {"success": v, "n_episodes": per_task_n.get(t)}
                                for t, v in per_task.items()},
        "current_scaffold": {
            "general_skill": current_scaffold.get("general_skill", ""),
            "skills": current_scaffold.get("skills", {}),
            "p_task": current_scaffold.get("p_task", {}),
            "version": current_scaffold.get("version", 0),
        },
        "decision_history": compact_history(decision_history, recent=2),
    }
    if train_rollouts:
        obs["train_rollouts_by_category"] = train_rollouts
    if cycle is not None:
        n_declined = sum(1 for d in (decision_history or [])
                         if (d.get("summary") or {}).get("noop"))
        obs["progress"] = {
            "cycle": cycle, "n_cycles": n_cycles,
            "cycles_remaining": (n_cycles - cycle) if n_cycles else None,
            "scaffold_has_text": has_text,
            "cycles_declined_so_far": n_declined,
            "forced_measure_after_empty_cycles": floor_cycles,
        }
    return obs
