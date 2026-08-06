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
      zero_gradient_groups: {task: {"all_fail": int, "total": int}}               (train rollout groups)
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
        # `items` is the addressable form and `general_skill`/`skills` its rendering. Both are
        # shown: the ids are what an update or delete names, and the rendered strings are what the
        # policy actually reads, so a Teacher given only one of them could either not address an
        # entry or not see what that entry does to the prompt.
        "current_scaffold": {
            "items": {sc: [{k: it.get(k) for k in ("id", "kind", "text", "step")}
                           for it in S.items_of(current_scaffold, sc)]
                      for sc in S.scopes_of(current_scaffold, domain)},
            "general_skill": current_scaffold.get("general_skill", ""),
            "skills": current_scaffold.get("skills", {}),
            "p_task": current_scaffold.get("p_task", {}),
            "version": current_scaffold.get("version", 0),
        },
        "item_budget": S.cycle_budget(current_scaffold, domain),
        "signals": {
            "per_task_gap": signals.get("per_task_gap", {}),
            "zero_gradient_groups": signals.get("zero_gradient_groups", {}),
            "valid_seen": signals.get("valid_seen", {}),
            "eval_trajectory": eval_trajectory(
                decision_history, step, (signals.get("valid_seen") or {}).get("avg"),
                (signals.get("valid_seen") or {}).get("draws")),
            # per_instance stays only where the loop can fill it; an always-empty field described
            # in the prompt is a claim the data never supports.
            **({"per_instance": signals["per_instance"]} if signals.get("per_instance") else {}),
        },
        # The zero-gradient failures, beside what succeeding in that category looks like.
        # The failures come from ALL-FAIL groups because those are what the scaffold exists
        # to fix; requiring a same-game success to pair with would have excluded exactly
        # them. A category with nothing on one side says why rather than going blank.
        **({"contrastive_traces": signals.get("contrastive_traces", {})} if domain.multi_turn
           else {"failure_trajectories": signals.get("failures", [])}),
        # Counted over EVERY failed rollout of the cycle, not over the handful that fit in
        # the prompt, and a few hundred characters against tens of thousands for traces —
        # so it survives any trimming the traces get. Gated with the traces it summarises:
        # the patterns are step-sequence properties (a command reissued, a reply looped),
        # which a single-turn domain has no steps to compute.
        **({"failure_patterns": signals.get("failure_patterns", {})} if domain.multi_turn
           else {}),
        "decision_history": compact_history(decision_history),
    }


_SIGNAL_MEANINGS = (
    "WHAT THE SIGNALS MEAN (descriptions, NOT instructions):\n"
    "- signals.per_task_gap[task] = success on the TRAINING rollouts of THE LAST CYCLE ONLY, split "
    "by whether the injection probability fired for that group: 'bare' (no text) vs 'injected' "
    "(current text). The gap (injected - bare) is how much the current text changes in-context "
    "success. Which side a group lands on is drawn at random with probability p_task and does not "
    "depend on the group's difficulty, so the two sides are comparable. Two properties to keep in "
    "view: the rollouts span every policy the cycle trained through rather than one frozen policy, "
    "so the gap carries that drift; and `n_bare` and `n_injected` are RAW episode counts from that "
    "one cycle, with NOTHING carried over from earlier ones. There is no smoothing here — a gap "
    "that moves between cycles may be a real change or may be sampling noise at that n, and the "
    "only defence is to read the n before reading the gap. A category shows "
    "gap=null when one side has no rollouts, and `no_injection_reason` or `no_bare_reason` then "
    "states which of three situations produced it: the scope holds no text, p_task is zero there, "
    "or p_task is positive but no group happened to fire. The three call for different actions, "
    "and none of them is a measured gap of zero.\n"
    "{trace_shape}"
    "- signals.zero_gradient_groups[task] = of the TRAINING rollout groups in the last cycle, how "
    "many produced NO GRADIENT. A group is one prompt's rollout_n completions and the update is "
    "group-relative (advantage = reward minus the group's own mean), so a group whose rollouts "
    "all score THE SAME contributes nothing — whether they all failed or all succeeded. The "
    "breakdown `all_fail` / `all_succeed` ships with the total because the two have OPPOSITE "
    "remedies: all-fail means the instance is out of reach and text may buy a foothold; "
    "all-succeed means it is already solved and text buys nothing there. `unit` and "
    "`rollout_n_median` travel with the number so you can see what a group is here.\n"
    "- signals.valid_seen = the held-out standalone success (the objective). Read-only: you "
    "cannot see or optimize the final test set (unseen); this is the validation signal.\n"
    "- signals.eval_trajectory = that same held-out number across ALL cycles so far: `series` "
    "(step, value, and the individual `draws` behind that average), `deltas` (cycle-over-cycle "
    "change), `last_delta`, and `best`. The draws are repeats on the SAME weights, so their "
    "spread measures how much a delta can move by chance. All of it is measurement; what the "
    "shape implies is your call.\n"
)

_MECHANISM = (
    "THE INJECTION MECHANISM (what the system physically does with your text):\n"
    "- You attach TEXT to a SCOPE. During TRAINING ONLY, the text attached to an episode's "
    "scope is spliced into that episode's prompt with probability p (decided once per rollout "
    "group). At EVALUATION nothing is ever spliced in.\n"
    "- Scopes available in this domain: {scopes}.\n"
    "  'general' text is spliced into every episode; text on a category label is spliced only "
    "into episodes carrying that label. Text on both is spliced together.\n"
    "- HOW FAR A CHOICE REACHES. The fraction of training rows that carry your text is "
    "(rows in the chosen scope) x p. The per-category row counts are in the domain facts below, "
    "so this is arithmetic you can do before you propose. It is worth doing: one category at "
    "p=0.2 in a six-category set reaches about 3% of training, and even at p={p_max} reaches "
    "about 8%, while the same p on 'general' reaches six times that. A change that touches 3% of "
    "rows can still be the right change — but it will not move the held-out curve enough to be "
    "visible, so do not read a flat curve afterwards as evidence the text was wrong.\n"
    "- Scope breadth and content specificity are INDEPENDENT choices, and the guidance below to "
    "be concrete is about content, not about scope. Text can be concrete and still belong on "
    "'general' — an output contract, a shared convention, a boilerplate shape apply to "
    "every category and are checkable in {output_unit}. Put text on a category when it "
    "would be WRONG elsewhere, not merely when it was a category's rollouts that revealed it.\n"
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
    "- Where the domain says text can be attached to an individual instance, an item whose scope "
    "is `instance:<instance_id>` attaches to THAT instance only. It is spliced after the general "
    "and category text, so an instance receives all three. The instance's own injection still "
    "rides its category's p — attaching text to an instance whose category has p=0 changes "
    "nothing.\n"
    "- You may also intervene on nothing this cycle.\n"
)

_WHEN_TO_INTERVENE = (
    "WHEN TO INTERVENE (this section IS a directive, unlike the mechanism sections above):\n"
    "\n"
    "What injecting costs, every time it happens:\n"
    "- Context. The spliced block occupies prompt budget the policy would otherwise spend on "
    "the problem statement itself.\n"
    "- Exploration. An injected group is a group the policy did NOT explore on its own. You are "
    "spending its rollouts to see what YOUR text elicits instead of what the policy would have "
    "found. When the policy is already finding successes unaided, that trade is a loss.\n"
    # Deliberately says nothing about WHICH prompt the loss is conditioned on: that differs by
    # arm and is stated once, in the loss section, from the same variable the trainer reads. This
    # bullet used to assert the bare-prompt variant unconditionally — carried over from the
    # ALFWorld arm, which runs it — so on this arm the Teacher was told both that the loss is
    # computed on the bare prompt AND that it is computed on the prompt actually used.
    "- Transfer risk. The student is evaluated with the text absent, so behaviour it cannot "
    "reproduce without the text is gradient spent on something unreachable. How much of that "
    "risk you are running depends on the loss, described below.\n"
    "\n"
    "The benefit exists where the policy is STUCK — but 'stuck' is not one thing, and the "
    "situations listed above are the forms it takes. Where nothing is stuck, there is little to "
    "buy and the three costs above are still paid in full.\n"
    "\n"
    "Therefore:\n"
    "- The held-out curve is ONE reading of whether the policy is stuck, and on a small noisy "
    "eval it is often the weakest one available. A category that has been at or near zero for "
    "cycles is stuck whatever the total is doing; the total can keep climbing on the categories "
    "that are working while another sits dead. Do not require the total to flatten before acting "
    "on a category the per-category signals say is not learning — those are different questions "
    "and only one of them is answerable at this sample size.\n"
    "- A flat or regressing curve is a reason to look, not the only one. When it does flatten, "
    "the categories whose all-fail mass stayed high are where the remaining room is.\n"
    "- Read signals.eval_trajectory before acting. Each point carries the individual draws behind "
    "its average — repeated evaluations of the SAME weights — so the spread among them tells you "
    "directly how large a delta has to be before it means anything. If that spread is wider than "
    "any plausible per-cycle change, the curve cannot settle the question and the per-category "
    "signals have to.\n"
    "- Declining is cheap for one cycle and not cheap across a run. A run that ends with an empty "
    "scaffold has measured nothing about whether text helps, which is the question it exists to "
    "answer. Weigh a decline against that, not only against the cost of the measurement.\n"
    # The A/B mechanism itself is stated under HOW YOUR OUTPUT IS USED; only its bearing on the
    # decision to intervene belongs here.
    "- A wrong proposal costs a measurement, not a damaged run: the A/B discards it.\n"
    "{p_limits}"
)

def _p_limits_text():
    """The p rules AS CONFIGURED. Both are switchable per arm, so neither may be hard-coded into
    the prompt: an arm that pins the older rules would otherwise be told it is bound by limits it
    is not, and plan around a constraint that does not exist."""
    # States only what the caps IMPLY for planning. The caps themselves are in the mechanism
    # section; repeating the numbers here spent prompt budget saying the same thing twice.
    out = []
    if S.P_MAX_DELTA < S.P_MAX:
        out.append("- Reaching a high injection rate takes several cycles by design, so a "
                   "category you want to help needs a decision made early enough to ramp.\n")
    if p_gated_by_ab():
        out.append("- To let a p change stand on its own merits, propose it WITHOUT a text "
                   "edit; bundled with text, it dies with a failed A/B.\n")
    if not out:
        return ""
    return "\nWhat the p limits imply for planning:\n" + "".join(out)

_CONTENT_VOCAB = (
    "KINDS OF SCAFFOLD CONTENT (a vocabulary, NOT a recommendation — the text is free-form, so "
    "what you write into a scope is your choice). Each acts on a different reason a policy "
    "fails, and each fails in its own way; what follows is what they do and what they cost, not "
    "which to pick.\n"
    "Before the kinds, the situations. Text can only help by changing what the policy explores, "
    "and there is more than one way for that exploration to be the thing holding it back. These "
    "are different problems and the signals distinguish them:\n"
    "- No gradient at all. Every rollout in a group scores the same — usually all zero — so the "
    "update has nothing to work with. Anything that makes some of them succeed restores a signal "
    "that was absent.\n"
    "- Growth has stalled. The group has plenty of spread and the policy is still learning from "
    "it, but held-out success has stopped moving: the behaviour that would score higher is "
    "simply never sampled, so no amount of further training reaches it. Here text moves a "
    "ceiling, not a floor, and the failing rollouts may look no different from before.\n"
    "- Converged on a mode that pays but is not the goal. Rollouts score consistently and "
    "mediocrely; the gradient exists and is being followed, into a place the policy will not "
    "leave on its own. This is the hardest to see, because nothing looks broken — the scores are "
    "not zero and the curve is not obviously flat. Text here has to change what the policy is "
    "AIMING at, not whether it succeeds at what it currently aims at.\n"
    "The kinds below differ in which of these they can act on, in how forcefully, and in what "
    "the force costs.\n"
    "\n"
    "- skills: strategy or procedure guidance for a class of problem. Acts on a policy that has "
    "the capability but not the routine — the failures share a procedural cause. Costs nothing "
    "but prompt budget when it names a routine the policy already follows, which is the common "
    "way skills waste a cycle. What it can leave behind is a habit, so it transfers relatively "
    "well to the unscaffolded evaluation.\n"
    "\n"
    "{rubric_kind}"
    "- examples: a worked demonstration of a solved case. Acts on failures of FORM — structure, "
    "API use, boilerplate, output contract — where the policy knows the substance but not the "
    "shape. The most expensive kind in prompt budget, and the one most likely to be copied "
    "specifically rather than generalised. It is also the kind most liable to work against the "
    "first situation above: rollouts that all imitate one demonstration resemble each other more, "
    "which "
    "can flatten a group rather than spread it.\n"
    "{hints_kind}"
    "\n"
    "These are not modes to select, and nothing stops one text from doing several at once. "
    "Which is WORTH using is not stated anywhere — that is what the SIGNALS are for.\n"
)

# Spliced in only where the domain can actually carry it. A kind the mechanism cannot support is
# not worth describing: the previous version spent ~660 characters explaining hints and then, two
# paragraphs later, told the Teacher 'hints: NO'. Explaining a tool and then withdrawing it costs
# prompt budget and invites the Teacher to reason toward something it cannot propose.
_KIND_HINTS = (
    "\n"
    "- hints: a partial reveal of a known-good solution for a specific instance. Acts on one "
    "instance rather than a class, so its reach is small. What sets it apart is force: revealing "
    "part of the answer can make success nearly certain rather than merely more likely, so it is "
    "the surest way to restart a group where nothing succeeds — and the one whose cost is "
    "hardest to avoid. The side effects scale with that force — the policy is "
    "evaluated without the text, so the more of the answer a hint supplies, the more of what it "
    "learns is that answer rather than the reasoning that reaches it.\n"
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
    "instance at all. signals.zero_gradient_groups counts exactly those.\n"
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
    "   - signals.per_task_gap (injected minus bare, from the training rollouts) is the size of "
    "the crutch "
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


# The hint entry of the item-definition list, spliced in only where the domain can carry one.
# Listing it unconditionally described an operation `item_budget` does not offer here.
# The rubrics entry of the content vocabulary, spliced in only where the score comes from
# criteria the policy is never shown. Where reward is the task's own verifiable outcome the
# failure mode a rubric acts on does not arise, and describing the kind there offers an
# operation `item_budget` does not list.
CONTENT_RUBRICS = (
    "- rubrics: explicit criteria for what a good answer must satisfy. Acts on a different "
    "failure entirely — a policy producing acceptable work that scores badly, i.e. optimising "
    "something other than what it is graded on. Says what is being judged, not how to do it, so "
    "it is useless when the policy simply cannot do the thing: naming a standard does not supply "
    "the means, and stating criteria invites satisfying them literally. Worth knowing here that "
    "the student is scored against criteria it is never shown.\n"
    "\n"
)


ITEM_DEF_HINT = (
    "  * hint — ONE switch for the whole scaffold: partial reveal is on or off. When on, "
        "every injected row receives the leading `alpha` fraction of ITS OWN reference solution, "
        "and which rows are injected is decided by p, exactly as for the other kinds — so its "
        "reach is p, not a handful of hand-picked rows. It is the one kind whose CONTENT you do "
        "not write: the text is each row's own reference, and your only choice is `alpha`, how "
        "much of it to show. Add it at scope 'general' with an `alpha` in (0, 1]; change the "
        "amount with `update`, and turn it off with `delete`.\n"
)


def _content_availability(domain):
    """The `hints` paragraph, or nothing.

    skills / rubrics / examples are text the Teacher composes, so they are available wherever
    text is — saying so cost four lines that could not have come out any other way. hints is the
    one kind with a real precondition (a known-good solution must ship with the instance AND text
    must be attachable to that instance), so it is the only one worth gating. Gating means
    OMITTING it rather than describing it and then denying it: this domain answers NO, and the
    Teacher does not need a vocabulary entry it cannot use.
    """
    return _KIND_HINTS if (domain.has_reference_solutions and domain.instance_scope) else ""


_MEMORY = (
    "YOUR OWN MEMORY (decision_history):\n"
    "- decision_history is every cycle you have already decided, oldest first. Each entry holds "
    "what you proposed (summary.text_proposed, the item ops with their EXACT wording), what the "
    "A/B returned (`ab`), whether it was applied (`accepted_text`, `p_applied`, `verdict`), the "
    "held-out success measured just before you acted (`sr_before`) and the one measured after "
    "the next training stretch (`sr_after`, null until it exists).\n"
    "- The wording is kept verbatim for the most recent cycles and dropped from older ones to "
    "bound the prompt, so an old entry shows the outcome but not the text.\n"
    "- It is there so a wording the A/B already refused is not proposed again, and so a decline "
    "you already made for a stated reason is not re-derived from scratch. Nothing in it is an "
    "instruction; it is a record of what you did and what happened.\n"
    "- A cycle marked triage_declined never reached the measurement stage, so its entry carries "
    "a diagnosis and no A/B.\n"
)


# How behavioural evidence is shaped, and therefore what to call it. A multi-turn domain
# can pair a success and a failure of the SAME episode step by step; a single-turn one
# has one answer per attempt and nothing to walk through, so it gets the per-instance
# failure list instead. Describing the wrong one offers a field that is always empty.
TRACE_MULTI_TURN = (
    "- contrastive_traces[category].zero_gradient_failures = up to three GAMES of that "
    "category that the policy is failing, each shown as that group's longest trajectory — "
    "the one that used its whole step budget without arriving. Each step is {a: the command "
    "the environment executed, o: the environment's reply, v: false when the generation "
    "contained no parseable action}.\n"
    "  `group_success_rate` on each entry says how many of that group's rollouts succeeded, "
    "and `zero_gradient` is true when it is 0.0. A zero-gradient group has no reward "
    "variance, so GRPO produces no gradient from it at all — nothing in training is currently "
    "able to move it. Those are taken first. When a category has fewer than three of them, "
    "the remaining slots go to its LOWEST-success-rate groups instead, and a `note` says how "
    "many of each kind you are looking at.\n"
    "- contrastive_traces[category].successes_same_category = up to three of the SHORTEST "
    "successful trajectories the policy produced in that same category this cycle, on "
    "DIFFERENT games. They are not the same problem as the failures, so do not read them "
    "step against step; what they show is what arriving in this category looks like — the "
    "order things happen in, which command finally lands — against trajectories that never "
    "got there.\n"
    "- A category may carry `no_success_to_contrast`: the policy solved it zero times this "
    "cycle, so there is no successful trajectory of it to put beside the failures. "
    "`n_all_fail_groups` / `n_groups` give the scale behind the three shown — three traces "
    "out of forty stuck groups and three out of four mean different things.\n"
    "- failure_patterns[category] = failure modes counted by rule over EVERY failed rollout "
    "this cycle, not only the ones shown above: a command reissued three or more times and "
    "covering half the trajectory, an environment reply received three or more times, and the "
    "share of steps whose generation held no parseable action. These are counts, not "
    "impressions, and they cover rollouts the traces had no room for.\n"

)

TRACE_SINGLE_TURN = (
    "- failure_trajectories = the instances this cycle trained on that the bare policy "
    "scored worst, with their measured correct-rate over the group. One attempt is one "
    "answer, so there is no step sequence to walk through — what the list tells you is "
    "WHICH problems the policy is failing, not where inside an episode it went wrong.\n"
)


def kinds_for(domain=None):
    """Which item kinds THIS domain can carry, in a fixed order.

    Single source for every place the prompt enumerates kinds. Three separate places used to
    decide this independently — the content vocabulary, the item-definition list, and the JSON
    output spec — and each had to be gated by hand when a kind became conditional. The output
    spec was missed twice: it offered ALFWorld a `hint` (which needs reference solutions the
    domain has none of) and a `rubric` (which needs grading criteria the policy is not shown),
    both of which validate_item_ops then refuses. Offering an operation the system will reject
    spends the Teacher's cycle on a proposal that cannot land.
    """
    dom = domain or S.CUDA_DOMAIN
    out = []
    for k in ("rubric", "skill", "example", "hint"):
        if k == "rubric" and not dom.hidden_grading_criteria:
            continue
        if k == "hint" and not dom.has_reference_solutions:
            continue
        out.append(k)
    return out


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
        "being trained with reinforcement learning. Your job is to shape the text that "
        "is injected into its TRAINING prompts.\n\n"
        "DOMAIN STRUCTURE (facts, not advice):\n"
        + render_domain_facts(domain) + "\n\n"
        + _MECHANISM.format(scopes=domain.scopes(), p_max=S.P_MAX,
                            p_max_delta=S.P_MAX_DELTA,
                            output_unit=domain.output_unit) + "\n"
        + _CONTENT_VOCAB.format(
            hints_kind=_content_availability(domain),
            rubric_kind=(CONTENT_RUBRICS
                         if (domain or S.CUDA_DOMAIN).hidden_grading_criteria else ""))
        + "\n"
        + _OPTIMIZATION.format(
            loss_section=(_OPT_LOSS_BARE if bare_prompt_loss() else _OPT_LOSS_STANDARD)) + "\n"
        + _SIGNAL_MEANINGS.format(
            trace_shape=(TRACE_MULTI_TURN if domain.multi_turn else TRACE_SINGLE_TURN))
        + "\n"
        + _MEMORY + "\n"
        + _WHEN_TO_INTERVENE.format(p_limits=_p_limits_text()) + "\n"
        + (_PRIORS + "\n" if priors else "")
        + "HOW YOUR OUTPUT IS USED: text changes are accepted only if, on the current frozen "
        "policy, the new text beats the current text in a paired A/B on the HELD-OUT file — all "
        "three conditions scored on the same problems at the same checkpoint. p changes are "
        f"capped at {S.P_MAX}"
        + (f" and rate-limited to {S.P_MAX_DELTA} per cycle" if S.P_MAX_DELTA < S.P_MAX else "")
        + ("; p edits submitted together with text that then FAILS its A/B are discarded along "
           "with that text, so a p change you want judged on its own should be submitted without "
           "a text edit" if p_gated_by_ab() else "")
        + ". Nothing rewinds a regression: there is no revert, so a bad cycle stays in the "
        "curve. You are told this so you understand the loop — decide freely based on the "
        "signals.\n\n"
        "HOW TO WRITE THE TEXT. Be CONCISE. This is a real constraint, not a style note: the "
        "block you write is spliced on top of a training prompt that is already long, so every "
        "line of it displaces context the policy would otherwise spend on the problem itself. "
        "Write the shortest text that is still unambiguous, and cut anything the policy would "
        "have done anyway. A few precise lines beat a page.\n"
        "- The reader is a small model that already knows the vocabulary. Restating what it knows "
        "buys nothing; the text is only worth its prompt budget where it says something the "
        "policy is not already doing. Look at what the rollouts actually got wrong and address "
        "that, not the topic in general.\n"
        "- Prefer instructions whose effect you could check by reading " + domain.output_unit
        + ". An instruction that changes what the policy emits is worth its space; one that "
        "names a goal without naming an action is not.\n"
        "- Name the concrete thing: " + domain.concrete_nouns
        + ". Vague text is worse than no text — it costs prompt budget and the A/B gate cannot "
        "tell it apart from nothing, so it wastes the cycle it is measured in.\n"
        "- The policy is evaluated with this text ABSENT. Write what teaches a habit it can keep, "
        "not a crutch that only works while the text is there.\n\n"
        "\nTHE SCAFFOLD IS A SET OF ITEMS, NOT A BLOCK OF TEXT. Each scope holds separately "
        "addressable entries, each with a stable id (g3, conv7). You add, update or delete them "
        "one at a time; you never rewrite a scope wholesale. The entries of a scope are spliced "
        "together, in order, with a blank line between them.\n"
        "- WHAT COUNTS AS ONE ITEM depends on its kind, and the limits below follow from that:\n"
        + ("  * rubric — ONE criterion: something that can independently pass or fail. Rubrics "
           "are naturally a list, so several short ones is the right shape.\n"
           if (domain or S.CUDA_DOMAIN).hidden_grading_criteria else "")
        + 
        "  * skill — ONE rule: a trigger and the action it licenses. The test for whether you "
        "have written one or two: delete any single sentence and ask whether the rule still "
        "stands. If it does, that sentence was a second item and belongs in its own entry.\n"
        "  * example — ONE worked input-to-output demonstration. INDIVISIBLE: half a "
        "demonstration teaches nothing, so it stays one entry however long it runs. It is also "
        "the kind that consumes the prompt budget, which is why a scope may hold only one.\n"
        + (ITEM_DEF_HINT if (domain or S.CUDA_DOMAIN).has_reference_solutions else "")
        + "- Splitting matters beyond bookkeeping: an item is the smallest thing that can be "
        "deleted on its own. Two rules fused into one entry can only be removed together, so a "
        "wrong half drags a right half out with it.\n"
        "- The exact per-kind size and count limits, which kinds this domain can carry, and how "
        "many entries you may add or update THIS cycle, are in `item_budget` in the observation. "
        "An edit over budget is DROPPED and reported — the rest of your action still applies, so "
        "an over-long list costs you the surplus rather than the whole cycle.\n\n"
        "Return ONLY JSON: {\"diagnosis\": \"<your reasoning>\", "
        "\"item_ops\": ["
        "{\"op\": \"add\", \"scope\": \"<scope>\", \"kind\": \""
        + "|".join(k for k in kinds_for(domain) if k != "hint") + "\", "
        "\"text\": \"...\"}, "
        + ("{\"op\": \"add\", \"scope\": \"general\", \"kind\": \"hint\", \"alpha\": <0..1>}, "
           if "hint" in kinds_for(domain) else "")
        + 
        "{\"op\": \"update\", \"id\": \"<item id>\", \"text\": \"...\"}, "
        "{\"op\": \"delete\", \"id\": \"<item id>\"}], "
        "\"p_ops\": [{\"task\": \"<category>\", \"p\": <float 0..1>}]}. "
        "Empty item_ops and empty p_ops means you choose not to intervene this cycle."
    )


def _trim_traces(traces, keep_fn):
    """Drop one trace at a time, always from whichever category currently has the most.

    Trimming from the tail is what a naive budget cap does and it is wrong here: the traces
    arrive grouped by category, so a budget that fits two thirds of them deletes entire
    categories chosen by nothing but dict order. The Teacher then sees traces for the first
    few categories and cannot tell that from "those categories had none". Taking from the
    largest keeps every category represented for as long as the budget allows, and the
    weaker evidence (an unpaired failure) goes before a pair.
    """
    cur = {c: dict(v) for c, v in (traces or {}).items()}
    while not keep_fn(cur):
        sizes = {c: len(v.get("zero_gradient_failures") or [])
                    + len(v.get("successes_same_category") or [])
                 for c, v in cur.items()}
        if not sizes or max(sizes.values()) == 0:
            return {}                       # nothing left to give back; the caller reports it
        worst = max(sizes, key=lambda c: (sizes[c], c))
        v = cur[worst]
        # Successes go first. The zero-gradient failures are the thing the run is about; a
        # success is context for them, so if only one side can fit it is the failures.
        if v.get("successes_same_category"):
            v["successes_same_category"] = v["successes_same_category"][:-1]
        elif v.get("zero_gradient_failures"):
            v["zero_gradient_failures"] = v["zero_gradient_failures"][:-1]
    return cur


def _count_traces(traces):
    return {c: {"failures": len(v.get("zero_gradient_failures") or []),
                "successes": len(v.get("successes_same_category") or [])}
            for c, v in (traces or {}).items()}


def render_user_prompt(obs):
    """The measured observation packet, trimmed to a char budget on the heavy field.

    A trim is recorded in the packet itself (contrastive_traces_dropped): silently shrinking the
    Teacher's only view of real behaviour makes "this category had nothing to show" and
    "trimmed away" indistinguishable, and the Teacher reasons from that field.

    failure_patterns is never trimmed. It is a few hundred characters, it is counted over
    every rollout rather than the sampled ones, and it is the part that stays true when the
    traces do not fit.
    """
    body = json.dumps(obs, ensure_ascii=False, default=str)
    budget = _FAIL_CHARS + 40000
    if len(body) > budget and obs.get("contrastive_traces"):
        trimmed = dict(obs)
        before = _count_traces(obs["contrastive_traces"])
        kept = _trim_traces(
            obs["contrastive_traces"],
            lambda cand: len(json.dumps({**trimmed, "contrastive_traces": cand},
                                        ensure_ascii=False, default=str)) <= budget)
        trimmed["contrastive_traces"] = kept
        after = _count_traces(kept)
        if after != before:
            trimmed["contrastive_traces_dropped"] = {
                "note": "trimmed to a char budget, taken from the largest category each time; "
                        "absence here does NOT mean the category had nothing to show. "
                        "failure_patterns is complete and was counted over every rollout.",
                "before": before, "after": after,
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
    "valid_seen_per_task is whatever breakdown the held-out evaluation itself reports, and it is "
    "often not a per-category one — check its keys before reasoning about it. Where it does split "
    "by category, a total that has stopped moving can mean every category has run out of room, or "
    "that most have while one or two sit stuck well below the rest, and only the breakdown "
    "separates those; read `n_episodes` first, because a rate over a handful of episodes can read "
    "0.000 or 1.000 and mean nothing. Where it has a single key, it carries no category "
    "information at all and cannot answer that question — train_rollouts_by_category is then the "
    "only per-category evidence available, and it is the better-sampled one regardless.\n"
    "\n"
    "current_scaffold is what is live right now — the text at each scope and its injection "
    "probability. Read it before anything else: the same trajectory means different things "
    "depending on whether it was produced with text in play or with none.\n"
    "\n"
    "eval_trajectory gives each point's individual draws as well as its average. Those draws "
    "re-evaluate the SAME weights, so their spread is your measurement of how much a delta can "
    "move by chance alone — use it to decide what counts as flat. Note what follows if that "
    "spread is wide: a delta smaller than it can never be established, so 'the curve has not "
    "convincingly flattened' may be unfalsifiable on this run rather than evidence of progress. "
    "When the held-out signal cannot settle the question, decide on train_rollouts_by_category, "
    "which is sampled orders of magnitude better and does not depend on that curve at all.\n"
    "- zero_gradient_groups[category] counts the training rollout groups of the last cycle that "
    "produced no gradient, split into all_fail and all_succeed. A group is one instance's "
    "rollout_n completions, and the update is group-relative, so a group whose rollouts all score "
    "the same contributes nothing whatever that score is. A category can hold a moderate success "
    "rate while most of its instances are silent. The two causes point opposite ways: all_fail "
    "means the instances are out of reach, all_succeed means they are already solved.\n"
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
                                cycle=None, n_cycles=None, floor_cycles=None,
                                zero_gradient=None):
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
        # A COUNT, not the budget: this pre-check only decides whether to pay for a measurement,
        # and how many entries may change is a question for the cycle that actually proposes. What
        # it does need is how full the scaffold already is — 'nothing has been written yet' and
        # 'every scope is at capacity' are different reasons to answer the same question.
        "current_scaffold": {
            # Keys come from the scaffold's OWN items dict, not from scopes_of(scaffold) with no
            # domain — that call falls back to ALF_DOMAIN and listed ALFWorld task names to a
            # Teacher training on Triton kernels.
            "n_items_by_scope": {sc: len(v or [])
                                 for sc, v in (current_scaffold.get("items") or {}).items()},
            "general_skill": current_scaffold.get("general_skill", ""),
            "skills": current_scaffold.get("skills", {}),
            "p_task": current_scaffold.get("p_task", {}),
            "version": current_scaffold.get("version", 0),
        },
        "decision_history": compact_history(decision_history, recent=2),
    }
    if train_rollouts:
        obs["train_rollouts_by_category"] = train_rollouts
    # Zero-gradient groups are free (already computed from the cycle's training rollouts) and
    # answer a question the per-category rate cannot: an instance where every rollout scores the
    # same contributes nothing to the update whatever that score is. A category can hold a
    # moderate success rate while most of its instances are silent.
    if zero_gradient:
        obs["zero_gradient_groups"] = zero_gradient
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
