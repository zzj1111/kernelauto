"""Scaffold schema + Teacher action space for the auto-scaffold harness (ALFWorld).

The scaffold is the text injected DURING TRAINING only (evaluation is always
standalone). It is stored in the exact JSON shape ``SkillStore.from_json`` reads,
so the env-side injection path is unchanged:
``{mode, default_p, general_skill, skills, p_task, ...}``.

Design decisions locked with the user:
  - initial scaffold is EMPTY ("真空"): no general skill, per-category skills empty.
  - toolbox = general skill + per-category skill + per-category injection p + no-op.
    (NO per-instance skills, NO free bucketing — those belong to the math domain.)
  - the harness never decides WHAT to write; the Teacher does. This module only
    holds the structure, applies the Teacher's ops IMMUTABLY, and validates that an
    action is physically well-formed (never that its content is "good" — that is the
    A/B gate's job).
"""
from __future__ import annotations

import copy
import functools
import os
import subprocess

class Domain:
    """FACTS about a training domain's structure — never advice about what scaffold to use.

    The Teacher is shown this and must work out for itself what kind of scaffolding the
    structure affords: that labelled categories make per-category text possible, that
    reference solutions make partial-solution hints possible, and so on. We deliberately do
    NOT tell it "write one skill per category" — that inference is the point.
    """

    def __init__(self, name, episode_desc, categories=(), category_info=None,
                 action_primitives=(), has_reference_solutions=False, instance_scope=False,
                 output_unit="one rollout", concrete_nouns="the thing to do, the failure to avoid",
                 multi_turn=False,
                 extra_facts=(), hidden_grading_criteria=False):
        self.name = name
        self.episode_desc = episode_desc              # what one episode/instance is, and how it is scored
        self.categories = list(categories)            # labels each instance carries ([] if unlabelled)
        self.category_info = dict(category_info or {})
        self.action_primitives = list(action_primitives)
        # What one rollout produces, in the words of THIS domain, and the nouns a concrete
        # instruction in it would name. Both appear in the "how to write the text" guidance,
        # which is otherwise shared across domains — and shared guidance written in one domain's
        # vocabulary tells the other domain's Teacher to check something that does not exist
        # there. The ALFWorld prompt was telling it to verify its text by reading an emitted
        # CUDA kernel.
        # Whether one episode is a sequence of environment steps. It decides the SHAPE of the
        # behavioural evidence: a multi-turn domain can pair a success and a failure of the
        # same game step by step, a single-turn one has one answer per attempt and nothing to
        # walk through. Describing the wrong shape offers the Teacher a field that is always
        # empty — the defect this prompt has already had to correct for hints and rubrics.
        self.multi_turn = multi_turn
        self.output_unit = output_unit
        self.concrete_nouns = concrete_nouns
        self.has_reference_solutions = has_reference_solutions   # is a known-good solution available?
        self.instance_scope = instance_scope          # can text be attached to a single instance?
        # Is the score decided by criteria the policy is never shown — an LLM rubric, a human
        # preference model, an open-ended judgement? This gates the `rubric` content kind. A
        # rubric acts on a policy producing acceptable work that scores badly, which requires the
        # grading to be something other than the obvious objective. Where reward is the task's own
        # verifiable outcome (ALFWorld: did the goal state hold), that failure mode does not
        # exist, and a "rubric" there is a skill wearing a different label.
        self.hidden_grading_criteria = hidden_grading_criteria
        self.extra_facts = list(extra_facts)

    def scopes(self):
        """Injection scopes this domain actually supports (mechanism, not recommendation)."""
        s = [GENERAL] + list(self.categories)
        return s


# The 6 ALFWorld task types: the fixed category set the Teacher writes skills for.
TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# FACTUAL per-category descriptions (what the task IS — never how to win). Given to
# the Teacher as fixed context so it knows the taxonomy exists and can write per-type.
TASK_INFO = {
    "pick_and_place": "Find object X and put it in/on receptacle Y. Single object, no state change.",
    "pick_two_obj_and_place": "Find TWO objects of type X and put both in/on the same receptacle Y.",
    "look_at_obj_in_light": "Find object X, then use a desklamp to look at/examine it under the light.",
    "pick_heat_then_place_in_recep": "Find object X, heat it with the microwave, then put it in/on receptacle Y.",
    "pick_cool_then_place_in_recep": "Find object X, cool it with the fridge, then put it in/on receptacle Y.",
    "pick_clean_then_place_in_recep": "Find object X, clean it at the sinkbasin, then put it in/on receptacle Y.",
}

@functools.lru_cache(maxsize=1)
def _gpu_probe():
    """(name, compute_cap) of the TRAINING GPU, one cached nvidia-smi call per process.

    Cached because this file's two Domain literals both want it at import, and each driver
    query costs 100ms healthy and up to the 10s timeout on a busy node. Queries the first
    GPU the run will actually train on (ARM_GPUS, else CUDA_VISIBLE_DEVICES) rather than
    slot 0 — on a heterogeneous node the Teacher should be told about the training silicon,
    not whatever sits in the first PCIe slot. If the driver is wedged (D-state), set
    ARM_GPU_DESC to skip the subprocess entirely; _gpu_fact honors it before calling here.
    """
    first = (os.environ.get("ARM_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES")
             or "0").split(",")[0].strip()
    out = subprocess.run(
        ["nvidia-smi", "-i", first, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10)
    name, cap = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")[:2]]
    return name, cap


def _gpu_fact(suffix=""):
    """The domain-facts sentence naming the target GPU, detected from the driver.

    Was a hardcoded "H200 (Hopper, sm_90)" — harmless while every run WAS that machine, and
    wrong the moment the B200s arrive: the Teacher would write Hopper-specific guidance
    (tensor-core shapes, smem sizes) against Blackwell silicon. ARM_GPU_DESC overrides for
    cross-compile setups; detection failure falls back to naming nothing rather than lying.
    """
    desc = os.environ.get("ARM_GPU_DESC")
    if not desc:
        try:
            name, cap = _gpu_probe()
            fam = {"7": "Volta", "8": "Ampere/Ada", "9": "Hopper",
                   "10": "Blackwell", "12": "Blackwell"}.get(cap.split(".")[0], "")
            sm = "sm_" + cap.replace(".", "")
            desc = f"{name} ({fam + ', ' if fam else ''}{sm})"
        except Exception:
            return "The target GPU is the one this run's driver reports" + suffix + "."
    return f"The target GPU is an {desc}" + suffix + "."

# The action primitives that exist in ALFWorld (facts about the env, not advice).
ACTION_PRIMITIVES = [
    "go to <recep>", "open <recep>", "close <recep>", "take <obj> from <recep>",
    "put <obj> in/on <recep>", "use <obj>", "heat <obj> with microwave",
    "cool <obj> with fridge", "clean <obj> with sinkbasin", "look", "inventory",
]

# Injection probability is CAPPED. Always injecting is the documented failure mode: in a matched
# ablation on another domain, never withdrawing left standalone success BELOW the no-scaffold
# baseline (37.6 vs 42.9) while decaying p reached 45.9. The model has to spend training steps on
# the bare prompt it will actually face at evaluation, so a ceiling guarantees at least half the
# rollout groups are bare no matter what the Teacher proposes.
P_MAX = 0.5

# Cold start injects NOTHING. p is opt-in: the Teacher has to raise it deliberately, and that
# shows up in the decision record. Starting at the cap instead would mean the very first cycle
# trains half its groups on a scaffold nobody chose, and a scaffold that was never measured.
DEFAULT_P = 0.0

# Per-cycle ceiling on how far any category's p may move. P_MAX bounds the DESTINATION; this
# bounds the STEP. p is the unmeasured lever — text has to clear a paired A/B, p never does — so
# a single cycle must not be able to swing a category from fully withdrawn to the cap on one
# unverified judgment. Moving in bounded steps means every intermediate value gets a cycle of
# held-out evidence behind it before the next move. Applies to raises and cuts alike: an abrupt
# withdrawal is just as unmeasured as an abrupt injection.
P_MAX_DELTA = float(os.environ.get("AUTOSCAFFOLD_MAX_DP", "0.2"))

# Crash-guard cap only, never a quality judgment — and deliberately far above what the Teacher
# actually writes. At 1200 it was neither: offline probing showed proposals landing at 870-1000
# characters, so roughly half tipped over and were rejected, and rejection is whole-action (one
# long paragraph discards every other text_op and every p_op with it) reached only AFTER the
# measurement pass has been paid for. A cap that binds is a cap that silently edits the Teacher's
# judgment toward brevity, which is the opposite of what this scaffold needs: text that is vague
# enough to fit is text the A/B gate cannot distinguish from nothing.
#
# 2000 is a CRASH GUARD, not the length policy — brevity is asked for in the prompt instead
# (HOW TO WRITE THE TEXT), because exceeding this voids the whole action rather than truncating
# it, and the number is not disclosed, so a cap tight enough to bite would turn proposals into
# silent no-ops the Teacher could not diagnose. Observed proposals run 1450-1550 chars, so this
# sits above them: it stops a runaway, it does not shape the text. Prompt-budget headroom:
# max_prompt_length
# is 8192 and observed prompts peak at 2772 tokens, so even general + category + instance all at
# the cap (~3000 tokens) fits with margin.
MAX_TEXT_CHARS = int(os.environ.get("AUTOSCAFFOLD_MAX_TEXT_CHARS", "2000"))

# --- item budget -----------------------------------------------------------------------------
# The scaffold is a set of countable entries, so how fast it may change is a number rather than a
# matter of taste. Two limits, and they do different jobs:
#
#   CAPACITY (MAX_ITEMS_PER_SCOPE, MAX_ITEM_CHARS, MAX_TEXT_CHARS) bounds the STATE. It stops the
#   injected block from growing without limit, which matters because every character of it
#   displaces the problem statement in an 8192-token prompt.
#
#   PER-CYCLE BUDGET (BUDGET_*) bounds the CHANGE. The A/B compares the whole set at once, so a
#   cycle that rewrites five entries produces one verdict on five changes and cannot say which one
#   earned it. Holding the per-cycle change small is what keeps that single verdict interpretable.
#
# Over capacity, an add must be paid for with a delete (the rule PATS enforces at its COMPRESS
# band). Deletes are never rate-limited: removing text can always be undone by re-adding it, and
# there is no revert gate here, so the cheap direction should stay open.
# Caps are PER KIND, because the four kinds are not the same unit and a single number silently
# picks a winner. What "one item" means, and the test for whether something is one or two:
#
#   rubric  — one criterion. Test: it can independently pass or fail. Naturally a list, so many
#             short entries are the correct shape.
#   skill   — one rule: a trigger plus the action it licenses. Test: delete any one sentence and
#             ask whether the rule still stands; if it does, that sentence was a second item.
#   example — one worked input->output demonstration. INDIVISIBLE: half an example teaches
#             nothing, so it is one item however long it runs. It is also the kind that eats the
#             prompt budget, which is why at most one is allowed per scope — that is the real
#             trade-off the observation already describes, now enforced rather than just stated.
#   hint    — ONE item for the whole scaffold: partial reveal is either ON or OFF. When on, every
#             injected row gets the leading `alpha` fraction of ITS OWN reference solution, and
#             which rows are injected is decided by p exactly as for every other kind. So its
#             reach is p, the same lever as the rest, and it can move the aggregate.
#             The earlier design made it per-instance and opt-in, which capped it at a couple of
#             rows per cycle (2 of 600 = 0.3% of training) — a setting in which it could never
#             have moved anything. It is also the one kind whose CONTENT the Teacher does not
#             write: the text is each row's own reference, and the only decision is `alpha`, how
#             much of it to show.
#
# A uniform 600-char cap (the first version of this) would have rejected every example outright
# while the prompt went on offering examples as an option — the same defect as describing `hints`
# in a domain that cannot carry them.
# `budget_change` counts ADDS AND UPDATES together, not adds alone. The A/B returns one verdict
# on the whole diff, so what has to stay small is the SIZE OF THE DIFF — and rewriting an entry
# changes the injected text exactly as much as adding one. Separate add/update budgets let a
# cycle move 3+2 = 5 entries behind a single verdict while neither number said 5.
ITEM_KINDS = {
    "rubric":  {"max_chars": 200,  "max_per_scope": 8, "budget_change": 3},
    "skill":   {"max_chars": 400,  "max_per_scope": 6, "budget_change": 2},
    "example": {"max_chars": 1500, "max_per_scope": 1, "budget_change": 1},
    # One switch for the whole scaffold: at most one, and at most one add per cycle. max_chars is
    # irrelevant (the Teacher writes no text for it) but kept so the schema is uniform.
    "hint":    {"max_chars": 600,  "max_per_scope": 1, "budget_change": 1},
}
DEFAULT_KIND = "skill"
# Total CHANGES (adds + updates) across all kinds in one cycle, on top of the per-kind caps.
# Deletes are excluded and never limited: they only ever shrink what is injected, and with no
# revert gate the direction that removes text has to stay open. A delete still has to win the
# A/B like anything else, so it is not a way around the verdict.
BUDGET_CHANGES = int(os.environ.get("AUTOSCAFFOLD_BUDGET_CHANGES", "3"))
# Scope-level count cap, independent of kind: even all-rubric, a scope stays readable.
MAX_ITEMS_PER_SCOPE = int(os.environ.get("AUTOSCAFFOLD_MAX_ITEMS", "8"))
MAX_ITEM_CHARS = max(k["max_chars"] for k in ITEM_KINDS.values())


def kind_cap(kind, field):
    return ITEM_KINDS.get(kind, ITEM_KINDS[DEFAULT_KIND])[field]

GENERAL = "general"            # the target name for the shared general skill
# Prefix that turns a text_op target into a PER-INSTANCE one: `instance:27_RegNet`.
# Namespaced so an instance can never collide with a category label.
INSTANCE_PREFIX = "instance:"
MAX_INSTANCE_ENTRIES = 40      # crash guard: the prompt carries them back, so cap the count


ALF_DOMAIN = Domain(
    name="alfworld",
    episode_desc=("One episode places the agent in a household and gives it a goal in natural "
                  "language. The agent emits text actions for up to 50 steps. Reward is 1 if the "
                  "goal state is reached, else 0."),
    categories=TASKS,
    category_info=TASK_INFO,
    action_primitives=ACTION_PRIMITIVES,
    multi_turn=True,
    output_unit="one episode transcript",
    concrete_nouns="the receptacle, the object, the ordering constraint, the failure to avoid",
    has_reference_solutions=False,   # ALFWorld episodes ship no solution trace we expose here
    instance_scope=False,            # individual games are not addressable as scaffold targets
    extra_facts=("Every episode carries exactly one of the category labels above.",
                 "An action outside the primitive set returns 'Nothing happens'."),
)


def _cats(domain):
    """Category labels of the active domain. Defaults to ALFWorld so every existing call
    site keeps its exact behaviour; the math domain passes its own subject labels."""
    return list((domain or ALF_DOMAIN).categories)



# ---------------------------------------------------------------------------------------------
# Triton domain: hkust-nlp/drkernel-rl-data, downsampled by dataset/CudaForge/build_triton_split.py
# ---------------------------------------------------------------------------------------------

# Categories are operator FAMILIES, not difficulty levels — this dataset's `level` is 0 on all
# 71,996 rows, so the CUDA set's scratch/improve_l* axis does not exist here. The family axis is
# also the one that has actually predicted outcomes so far: fusion-friendly work behaves very
# differently from work backed by a tuned library kernel.
TRITON_CATEGORIES = ["elementwise", "reduce", "norm_softmax", "matmul", "conv", "loss"]

TRITON_CATEGORY_INFO = {
    "elementwise": "The heaviest operator is elementwise/unary/binary — activations, clamps, "
                   "arithmetic chains. Nothing here is library-bound, so the whole chain can be "
                   "fused into one kernel. (100 of the 600 training rows.)",
    "reduce": "The heaviest operator reduces along an axis — sum/mean/max/argmax/cumsum/var. "
              "(100 rows.)",
    "norm_softmax": "The heaviest operator is a normalisation or softmax — layer/batch/group "
                    "norm, softmax, logsoftmax. These have a reduce-then-apply structure. "
                    "(100 rows.)",
    "matmul": "The heaviest operator is a matmul/bmm/linear/einsum, i.e. work cuBLAS is already "
              "tuned for. (100 rows.)",
    "conv": "The heaviest operator is a convolution, including transposed and depthwise, i.e. "
            "work cuDNN is already tuned for. (100 rows.)",
    "loss": "The heaviest operator is a loss or divergence — MSE, smooth L1, KL, cosine "
            "similarity. Usually a reduction over a small elementwise expression. (100 rows.)",
}

# What the student may write. Not a recommendation — the mechanical surface it acts on.
TRITON_PRIMITIVES = [
    "a @triton.jit kernel using triton.language (tl.program_id, tl.arange, tl.load/tl.store "
    "with masks, tl.sum/tl.max, tl.dot), launched with a grid from the Python side",
    "a class ModelNew(nn.Module) mirroring the reference module's inputs and outputs",
]

TRITON_DOMAIN = Domain(
    name="triton_kernel",
    episode_desc=(
        "One instance gives a PyTorch reference module and asks for a drop-in replacement that "
        "computes the same thing using custom Triton kernels. The answer is imported, run against "
        "the reference on the same inputs, and timed. Reward is 0 unless the output matches the "
        "reference within tolerance; given correctness it scales with measured speedup over the "
        "reference (clipped at 5x) and is multiplied by an LLM rubric score. Reward is forced to 0 "
        "if the rubric flags reward hacking — returning the reference renamed, defining a "
        "@triton.jit kernel that is never launched or whose result is discarded, or using "
        "torch.compile in place of writing a kernel."),
    categories=TRITON_CATEGORIES,
    category_info=TRITON_CATEGORY_INFO,
    action_primitives=TRITON_PRIMITIVES,
    output_unit="one emitted kernel",
    concrete_nouns="the operator, the layout, the launch shape, the failure to avoid",
    # `extra_info.answer` is the PyTorch reference to be BEATEN, not a fast kernel to reveal.
    has_reference_solutions=False,
    # Mechanically available — every row carries a task_name — but see the reach fact below.
    instance_scope=True,
    extra_facts=(
        "Every instance carries exactly one of the category labels above, assigned by its "
        "heaviest operator. Most instances also contain lighter elementwise operators alongside "
        "it; the label names the dominant one, not the only one.",
        "Each instance fuses 2 to 8 PyTorch operators, 3 at the median, drawn from a vocabulary "
        "of 233 distinct operators across the 600 training rows.",
        "Given correctness, the score is driven by measured speedup over the PyTorch reference "
        "(clipped at 5x) and then multiplied by the rubric. Correctness is therefore a gate, not "
        "the objective: a category can have a healthy correct_rate and still be contributing "
        "almost nothing.",
        "Input sizes vary enormously — the median instance touches about 14,000 elements, the "
        "10th percentile 128 and the 90th 3.1 million. On the small ones a Triton launch can cost "
        "more than the PyTorch operator it replaces, so a genuine kernel there may measure SLOWER "
        "than leaving the work to PyTorch. The rubric, not the speedup term, is what separates a "
        "real kernel from an answer that avoided writing one.",
        "Compilation failure, an incorrect result, and a timeout are all scored 0 — they are "
        "indistinguishable in the reward.",
        _gpu_fact(),
        "All 600 training rows have DISTINCT task_names, so an instance recurs only once per "
        "epoch — roughly two or three times across a 200-step run at batch 8. Per-instance text "
        "is mechanically available but reaches far less repetition than a category-level scope "
        "does; weigh that against its precision.",
        "The held-out set is 180 rows, 30 per category, drawn from the same pool with no "
        "reference implementation shared with training. Unlike the CUDA arm it covers EVERY "
        "category, so a gain confined to one category is visible in valid_seen rather than "
        "invisible.",
    ),
    hidden_grading_criteria=True,   # reward is multiplied by an LLM rubric score
)

def empty_scaffold(domain=None):
    """Cold-start scaffold: 真空 — nothing injected. version 0."""
    cats = _cats(domain)
    return {
        "mode": "full",                       # SkillStore reads this; eval forces none
        "default_p": DEFAULT_P,
        # The scaffold IS `items`: a list of separately-addressable entries per scope, each with
        # a stable id. `general_skill` and `skills[cat]` below are the RENDERED form of those
        # lists and are kept in sync by _sync_text — every reader (splice.render_block, has_text,
        # the observation) goes on reading strings and needs no change.
        #
        # Why items rather than one blob per scope. A blob can only be replaced, so the Teacher
        # rewrote all of it every cycle, nothing could be attributed to a part of it, and there
        # was no unit to count — which meant no way to budget how much may change per cycle, and
        # no way to remove one bad sentence without discarding the good ones next to it.
        "items": {s: [] for s in ([GENERAL] + list(cats))},
        "next_item_n": 1,                     # monotonic; ids are never reused after a delete
        "general_skill": "",
        "skills": {t: "" for t in cats},
        "p_task": {t: DEFAULT_P for t in cats},
        "version": 0,
        "history": [],                        # decision->outcome log (loop appends)
        # Per-instance disclosure of a KNOWN-GOOD solution, for domains that have one.
        # Empty (and unused) unless domain.instance_scope and domain.has_reference_solutions.
        "alpha": {},                          # {instance_uid: fraction of the reference solution}
        # Free-form text attached to ONE instance, keyed by the instance id the domain uses
        # (task_name here). Distinct from `alpha`: that discloses part of a known-good solution
        # and needs domain.has_reference_solutions; this is the Teacher's own words about one
        # task and needs only domain.instance_scope. Viable here and not on the earlier arms
        # because the training set is ~400 fixed rows over ~150 tasks, so an annotated instance
        # is seen again many times instead of never.
        "instances": {},                      # {instance_id: text}
        "default_alpha": 0.0,                 # nothing disclosed unless the Teacher says so
    }


def injects_nothing(scaffold):
    """True when this scaffold cannot change any prompt.

    SkillStore.render() returns "" if the task text and general_skill are both blank, and
    splice_skill() returns the prompt unchanged on an empty block — so such a scaffold is
    token-for-token identical to no scaffold at all. Worth naming: it lets the A/B gate skip
    a redundant arm instead of paying for a second noisy sample of the same condition.
    """
    if not scaffold:
        return True
    if (scaffold.get("general_skill") or "").strip():
        return False
    return not any((v or "").strip() for v in (scaffold.get("skills") or {}).values())


# --------------------------------------------------------------------------- #
# Teacher action = {diagnosis, text_ops, p_ops}
#   text_ops: [{"target": "general"|<task>, "text": str}]  -> gated by the A/B door
#   p_ops:    [{"task": <task>, "p": float}]               -> applied directly (anchor governs)
#   both empty                                             -> no-op (do nothing this cycle)
# --------------------------------------------------------------------------- #

def validate_action(action, domain=None):
    """PHYSICAL well-formedness only (no content/quality judgment). -> (ok, reason).
    Valid text targets are 'general' plus the ACTIVE DOMAIN's category labels — hardcoding
    ALFWorld's six here silently turned every well-formed math proposal into a no-op."""
    cats = _cats(domain)
    if not isinstance(action, dict):
        return False, "action is not an object"
    text_ops = action.get("text_ops", [])
    p_ops = action.get("p_ops", [])
    if not isinstance(text_ops, list) or not isinstance(p_ops, list):
        return False, "text_ops/p_ops must be lists"
    for op in text_ops:
        if not isinstance(op, dict):
            return False, "text_op is not an object"
        tgt = op.get("target")
        if isinstance(tgt, str) and tgt.startswith(INSTANCE_PREFIX):
            if not (domain or ALF_DOMAIN).instance_scope:
                return False, (f"text_op target '{tgt}' is per-instance, but this domain cannot "
                               "attach text to a single instance")
            if not tgt[len(INSTANCE_PREFIX):].strip():
                return False, f"text_op target '{tgt}' names no instance"
        elif tgt != GENERAL and tgt not in cats:
            return False, f"text_op target '{tgt}' is not '{GENERAL}', a category, or "\
                          f"'{INSTANCE_PREFIX}<instance_id>'"
        txt = op.get("text")
        if not isinstance(txt, str) or not txt.strip():
            return False, f"text_op for '{tgt}' has empty/non-string text"
        if len(txt) > MAX_TEXT_CHARS:
            return False, f"text_op for '{tgt}' exceeds {MAX_TEXT_CHARS} chars"
    prefix_ops = action.get("prefix_ops", [])
    if not isinstance(prefix_ops, list):
        return False, "prefix_ops must be a list"
    if prefix_ops:
        dom = domain or ALF_DOMAIN
        # Never accept an op the mechanism cannot carry out: ALFWorld has no reference
        # solution and no per-instance slot, so a prefix op there is meaningless.
        if not (dom.instance_scope and dom.has_reference_solutions):
            return False, (f"domain '{dom.name}' has no per-instance reference-solution slot; "
                           "prefix_ops are not available here")
        for op in prefix_ops:
            if not isinstance(op, dict) or not str(op.get("uid", "")).strip():
                return False, "prefix_op needs a non-empty 'uid'"
            try:
                a = float(op.get("alpha"))
            except (TypeError, ValueError):
                return False, f"prefix_op alpha '{op.get('alpha')}' is not a number"
            if not (0.0 <= a <= 1.0):
                return False, f"prefix_op alpha={a} out of [0,1]"

    for op in p_ops:
        if not isinstance(op, dict):
            return False, "p_op is not an object"
        if op.get("task") not in cats:
            return False, f"p_op task '{op.get('task')}' is not a known category"
        p = op.get("p")
        try:
            p = float(p)
        except (TypeError, ValueError):
            return False, f"p_op p '{p}' is not a number"
        if not (0.0 <= p <= 1.0):   # out of range is malformed; above P_MAX is clamped, not rejected
            return False, f"p_op p={p} out of [0,1]"
    return True, "ok"


def is_noop(action):
    """True when the Teacher chose not to intervene at all.

    `item_ops` MUST be listed here: it is now the only way to edit text, so omitting it made
    every real proposal read as a decline — the loop would take the noop branch and the A/B would
    never run. `text_ops`/`prefix_ops` stay for actions built the older way.
    """
    return not (action.get("item_ops") or action.get("text_ops")
                or action.get("p_ops") or action.get("prefix_ops"))


def has_text(scaffold):
    """True when the scaffold would actually inject something at any scope.

    Injection probability alone does not count: p>0 over empty text still splices nothing, so a
    scaffold with p set and no wording has not begun to answer whether text helps. Used by the
    intervention floor, which exists precisely to stop a run from ending in that state.
    """
    if (scaffold.get("general_skill") or "").strip():
        return True
    for v in (scaffold.get("skills") or {}).values():
        t = v if isinstance(v, str) else (v or {}).get("text", "")
        if (t or "").strip():
            return True
    for v in (scaffold.get("instances") or {}).values():
        if str(v or "").strip():
            return True
    return False


def apply_prefix_ops(scaffold, prefix_ops):
    """Return a NEW scaffold with per-instance disclosure levels applied. Never mutates.

    alpha == 0 removes the entry rather than storing a zero, so `alpha` stays a record of
    what is CURRENTLY disclosed and does not grow by one dead key per withdrawal.
    """
    nxt = copy.deepcopy(scaffold)
    alpha = dict(nxt.get("alpha") or {})
    for op in prefix_ops:
        a = float(op["alpha"])
        if a <= 0.0:
            alpha.pop(op["uid"], None)
        else:
            alpha[op["uid"]] = a
    nxt["alpha"] = alpha
    if prefix_ops:
        nxt["version"] = scaffold.get("version", 0) + 1
    return nxt


def touched_tasks(text_ops, domain=None, instance_categories=None):
    """Categories whose injected text changes under these text_ops.
    A 'general' edit touches ALL categories; a per-category edit touches that one."""
    tasks = set()
    for op in text_ops:
        tgt = op["target"]
        if tgt == GENERAL:
            tasks.update(_cats(domain))
        elif tgt.startswith(INSTANCE_PREFIX):
            # An instance edit changes what a few rows of some category see. Which category is a
            # data question the caller answers via `instance_categories`; absent that, fall back
            # to every category so the A/B cannot miss the one that actually moved.
            tasks.update((instance_categories or {}).get(tgt[len(INSTANCE_PREFIX):],
                                                         _cats(domain)))
        else:
            tasks.add(tgt)
    return sorted(tasks)


def apply_text_ops(scaffold, text_ops):
    """Return a NEW scaffold with the text edits applied (version+1). Never mutates."""
    nxt = copy.deepcopy(scaffold)
    inst = dict(nxt.get("instances") or {})
    for op in text_ops:
        tgt = op["target"]
        if tgt == GENERAL:
            nxt["general_skill"] = op["text"].strip()
        elif tgt.startswith(INSTANCE_PREFIX):
            key = tgt[len(INSTANCE_PREFIX):].strip()
            txt = op["text"].strip()
            if txt:
                inst[key] = txt
            else:
                inst.pop(key, None)     # empty text removes the entry rather than storing ""
        else:
            nxt["skills"][tgt] = op["text"].strip()
    # Oldest-first eviction keeps the scaffold (and the prompt that echoes it) bounded; the
    # Teacher is told the cap so a silent drop cannot look like an edit that did not take.
    if len(inst) > MAX_INSTANCE_ENTRIES:
        for k in list(inst)[:len(inst) - MAX_INSTANCE_ENTRIES]:
            inst.pop(k)
    nxt["instances"] = inst
    nxt["version"] = scaffold.get("version", 0) + 1
    return nxt


# ==============================================================================================
# Items: the scaffold as a set of separately-addressable entries.
#
# `items[scope]` is the source of truth; `general_skill` / `skills[cat]` are its rendering and are
# rebuilt by _sync_text after every change, so nothing downstream had to learn about items.
# ==============================================================================================


def scopes_of(scaffold, domain=None):
    """Scopes that can hold items: 'general', the domain's categories, and any instance scope
    that already holds one. Instance scopes are discovered from the scaffold rather than listed,
    because the set of them is the training set, not a fixed vocabulary."""
    fixed = [GENERAL] + list(_cats(domain))
    live = [k for k in (scaffold.get("items") or {}) if str(k).startswith(INSTANCE_PREFIX)]
    return fixed + sorted(live)


def migrate_items(scaffold, domain=None):
    """Return a scaffold that HAS an items list, deriving one from the legacy strings if needed.

    Runs on every load. A run resumed from a state.json written before items existed would
    otherwise show the Teacher an empty item list next to non-empty injected text — it would
    then 'add' what is already being injected. Each legacy string becomes one item, which is
    exactly what it was: a scope's entire text as a single indivisible entry.
    """
    nxt = copy.deepcopy(scaffold)
    if isinstance(nxt.get("items"), dict):
        for s in scopes_of(nxt, domain):
            nxt["items"].setdefault(s, [])
        nxt.setdefault("next_item_n", 1 + sum(len(v) for v in nxt["items"].values()))
        return _sync_text(nxt, domain)
    items, n = {}, 1
    for s in scopes_of(nxt, domain):
        txt = (nxt.get("general_skill") if s == GENERAL
               else (nxt.get("skills") or {}).get(s))
        txt = txt if isinstance(txt, str) else (txt or {}).get("text", "")
        items[s] = []
        if (txt or "").strip():
            items[s].append({"id": _mk_id(s, n), "scope": s, "kind": DEFAULT_KIND,
                             "text": txt.strip(), "step": 0})
            n += 1
    nxt["items"], nxt["next_item_n"] = items, n
    return _sync_text(nxt, domain)


def _mk_id(scope, n):
    """Short, stable, human-readable: g3, conv7. The Teacher addresses items by these."""
    return f"{'g' if scope == GENERAL else scope}{n}"


def items_of(scaffold, scope):
    return list((scaffold.get("items") or {}).get(scope) or [])


def render_scope(scaffold, scope):
    """A scope's items as the one string that gets spliced. Blank line between entries so the
    policy reads them as separate rules rather than one run-on paragraph."""
    return "\n\n".join(i["text"].strip() for i in items_of(scaffold, scope) if i.get("text", "").strip())


def _sync_text(scaffold, domain=None):
    """Rebuild the legacy rendered fields from `items`. Mutates and returns `scaffold`."""
    scaffold["general_skill"] = render_scope(scaffold, GENERAL)
    skills = dict(scaffold.get("skills") or {})
    for c in _cats(domain):
        skills[c] = render_scope(scaffold, c)
    scaffold["skills"] = skills
    return scaffold


def find_item(scaffold, item_id):
    """-> (scope, index) or (None, None)."""
    for scope, lst in (scaffold.get("items") or {}).items():
        for i, it in enumerate(lst or []):
            if it.get("id") == item_id:
                return scope, i
    return None, None


def cycle_budget(scaffold, domain=None):
    """What the Teacher may change THIS cycle, as counts it can plan against.

    `must_delete_to_add` fires per scope that is already at MAX_ITEMS_PER_SCOPE: the add is only
    admitted if the same action also deletes from that scope. This is the only pressure toward
    removing anything — without it a scope fills once and then can never change again except by
    update, and a wrong entry is permanent (there is no revert gate).
    """
    full = [s for s in scopes_of(scaffold, domain)
            if len(items_of(scaffold, s)) >= MAX_ITEMS_PER_SCOPE]
    dom = domain or ALF_DOMAIN
    kinds = {}
    for k, spec in ITEM_KINDS.items():
        if k == "hint" and not (dom.has_reference_solutions and dom.instance_scope):
            continue                              # not offered where it cannot be carried
        if k == "rubric" and not dom.hidden_grading_criteria:
            continue                              # nothing to state that the task does not state
        kinds[k] = {
            "max_chars": spec["max_chars"],
            "max_per_scope": spec["max_per_scope"],
            "max_changes_per_cycle": spec["budget_change"],
            "scopes_at_kind_capacity": [
                s for s in scopes_of(scaffold, domain)
                if sum(1 for i in items_of(scaffold, s)
                       if (i.get("kind") or DEFAULT_KIND) == k) >= spec["max_per_scope"]],
        }
    return {
        "max_changes": BUDGET_CHANGES,            # adds + updates together
        "max_delete": None,                       # deletes are never rate-limited
        "max_items_per_scope": MAX_ITEMS_PER_SCOPE,
        "max_scope_chars": MAX_TEXT_CHARS,
        "scopes_at_capacity": full,
        "by_kind": kinds,
    }


def validate_item_ops(item_ops, scaffold, domain=None):
    """STRUCTURAL well-formedness only -> (ok, reason). Budget is NOT checked here.

    The split is deliberate. A malformed op means the Teacher misunderstood the interface, and
    the whole action is refused so the mistake is loud. Exceeding a budget means it wanted more
    than it may have this cycle, which is ordinary; apply_item_ops trims the excess and reports
    it, because refusing the action outright would throw away the edits that DID fit and turn an
    over-eager cycle into a silent no-op.
    """
    if not isinstance(item_ops, list):
        return False, "item_ops must be a list"
    scopes = set(scopes_of(scaffold, domain))
    for op in item_ops:
        if not isinstance(op, dict):
            return False, "item_op is not an object"
        kind = op.get("op")
        if kind not in ("add", "update", "delete"):
            return False, f"item_op op '{kind}' is not add/update/delete"
        if kind == "add":
            sc = op.get("scope")
            # An instance scope is not in `scopes` because it is not a fixed label — it names one
            # row of the training set, so the valid set is the dataset, not a constant.
            if isinstance(sc, str) and sc.startswith(INSTANCE_PREFIX):
                if not (domain or ALF_DOMAIN).instance_scope:
                    return False, (f"add scope '{sc}' is per-instance, but this domain cannot "
                                   "attach text to a single instance")
                if not sc[len(INSTANCE_PREFIX):].strip():
                    return False, f"add scope '{sc}' names no instance"
            elif sc not in scopes:
                return False, (f"add scope '{sc}' is not {sorted(scopes)} or "
                               f"'{INSTANCE_PREFIX}<instance_id>'")
        else:
            if not isinstance(op.get("id"), str) or not op["id"].strip():
                return False, f"{kind} needs an existing item id"
            if find_item(scaffold, op["id"])[0] is None:
                return False, f"{kind} names unknown item id '{op['id']}'"
        if kind in ("add", "update"):
            k = op.get("kind") or DEFAULT_KIND
            if k not in ITEM_KINDS:
                return False, f"{kind} kind '{k}' is not {sorted(ITEM_KINDS)}"
            # A kind the domain cannot carry is refused here rather than silently stored: a hint
            # reveals part of a KNOWN-GOOD solution, so without reference solutions it would only
            # be the Teacher's own guess wearing the label.
            dom = domain or ALF_DOMAIN
            # A rubric states what the grader is looking for. That only differs from the task
            # itself where the score comes from criteria the policy is never shown. Where reward
            # is the task's own verifiable outcome, a "rubric" is a skill wearing another label.
            if k == "rubric" and not dom.hidden_grading_criteria:
                return False, ("kind 'rubric' needs grading criteria the policy is not shown; "
                               f"domain '{dom.name}' scores the task's own verifiable outcome, "
                               "so state the guidance as a skill instead")
            if k == "hint":
                if not dom.has_reference_solutions:
                    return False, ("kind 'hint' reveals part of a KNOWN-GOOD solution, and "
                                   f"domain '{dom.name}' ships none")
                sc = op.get("scope") if kind == "add" else find_item(scaffold, op["id"])[0]
                if sc != GENERAL:
                    return False, (f"kind 'hint' is one switch for the whole scaffold, so its "
                                   f"scope is '{GENERAL}', not '{sc}'")
                try:
                    a = float(op.get("alpha"))
                except (TypeError, ValueError):
                    return False, f"kind 'hint' needs alpha (how much to reveal), got {op.get('alpha')!r}"
                if not (0.0 < a <= 1.0):
                    return False, f"hint alpha={a} must be in (0, 1]"
            txt = op.get("text")
            if k == "hint":
                continue        # its content is each row's own reference; only alpha is a choice
            if not isinstance(txt, str) or not txt.strip():
                return False, f"{kind} has empty/non-string text"
            # A verbatim duplicate is refused. The Teacher sees its own recent proposals and can
            # re-propose one that already landed; nothing downstream would notice, and the scope
            # would render the same sentence twice to the policy while spending budget on it.
            # Compared after whitespace normalisation so re-wrapping does not read as new text.
            if kind == "add":
                sc = op.get("scope")
                norm = " ".join(txt.split())
                if any(" ".join((it.get("text") or "").split()) == norm
                       for it in items_of(scaffold, sc)):
                    return False, (f"add duplicates text already in scope '{sc}'; "
                                   f"update or delete that item instead")
            cap = kind_cap(k, "max_chars")
            if len(txt) > cap:
                return False, f"{kind} of kind '{k}' exceeds {cap} chars (that kind's limit)"
    return True, "ok"


def apply_item_ops(scaffold, item_ops, domain=None, step=0):
    """Return (new_scaffold, notes). Never mutates. Applies deletes first, then updates, then
    adds — so an action that deletes to make room for an add works in one cycle regardless of the
    order the Teacher listed them in. Trims to budget and records exactly what it dropped."""
    nxt = migrate_items(scaffold, domain)
    b = cycle_budget(nxt, domain)
    notes, n_chg = [], 0
    n_chg_kind = {}
    order = {"delete": 0, "update": 1, "add": 2}
    freed = {}
    for op in sorted(item_ops, key=lambda o: order.get(o.get("op"), 3)):
        kind = op["op"]
        if kind == "delete":
            scope, idx = find_item(nxt, op["id"])
            if scope is None:
                notes.append(f"delete {op['id']}: already gone")
                continue
            nxt["items"][scope].pop(idx)
            freed[scope] = freed.get(scope, 0) + 1
        elif kind == "update":
            if n_chg >= b["max_changes"]:
                notes.append(f"update {op['id']}: over the change budget "
                             f"({b['max_changes']}/cycle), dropped")
                continue
            scope, idx = find_item(nxt, op["id"])
            if scope is None:
                notes.append(f"update {op['id']}: no longer exists, dropped")
                continue
            uk = op.get("kind") or nxt["items"][scope][idx].get("kind") or DEFAULT_KIND
            if n_chg_kind.get(uk, 0) >= kind_cap(uk, "budget_change"):
                notes.append(f"update {op['id']}: over the per-kind budget "
                             f"({kind_cap(uk, 'budget_change')} {uk}/cycle), dropped")
                continue
            nxt["items"][scope][idx] = {**nxt["items"][scope][idx],
                                        **({"alpha": float(op["alpha"])} if "alpha" in op else {}),
                                        "text": (op.get("text") or "").strip(), "step": step,
                                        "kind": op.get("kind")
                                        or nxt["items"][scope][idx].get("kind") or DEFAULT_KIND}
            n_chg += 1
            n_chg_kind[uk] = n_chg_kind.get(uk, 0) + 1
        else:
            scope, k = op["scope"], (op.get("kind") or DEFAULT_KIND)
            if n_chg >= b["max_changes"]:
                notes.append(f"add to {scope}: over the change budget "
                             f"({b['max_changes']}/cycle), dropped")
                continue
            if n_chg_kind.get(k, 0) >= kind_cap(k, "budget_change"):
                notes.append(f"add {k} to {scope}: over the per-kind budget "
                             f"({kind_cap(k, 'budget_change')} {k}/cycle), dropped")
                continue
            here = items_of(nxt, scope)
            # Verbatim duplicate. validate_item_ops already refuses these, but apply is also
            # reached directly (loop.py calls it on an already-validated action, and a resumed
            # journal can replay one), and a duplicate that slipped through would render the same
            # sentence twice to the policy for the rest of the run.
            _norm = " ".join((op.get("text") or "").split())
            if _norm and any(" ".join((i.get("text") or "").split()) == _norm for i in here):
                notes.append(f"add to {scope}: duplicates text already there, dropped")
                continue
            if sum(1 for i in here if (i.get("kind") or DEFAULT_KIND) == k) >= kind_cap(k, "max_per_scope"):
                notes.append(f"add {k} to {scope}: that scope already holds "
                             f"{kind_cap(k, 'max_per_scope')} of kind '{k}', dropped")
                continue
            if len(here) >= MAX_ITEMS_PER_SCOPE:
                notes.append(f"add to {scope}: at capacity ({MAX_ITEMS_PER_SCOPE}) and this "
                             f"action deleted {freed.get(scope, 0)} there, dropped")
                continue
            n = int(nxt.get("next_item_n", 1))
            entry = {"id": _mk_id(scope, n), "scope": scope, "kind": k,
                     "text": (op.get("text") or "").strip(), "step": step}
            if k == "hint":
                entry["alpha"] = float(op["alpha"])
            nxt["items"].setdefault(scope, []).append(entry)
            nxt["next_item_n"] = n + 1
            n_chg += 1
            n_chg_kind[k] = n_chg_kind.get(k, 0) + 1
    # Instance scopes are unbounded in number, unlike the fixed category set, so the count of
    # them is capped as well. Oldest-first here rather than newest-first: an instance scope holds
    # at most a couple of entries and what matters is how many INSTANCES carry text, so the ones
    # written longest ago are the ones whose instance the policy has most likely moved past.
    inst = [sc for sc in nxt["items"] if str(sc).startswith(INSTANCE_PREFIX)]
    if len(inst) > MAX_INSTANCE_ENTRIES:
        def _age(sc):
            its = nxt["items"].get(sc) or []
            return min((int(i.get("step", 0)) for i in its), default=0)
        for sc in sorted(inst, key=_age)[:len(inst) - MAX_INSTANCE_ENTRIES]:
            nxt["items"].pop(sc, None)
            notes.append(f"{sc}: over the {MAX_INSTANCE_ENTRIES}-instance cap, dropped")

    # Scope-level cap on the RENDERED text, enforced after the fact because it is a property of
    # the set rather than of any one entry. Drop newest-first: the older entries have already
    # survived an A/B, the newest has not.
    for scope in scopes_of(nxt, domain):
        while len(render_scope(nxt, scope)) > MAX_TEXT_CHARS and nxt["items"].get(scope):
            dropped = nxt["items"][scope].pop()
            notes.append(f"{scope}: over {MAX_TEXT_CHARS} rendered chars, dropped {dropped['id']}")
    _sync_text(nxt, domain)
    nxt["version"] = scaffold.get("version", 0) + 1
    return nxt, notes


def touched_scopes(item_ops, scaffold, domain=None):
    """Scopes whose injected text changes under these ops. 'general' touches every category,
    because general text is spliced into every row — the A/B has to score them all."""
    cats = set(_cats(domain))
    out = set()
    for op in item_ops or []:
        scope = op.get("scope") if op.get("op") == "add" else find_item(scaffold, op.get("id"))[0]
        if scope == GENERAL:
            return set(cats)
        if scope in cats:
            out.add(scope)
    return out


def clamp_p(scaffold):
    """Return a NEW scaffold with every injection probability pulled down to P_MAX.

    Applied to whatever the Teacher proposes AND to any scaffold loaded from disk, so a run that
    started before the cap, or a Teacher that ignores it, still cannot inject on more than half
    the rollout groups. Clamping is recorded rather than silent: a p the Teacher asked for and did
    not get is exactly the kind of thing that must not vanish from the record.
    """
    nxt = copy.deepcopy(scaffold)
    clamped = {}
    for t, v in list((nxt.get("p_task") or {}).items()):
        if float(v) > P_MAX:
            clamped[t] = float(v)
            nxt["p_task"][t] = P_MAX
    if float(nxt.get("default_p", DEFAULT_P)) > P_MAX:
        clamped["__default__"] = float(nxt["default_p"])
        nxt["default_p"] = P_MAX
    if clamped:
        nxt["p_clamped"] = {"cap": P_MAX, "requested": clamped}
    return nxt


def apply_p_ops(scaffold, p_ops):
    """Return a NEW scaffold with the injection-probability edits applied. Never mutates.

    Two independent limits, both properties of the experiment rather than suggestions the
    Teacher can override:
      - P_MAX_DELTA bounds the STEP: |new - old| per cycle, applied first.
      - P_MAX bounds the DESTINATION (see clamp_p), applied after.
    Both record what they limited (`p_rate_limited` / `p_clamped`) so a decision that was
    trimmed is visible in the scaffold rather than silently different from what was asked.
    """
    nxt = copy.deepcopy(scaffold)
    default_p = float(nxt.get("default_p", DEFAULT_P))
    limited = {}
    for op in p_ops:
        task = op["task"]
        old = float(nxt["p_task"].get(task, default_p))
        want = float(op["p"])
        new = max(old - P_MAX_DELTA, min(old + P_MAX_DELTA, want))
        if new != want:
            limited[task] = {"from": old, "requested": want, "applied": new}
        nxt["p_task"][task] = new
    nxt = clamp_p(nxt)
    if limited:
        nxt["p_rate_limited"] = {"max_delta": P_MAX_DELTA, "ops": limited}
    else:
        nxt.pop("p_rate_limited", None)
    if p_ops:
        nxt["version"] = scaffold.get("version", 0) + 1
    return nxt


def validate_scaffold(scaffold, domain=None):
    """Physical structural check before writing to disk. -> (ok, reason)."""
    cats = _cats(domain)
    if not isinstance(scaffold, dict):
        return False, "scaffold is not an object"
    if scaffold.get("mode") not in ("full", "none"):
        return False, f"mode '{scaffold.get('mode')}' invalid"
    try:
        dp = float(scaffold.get("default_p", DEFAULT_P))
    except (TypeError, ValueError):
        return False, "default_p is not a number"
    if not (0.0 <= dp <= 1.0):
        return False, f"default_p={dp} out of [0,1]"
    if not isinstance(scaffold.get("general_skill", ""), str):
        return False, "general_skill is not a string"
    skills = scaffold.get("skills", {})
    p_task = scaffold.get("p_task", {})
    for t in cats:
        if t not in skills:
            return False, f"skills missing category '{t}'"
        if not isinstance(skills[t], str):
            return False, f"skills['{t}'] is not a string"
        try:
            p = float(p_task.get(t, dp))
        except (TypeError, ValueError):
            return False, f"p_task['{t}'] is not a number"
        if not (0.0 <= p <= 1.0):
            return False, f"p_task['{t}']={p} out of [0,1]"
    return True, "ok"


# ---------------------------------------------------------------- CUDA domain #
# KernelBench difficulty tiers, the one label every row of train_new.parquet carries.
# Chosen as the category set because it is the only stratification that (a) is present on
# every instance and (b) plausibly changes what advice is useful: a level-1 single operator
# and a level-3 whole architecture fail for different reasons.
# Categories must partition the data: EVERY row has to fall in exactly one, or the rows that
# fall in none can only ever receive the general skill. `level` alone does not partition it —
# only the 200 improvement rows carry a level; the 200 from-scratch rows carry none. So the
# category set crosses the task mode with the level where the level exists.
CUDA_LEVELS = ["scratch", "improve_l1", "improve_l2", "improve_l3"]

# Counts are those of train_new_clean.parquet. They were 200/80/79/41 over 400 rows until the
# split was rebuilt on 2026-08-03: 44 improvement rows turned out to be trajectories for problems
# in the TEST split, mislabelled `split="train"` in the source data, and were dropped. Stating a
# count the data does not have is the same class of defect as naming a signal field that does not
# exist — see tests/test_alignment.py.
CUDA_LEVEL_INFO = {
    "scratch": "Write a CUDA kernel for a PyTorch reference module from scratch; no prior "
               "attempt is supplied. (data_source=CudaForge, 200 of the 356 training rows.)",
    "improve_l1": "Improve a supplied kernel for a SINGLE PyTorch operator. (61 rows.)",
    "improve_l2": "Improve a supplied kernel for a small fused operator sequence, e.g. "
                  "conv+bias+relu. (63 rows.)",
    "improve_l3": "Improve a supplied kernel for a FULL model architecture — many operators, "
                  "fusion opportunities across layers. (32 rows.)",
}

# What the student may write. Not a recommendation — the mechanical surface it acts on.
CUDA_PRIMITIVES = [
    "inline CUDA source compiled with torch.utils.cpp_extension.load_inline",
    "a class ModelNew(nn.Module) mirroring the reference module's inputs and outputs",
]

CUDA_DOMAIN = Domain(
    name="cuda_kernel",
    episode_desc=(
        "One instance gives a PyTorch reference module and asks for a drop-in replacement that "
        "computes the same thing using custom CUDA kernels. The answer is compiled with nvcc, run "
        "against the reference on the same inputs, and timed. Reward is 0 unless the output matches "
        "the reference within tolerance; given correctness it scales with measured speedup over the "
        "reference (clipped at 5x) and is multiplied by an LLM rubric score. Reward is forced to 0 "
        "if the rubric flags reward hacking, e.g. a kernel that is compiled but never called, or "
        "leaving the heavy operator in PyTorch."),
    categories=CUDA_LEVELS,
    category_info=CUDA_LEVEL_INFO,
    action_primitives=CUDA_PRIMITIVES,
    output_unit="one emitted kernel",
    concrete_nouns="the operator, the layout, the launch shape, the failure to avoid",
    # No known-good CUDA solution ships with an instance. `extra_info.answer` is the PyTorch
    # REFERENCE to be beaten, not a fast kernel to reveal, so partial-solution hints are not
    # available here any more than they were on ALFWorld.
    has_reference_solutions=False,
    # Unlike the ALFWorld and Search-R1 arms, the training set is ~400 fixed rows over ~150
    # distinct task_names, so every instance recurs many times during a run. Text attached to a
    # single instance would actually be seen again.
    instance_scope=True,
    extra_facts=(
        "Every instance carries exactly one of the category labels above.",
        "The improvement categories supply the previous kernel and whether it was correct; the "
        "scratch category supplies only the PyTorch reference.",
        "About half of the improvement instances start from a kernel that was itself incorrect.",
        "Compilation failure, an incorrect result, and a timeout are all scored 0 — they are "
        "indistinguishable in the reward.",
        "Given correctness, the score is driven by measured speedup over the PyTorch reference "
        "(clipped at 5x) and then multiplied by the rubric. Correctness is therefore a gate, not "
        "the objective: a correct kernel that matches PyTorch's speed and a correct kernel that "
        "beats it 5x differ by roughly a factor of four in reward, while a correct-but-not-faster "
        "kernel and an incorrect one differ by far less. A category can have a healthy "
        "correct_rate and still be contributing almost nothing.",
        "The held-out set is 50 rows, ALL of them the scratch category, drawn from KernelBench "
        "problems that appear nowhere in training. So valid_seen carries no per-category "
        "breakdown — its only key is the data_source — and a gain confined to the improvement "
        "categories does not appear in it directly. It does NOT follow that the improvement "
        "categories are worth nothing: RL updates one policy, and the gradient from improvement "
        "rollouts passes through the same weights that write scratch answers, so a category that "
        "starts learning can raise scratch success as well. What this observation cannot do is "
        "attribute a move in valid_seen to the category that caused it. The improvement "
        "categories have an effect you cannot measure directly — which is not the same as an "
        "effect of zero, and reasoning as if it were is how 44% of the training data ends up "
        "written off.",
        _gpu_fact("; its specifications are already stated in every prompt"),
        "An instance is identified by its task_name (e.g. '27_RegNet'). About 120 distinct "
        "task_names cover the 156 improvement rows, so most instances appear once or twice per "
        "epoch rather than many times — per-instance text reaches a narrow slice. The 200 scratch "
        "rows carry NO task_name and therefore cannot be targeted individually; per-instance text "
        "reaches the improvement categories only.",
        "The rubric that multiplies the reward uses a DIFFERENT criteria set per category, each "
        "scored 1-5. For the scratch category: anti_hacking (no fake speedups), "
        "bottleneck_coverage (the kernel addresses the actual hot path), cuda_perf_quality "
        "(memory access, occupancy, use of the Hopper tensor cores), multi_component_focus "
        "(fusing across operators rather than optimising one in isolation). For the improvement "
        "categories: anti_hacking, bottleneck_effectiveness (the change actually removes the "
        "bottleneck), instruction_alignment (the answer follows the optimisation instructions "
        "present in its own prompt), optimization_scope_focus. Both sets also emit a "
        "major_hacking flag that forces the reward to 0. These criteria are fixed and you "
        "cannot edit them; they are stated because the student is scored against them and is "
        "never shown them.",
        "Note what instruction_alignment implies for the improvement categories: the judge "
        "checks whether the answer followed the optimisation instructions in its prompt, and "
        "text you inject becomes part of that prompt.",
    ),
    hidden_grading_criteria=True,   # reward is multiplied by an LLM rubric score
)
