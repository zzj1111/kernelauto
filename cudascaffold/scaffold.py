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
import os

class Domain:
    """FACTS about a training domain's structure — never advice about what scaffold to use.

    The Teacher is shown this and must work out for itself what kind of scaffolding the
    structure affords: that labelled categories make per-category text possible, that
    reference solutions make partial-solution hints possible, and so on. We deliberately do
    NOT tell it "write one skill per category" — that inference is the point.
    """

    def __init__(self, name, episode_desc, categories=(), category_info=None,
                 action_primitives=(), has_reference_solutions=False, instance_scope=False,
                 extra_facts=()):
        self.name = name
        self.episode_desc = episode_desc              # what one episode/instance is, and how it is scored
        self.categories = list(categories)            # labels each instance carries ([] if unlabelled)
        self.category_info = dict(category_info or {})
        self.action_primitives = list(action_primitives)
        self.has_reference_solutions = has_reference_solutions   # is a known-good solution available?
        self.instance_scope = instance_scope          # can text be attached to a single instance?
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
# 4000 leaves real room to be specific while staying inside the prompt budget: max_prompt_length
# is 8192 and observed prompts peak at 2772 tokens, so even general + category + instance all at
# the cap (~3000 tokens) fits with margin.
MAX_TEXT_CHARS = int(os.environ.get("AUTOSCAFFOLD_MAX_TEXT_CHARS", "4000"))
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
        "The target GPU is an NVIDIA H200 (Hopper, sm_90).",
        "All 600 training rows have DISTINCT task_names, so an instance recurs only once per "
        "epoch — roughly two or three times across a 200-step run at batch 8. Per-instance text "
        "is mechanically available but reaches far less repetition than a category-level scope "
        "does; weigh that against its precision.",
        "The held-out set is 180 rows, 30 per category, drawn from the same pool with no "
        "reference implementation shared with training. Unlike the CUDA arm it covers EVERY "
        "category, so a gain confined to one category is visible in valid_seen rather than "
        "invisible.",
    ),
)

def empty_scaffold(domain=None):
    """Cold-start scaffold: 真空 — nothing injected. version 0."""
    cats = _cats(domain)
    return {
        "mode": "full",                       # SkillStore reads this; eval forces none
        "default_p": DEFAULT_P,
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
    """True when the Teacher chose not to intervene at all."""
    return not (action.get("text_ops") or action.get("p_ops") or action.get("prefix_ops"))


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
        "The target GPU is an NVIDIA H200 (Hopper, sm_90); its specifications are already stated "
        "in every prompt.",
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
)
