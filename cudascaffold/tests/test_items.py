"""The scaffold as a set of addressable items, and the budget that bounds how fast it changes.

These assert the CLAIMS the design rests on, not the implementation:
  - an item is the smallest thing that can be deleted on its own;
  - the four kinds are different units, so their limits differ;
  - a hint is one switch for the whole scaffold, reaching rows through p like everything else;
  - the budget TRIMS (structural errors reject), so an over-eager cycle is not a lost cycle;
  - every path that reads the action knows about item_ops.

The last one is here because omitting it was the most dangerous bug of the refactor: is_noop did
not list item_ops, so every real proposal read as a decline and the A/B would never have run.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cudascaffold import loop as L  # noqa: E402
from cudascaffold import scaffold as S  # noqa: E402
from cudascaffold import splice as SP  # noqa: E402

D = S.TRITON_DOMAIN
# A domain that CAN carry hints, to test the kind rather than this dataset's lack of references.
HINTABLE = S.Domain("probe", "one kernel", categories=["a", "b"],
                    has_reference_solutions=True, instance_scope=True)


def _add(scaf, scope, kind, text, domain=D, step=1):
    return S.apply_item_ops(scaf, [{"op": "add", "scope": scope, "kind": kind, "text": text}],
                            domain, step=step)[0]


# ---------- an item is the unit of deletion ----------

def test_deleting_one_item_leaves_the_others_rendering():
    scaf = S.empty_scaffold(D)
    scaf = _add(scaf, "conv", "skill", "first rule")
    scaf = _add(scaf, "conv", "skill", "second rule")
    assert SP.render_block(scaf, "conv") == "first rule\n\nsecond rule"
    victim = S.items_of(scaf, "conv")[0]["id"]
    scaf = S.apply_item_ops(scaf, [{"op": "delete", "id": victim}], D)[0]
    assert SP.render_block(scaf, "conv") == "second rule", "the survivor must still be injected"
    assert scaf["skills"]["conv"] == "second rule", "the rendered cache must follow the items"


def test_ids_are_never_reused_after_a_delete():
    """A reused id would silently redirect an update aimed at the entry that used to hold it."""
    scaf = _add(S.empty_scaffold(D), "loss", "skill", "one")
    gone = S.items_of(scaf, "loss")[0]["id"]
    scaf = S.apply_item_ops(scaf, [{"op": "delete", "id": gone}], D)[0]
    scaf = _add(scaf, "loss", "skill", "two")
    assert S.items_of(scaf, "loss")[0]["id"] != gone


# ---------- the kinds are different units ----------

def test_each_kind_has_the_size_its_definition_implies():
    scaf = S.empty_scaffold(D)
    for kind, fits, over in (("rubric", 200, 201), ("skill", 400, 401), ("example", 1500, 1501)):
        assert S.validate_item_ops(
            [{"op": "add", "scope": "conv", "kind": kind, "text": "x" * fits}], scaf, D)[0]
        assert not S.validate_item_ops(
            [{"op": "add", "scope": "conv", "kind": kind, "text": "x" * over}], scaf, D)[0], (
            f"{kind} must be capped at its own limit, not a shared one")


def test_an_example_is_indivisible_so_a_scope_holds_one():
    """A uniform cap would have rejected every example while the prompt kept offering them."""
    assert S.ITEM_KINDS["example"]["max_per_scope"] == 1
    assert S.ITEM_KINDS["example"]["max_chars"] > S.ITEM_KINDS["skill"]["max_chars"]


# ---------- hint: one switch, reaching rows through p ----------

def test_hint_is_one_switch_at_general_carrying_a_fraction():
    scaf = S.empty_scaffold(HINTABLE)
    assert not S.validate_item_ops(
        [{"op": "add", "scope": "a", "kind": "hint", "alpha": 0.5}], scaf, HINTABLE)[0], \
        "a hint is scaffold-wide, not per category"
    assert not S.validate_item_ops(
        [{"op": "add", "scope": "general", "kind": "hint"}], scaf, HINTABLE)[0], \
        "alpha is the whole decision, so it cannot be omitted"
    for bad in (0.0, 1.5, -0.2):
        assert not S.validate_item_ops(
            [{"op": "add", "scope": "general", "kind": "hint", "alpha": bad}], scaf, HINTABLE)[0]
    scaf = S.apply_item_ops(
        scaf, [{"op": "add", "scope": "general", "kind": "hint", "alpha": 0.4}], HINTABLE)[0]
    assert SP.hint_alpha(scaf) == 0.4
    scaf2 = S.apply_item_ops(
        scaf, [{"op": "add", "scope": "general", "kind": "hint", "alpha": 0.9}], HINTABLE)[0]
    assert sum(1 for i in S.items_of(scaf2, "general") if i["kind"] == "hint") == 1


def test_hint_actually_reaches_the_prompt():
    """alpha used to be stored and read by nothing: the Teacher could set it, the record showed it
    set, and not one character reached a prompt."""
    scaf = S.apply_item_ops(S.empty_scaffold(HINTABLE),
                            [{"op": "add", "scope": "general", "kind": "hint", "alpha": 0.4}],
                            HINTABLE)[0]
    ref = "\n".join(f"line{i}" for i in range(10))
    block = SP.render_block(scaf, "a", None, ref)
    assert "line0" in block and "line3" in block, "the revealed prefix must be present"
    assert "line9" not in block, "40% of 10 lines is 4, not all of them"
    assert "withheld" in block, "the policy must be able to tell a partial reveal from a whole one"
    assert SP.render_block(S.empty_scaffold(HINTABLE), "a", None, ref) == "", \
        "no hint item -> nothing revealed, even though a reference exists"


def test_hint_is_refused_where_no_known_good_solution_ships():
    assert not S.validate_item_ops(
        [{"op": "add", "scope": "general", "kind": "hint", "alpha": 0.5}],
        S.empty_scaffold(D), D)[0], "Triton ships a PyTorch reference, not a known-good kernel"
    assert "hint" not in S.cycle_budget(S.empty_scaffold(D), D)["by_kind"], \
        "a kind the domain cannot carry must not be offered in the budget either"


# ---------- budget trims, structure rejects ----------

def test_over_budget_trims_and_reports_rather_than_voiding_the_cycle():
    scaf = S.empty_scaffold(D)
    ops = [{"op": "add", "scope": "loss", "kind": "rubric", "text": f"R{i}"} for i in range(5)]
    assert S.validate_item_ops(ops, scaf, D)[0], "over budget is not a structural error"
    scaf, notes = S.apply_item_ops(scaf, ops, D)
    assert len(S.items_of(scaf, "loss")) == S.BUDGET_CHANGES
    assert any("budget" in n for n in notes), "a silent drop would read as an edit that did nothing"


def test_a_malformed_op_rejects_the_whole_action():
    scaf = S.empty_scaffold(D)
    for bad in ([{"op": "nope"}],
                [{"op": "add", "scope": "nosuch", "kind": "skill", "text": "x"}],
                [{"op": "update", "id": "does-not-exist", "text": "x"}],
                [{"op": "add", "scope": "conv", "kind": "invented", "text": "x"}]):
        assert not S.validate_item_ops(bad, scaf, D)[0]


def test_deletes_are_never_rate_limited():
    """There is no revert gate, so the cheap direction stays open."""
    scaf = S.empty_scaffold(D)
    for i in range(4):
        scaf = _add(scaf, "matmul", "skill", f"r{i}", step=i)
    ids = [i["id"] for i in S.items_of(scaf, "matmul")]
    scaf = S.apply_item_ops(scaf, [{"op": "delete", "id": i} for i in ids], D)[0]
    assert S.items_of(scaf, "matmul") == []


def test_an_add_can_be_paid_for_with_a_delete_in_the_same_action():
    scaf = S.empty_scaffold(D)
    for i in range(S.MAX_ITEMS_PER_SCOPE):
        scaf = S.apply_item_ops(
            scaf, [{"op": "add", "scope": "reduce", "kind": "rubric", "text": f"r{i}"}], D)[0]
    assert len(S.items_of(scaf, "reduce")) == S.MAX_ITEMS_PER_SCOPE
    victim = S.items_of(scaf, "reduce")[0]["id"]
    # Listed add-first on purpose: ordering inside one action must not decide whether it fits.
    scaf2, _ = S.apply_item_ops(scaf, [{"op": "add", "scope": "reduce", "kind": "rubric",
                                        "text": "new"},
                                       {"op": "delete", "id": victim}], D)
    ids = [i["id"] for i in S.items_of(scaf2, "reduce")]
    assert victim not in ids and len(ids) == S.MAX_ITEMS_PER_SCOPE


# ---------- every reader of the action knows about item_ops ----------

def test_an_action_with_only_item_ops_is_not_a_noop():
    """The refactor's most dangerous failure: is_noop omitting item_ops made every real proposal
    read as a decline, so the loop took the noop branch and the A/B never ran."""
    assert not S.is_noop({"item_ops": [{"op": "add", "scope": "conv", "kind": "skill",
                                        "text": "x"}], "p_ops": []})
    assert S.is_noop({"item_ops": [], "p_ops": []})


def test_the_record_keeps_every_proposal_separately():
    """compact_history feeds this back as the Teacher's memory. Keyed by op:scope it collided —
    three rubrics added to one scope left one entry, and the Teacher would re-propose the rest."""
    action = {"diagnosis": "d", "p_ops": [],
              "item_ops": [{"op": "add", "scope": "loss", "kind": "rubric", "text": f"R{i}"}
                           for i in range(3)]}
    proposed = L._summary(action)["text_proposed"]
    assert len(proposed) == 3
    assert {p["text"] for p in proposed} == {"R0", "R1", "R2"}


# ---------- resume ----------

def test_a_pre_items_scaffold_migrates_without_changing_what_is_injected():
    legacy = {"mode": "full", "default_p": 0.0, "general_skill": "OLD G", "version": 7,
              "skills": {c: ("OLD " + c if c == "conv" else "") for c in D.categories},
              "p_task": {c: 0.2 for c in D.categories}, "instances": {}, "alpha": {}}
    m = S.migrate_items(legacy, D)
    assert m["general_skill"] == "OLD G" and m["skills"]["conv"] == "OLD conv", \
        "migration must not change one character of what the policy reads"
    assert len(S.items_of(m, "general")) == 1 and len(S.items_of(m, "conv")) == 1
    assert S.items_of(m, "loss") == [], "an empty scope migrates to no items, not to one empty one"
    assert S.migrate_items(m, D)["items"] == m["items"], "migration must be idempotent"


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS {name}")
        except AssertionError as e:
            fails.append(name)
            print(f"  FAIL {name}: {e}")
    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


def test_every_state_field_the_loop_writes_survives_a_restart():
    """loop.py is shared with the ALFWorld arm; STATE_KEYS is not. A field the shared loop starts
    writing is persisted by whichever arm was edited and silently reset by the other on every
    restart — and the watchdog restarts routinely, so the loss is production behaviour, not an
    edge case. This arm was already carrying train_rollouts when the other had lost it."""
    import os
    import re
    from cudascaffold import loop as L, run_arm as R

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "loop.py")).read()
    written = set(re.findall(r'state\[\s*["\']([a-z_]+)["\']\s*\]\s*=', src))
    written |= set(re.findall(r'state\.setdefault\(\s*["\']([a-z_]+)["\']', src))
    lost = sorted(written - set(R.STATE_KEYS))
    assert not lost, f"loop.py writes {lost}; STATE_KEYS does not persist them"
    fresh = set(L.new_state(0))
    assert fresh == set(R.STATE_KEYS), (
        f"only in new_state: {sorted(fresh - set(R.STATE_KEYS))}, "
        f"only in STATE_KEYS: {sorted(set(R.STATE_KEYS) - fresh)}")


def test_the_trainer_loads_data_in_process():
    """verl's default of 8 dataloader workers forks the trainer eight more times.

    The trainer here is an 8B model plus a vLLM engine; a two-step smoke reached 174 GB of host
    RSS and had a worker killed by signal. The dataset is one file of short prompts, so the
    workers buy nothing and cost a fork each. The sibling ALFWorld arm carries the same setting
    after an OOM kill during a checkpoint save took a node with it.
    """
    import os as _os
    from cudascaffold import adapters as A
    cfg = {"train_batch_size": 8, "max_prompt_length": 2048, "max_response_length": 8192,
           "model": "/m", "lora_rank": 128, "lora_alpha": 128, "lr": 1e-6, "kl": 0.03,
           "kl_loss_coef": 0.03, "val_file": "/v.parquet", "ckpt_root": "/c", "exp": "e",
           "n_gpus": 2, "tp": 2, "tp_size": 2, "gpu_mem": 0.35, "mini_bs": 4, "micro_bs": 1,
           "ppo_mini_batch_size": 4, "rollout_n": 6, "project": "p", "reward_path": "/r.py",
           "steps_per_cycle": 10, "total_epochs": 20}
    cmd = A._train_cmd(cfg, "/t.parquet", 10)
    assert "data.dataloader_num_workers=0" in cmd, (
        "the trainer would fork 8 dataloader workers of an 8B-model process")
    assert "+data.dataloader_num_workers" not in cmd, (
        "this key already exists in the config; a leading plus makes hydra refuse to start")
