"""Guard against the failure mode that keeps recurring: the prompt and the machinery disagreeing.

Every defect this file was written for shares one shape — the Teacher was told something that was
not true of the system it was operating, and nothing crashed. It reasoned correctly about a
fiction. Code tests cannot catch that, because the code was doing what it was written to do.

The ones found so far:
  * per_task_gap described as a per-category breakdown while keyed by data_source, so it shared
    no key with any scope the Teacher could edit;
  * all_fail_groups named as the zero-gradient signal while always empty;
  * the triage prompt teaching the Teacher to find a stuck category in valid_seen_per_task, which
    in the CUDA arm has a single key;
  * the two prompts giving opposite directives — triage counting a local stall while the total
    climbs, propose saying to wait for the total to flatten;
  * splice's anchor hard-coded to one dataset's wording, so 56% of CudaForge rows had the
    scaffold appended AFTER the output contract instead of before it;
  * category derived from `level`, which is 0 on every Triton row, yielding "improve_l0" — a
    label in no domain, so nothing was ever injected;
  * the A/B gate falling back to the overall mean when its per-category lookup missed, judging a
    one-category proposal on a mean five sixths of which could not have changed.

The domain-parametrised tests run against BOTH arms, because every one of the above was found in
one arm after being fixed in the other.
"""
from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, "/mnt/data1/zha00175/StitchCUDA")

import pandas as pd  # noqa: E402

from cudascaffold import gates as G, observation as O, scaffold as S, splice as SP  # noqa: E402

TRI = O.render_triage_prompt()

# (domain, train parquet, held-out parquet). Held-out may be None where the arm has none yet.
ARMS = [
    (S.CUDA_DOMAIN,
     "/mnt/data1/zha00175/StitchCUDA/dataset/CudaForge/train_new_clean.parquet",
     "/mnt/data1/zha00175/StitchCUDA/dataset/CudaForge/test.parquet"),
    (S.TRITON_DOMAIN,
     "/mnt/data1/zha00175/StitchCUDA/dataset/Triton/train.parquet",
     "/mnt/data1/zha00175/StitchCUDA/dataset/Triton/test.parquet"),
]


def _ei(x):
    return x if isinstance(x, dict) else json.loads(x)


def _cats(df, domain):
    return collections.Counter(SP.level_of(x, None) for x in df.extra_info)


def _each_arm(fn):
    """Run `fn(domain, prompt, train_df, test_df)` for every arm, reporting which one failed."""
    for domain, tr_path, te_path in ARMS:
        tr = pd.read_parquet(tr_path)
        te = pd.read_parquet(te_path) if te_path else None
        try:
            fn(domain, O.render_system_prompt(domain), tr, te)
        except AssertionError as e:
            raise AssertionError(f"[{domain.name}] {e}") from None


# ---------- the two prompts must not contradict each other ----------

def test_both_prompts_agree_a_local_stall_counts_while_the_total_climbs():
    sys_p = O.render_system_prompt(S.CUDA_DOMAIN)
    assert "still climbing" in TRI and "wrong signal" in TRI
    assert "Do not require the total to flatten before acting" in sys_p
    assert "PREFER TO DECLINE" not in sys_p


def test_both_prompts_agree_declining_has_a_run_level_cost():
    sys_p = O.render_system_prompt(S.CUDA_DOMAIN)
    assert "measured nothing" in TRI
    assert "not cheap across a run" in sys_p and "zero cost" not in sys_p


def test_both_prompts_state_the_ab_gate_bounds_the_downside():
    for p in (O.render_system_prompt(S.CUDA_DOMAIN), TRI):
        assert "discarded unless it wins" in p


def test_no_stale_measured_claim_from_another_domain():
    p = O.render_system_prompt(S.CUDA_DOMAIN)
    assert "0.078" not in p and "0.128" not in p


# ---------- claims about the data must match the data ----------

def test_category_counts_claimed_match_the_training_file():
    def check(domain, prompt, tr, te):
        c = _cats(tr, domain)
        facts = " ".join(str(f) for f in domain.extra_facts) + str(domain.category_info)
        for cat, n in c.items():
            assert str(n) in facts, f"{cat}={n} rows is not reflected anywhere in the domain facts"
    _each_arm(check)


def test_every_training_row_lands_in_a_declared_category():
    """The Triton bug: level==0 on every row produced 'improve_l0', a label in no domain, and
    splice matched nothing. Silent — no error, no injected rows, every cycle."""
    def check(domain, prompt, tr, te):
        got = set(_cats(tr, domain))
        assert got <= set(domain.categories), \
            f"rows carry categories the domain does not declare: {sorted(got - set(domain.categories))}"
        assert got == set(domain.categories), \
            f"domain declares categories with no rows: {sorted(set(domain.categories) - got)}"
    _each_arm(check)


def test_held_out_composition_claim_is_true():
    def check(domain, prompt, tr, te):
        c = set(_cats(te, domain))
        facts = " ".join(str(f) for f in domain.extra_facts)
        if len(c) == 1:
            assert "no per-category breakdown" in facts, \
                "held-out has one category but the domain does not say so"
        else:
            assert "covers EVERY category" in facts or "every category" in facts.lower(), \
                "held-out covers several categories but the domain does not say so"
    _each_arm(check)


def test_training_and_held_out_share_no_reference_implementation():
    def check(domain, prompt, tr, te):
        a = {str(_ei(x).get("answer", "")).strip() for x in tr.extra_info}
        b = {str(_ei(x).get("answer", "")).strip() for x in te.extra_info}
        assert not (a & b), f"{len(a & b)} reference implementations appear on both sides"
    _each_arm(check)


# ---------- the injection mechanism must reach what the prompt offers ----------

def test_every_scope_the_prompt_offers_actually_injects():
    """splice falls back to appending at the end when its anchor is missing, which fails quietly.
    A scope that injects nothing, or injects after the output contract, is a scope the Teacher is
    being offered on false pretences."""
    def check(domain, prompt, tr, te):
        for cat in domain.categories:
            sc = S.apply_text_ops(S.empty_scaffold(domain),
                                  [{"target": cat, "text": "MARKER_XYZ"}])
            rows = [SP.splice(p[0]["content"], SP.render_block(sc, cat)) for p in tr.prompt[:400]]
            hits = [c for c in rows if "MARKER_XYZ" in c]
            assert hits, f"scope {cat} injected into nothing"
            for c in hits:
                pos = [c.index(a) for a in SP._ANCHORS if a in c]
                assert pos, f"scope {cat}: no known anchor in the prompt (would append at the end)"
                assert c.index("MARKER_XYZ") < min(pos), \
                    f"scope {cat}: text landed AFTER the output contract"
    _each_arm(check)


def test_validate_accepts_exactly_the_scopes_the_prompt_names():
    def check(domain, prompt, tr, te):
        for cat in domain.categories:
            assert cat in prompt, f"scope {cat} is editable but never named in the prompt"
            ok, why = S.validate_action(
                {"text_ops": [{"target": cat, "text": "x"}], "p_ops": []}, domain)
            assert ok, f"prompt offers scope {cat} but validate_action rejects it: {why}"
        ok, _ = S.validate_action(
            {"text_ops": [{"target": "definitely_not_a_scope", "text": "x"}], "p_ops": []}, domain)
        assert not ok, "an unknown target was accepted"
    _each_arm(check)


def test_hints_availability_follows_the_domain_not_the_prose():
    def check(domain, prompt, tr, te):
        possible = domain.has_reference_solutions and domain.instance_scope
        assert ("hints: YES" in prompt) == possible
    _each_arm(check)


# ---------- the A/B gate must judge what the Teacher actually changed ----------

def test_ab_gate_aggregates_over_the_touched_scope_only():
    """The gate reads {task: (rate, n)}. When its per-category lookup missed it used to fall back
    to the overall mean, so a one-category edit was judged on every category at once."""
    m = {"bare": {"a": (0.1, 24), "b": (0.9, 24)},
         "current": {"a": (0.1, 24), "b": (0.9, 24)},
         "candidate": {"a": (0.4, 24), "b": (0.9, 24)}}
    assert G.ab_gate(m, ["a"])["accept"], "an improvement in the touched scope was not seen"
    assert not G.ab_gate(m, ["b"])["accept"], "an untouched scope reported a change"


def test_ab_gate_rejects_rather_than_inventing_a_measurement():
    assert not G.ab_gate({"bare": {}, "current": {}, "candidate": {}}, ["a"])["accept"]
    assert not G.ab_gate({"bare": {}, "current": {}, "candidate": {}}, [])["accept"]


def test_general_edit_is_judged_on_every_category():
    def check(domain, prompt, tr, te):
        t = S.touched_tasks([{"target": "general", "text": "x"}], domain)
        assert set(t) == set(domain.categories)
    _each_arm(check)


# ---------- mechanism constants must be disclosed ----------

def test_p_limits_in_prompt_match_the_constants():
    def check(domain, prompt, tr, te):
        assert str(S.P_MAX_DELTA) in prompt, "per-cycle step limit not disclosed"
        assert str(S.P_MAX) in prompt, "destination cap not disclosed"
        assert "HARD CAP" in prompt
    _each_arm(check)


def test_text_cap_is_not_disclosed_as_a_number():
    def check(domain, prompt, tr, te):
        assert str(S.MAX_TEXT_CHARS) not in prompt
        assert "no length target" in prompt
    _each_arm(check)


def test_reward_structure_is_stated_since_the_signals_assume_it():
    def check(domain, prompt, tr, te):
        facts = " ".join(str(f) for f in domain.extra_facts)
        assert "gate, not the objective" in facts and "speedup" in facts
    _each_arm(check)


def test_every_signal_field_the_prompt_names_exists_in_the_packet():
    def check(domain, prompt, tr, te):
        signals = {
            "per_task_gap": {}, "all_fail_groups": {}, "failures": [], "successes": [],
            "valid_seen": {"avg": 0.3, "per_task": {}, "draws": [0.3]},
            "train_curve": {"bare_mean": 0.3, "injected_mean": 0.3},
        }
        obs = O.assemble_observation(S.empty_scaffold(domain), signals, [], 10, domain)
        flat = set(obs) | set(obs.get("signals", {}))
        for name in ("per_task_gap", "all_fail_groups", "valid_seen", "eval_trajectory",
                     "train_curve", "decision_history"):
            if f"signals.{name}" in prompt or name in prompt:
                assert name in flat, f"prompt names {name} but the observation lacks it"
    _each_arm(check)


def test_triage_packet_carries_what_its_prompt_tells_it_to_read():
    obs = O.assemble_triage_observation(
        S.empty_scaffold(S.TRITON_DOMAIN), {"series": []}, [], 10,
        per_task={"TritonKernel": 0.3}, per_task_n={"TritonKernel": 180},
        train_rollouts=[{"cycle": 1, "step": 10, "by_category": {}}],
        cycle=1, n_cycles=20, floor_cycles=3)
    for name in ("train_rollouts_by_category", "valid_seen_per_task", "eval_trajectory",
                 "progress"):
        assert name in TRI, f"{name} is supplied but never mentioned in the triage prompt"
        assert name in obs, f"triage prompt names {name} but the packet does not carry it"


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
    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
    sys.exit(1 if fails else 0)
