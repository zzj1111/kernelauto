"""v2 Teacher adapter for the kernel arm: teacherflow's investigative workflow behind the
SAME (action, note) contract as teacher.propose. Selected by AUTOSCAFFOLD_TEACHER=v2.

Window = this cycle's recorder rows (tail of work/rollouts.jsonl sized to K steps x
train batch x rollout n). The system prompt is cudascaffold's own domain prompt with
the shared all-fail-first mechanism block spliced in (kerneldomain.build_system), so
domain facts, item rules and the exact JSON grammar stay the ones normalize() validates.
"""
from __future__ import annotations

import json
import os
import sys

from . import observation as O
from . import teacher as T1

TEACHERFLOW_PATH = os.environ.get(
    "TEACHERFLOW_PATH", os.path.join(os.path.expanduser("~"), "teacherflow"))


def _client():
    from openai import OpenAI
    return OpenAI(api_key=T1._read_key(T1.DEFAULT_KEY_FILE), timeout=600, max_retries=2)


def make_teacher_fn(cfg):
    if TEACHERFLOW_PATH not in sys.path:
        sys.path.insert(0, TEACHERFLOW_PATH)
    from teacherflow import kerneldomain as KD
    from teacherflow.data import RunData
    from teacherflow.workflow import investigate_and_propose

    tdir = os.path.join(cfg["work"], "v2_transcripts")
    os.makedirs(tdir, exist_ok=True)
    tail_rows = int(os.environ.get("AUTOSCAFFOLD_V2_TAIL_ROWS", "0")) or (
        int(cfg.get("steps_per_cycle", 10)) * int(cfg.get("train_batch_size", 8)) * int(cfg.get("rollout_n", 6)))
    system = KD.build_system(O.render_system_prompt(cfg["domain"], priors=cfg.get("teacher_priors", False)))

    def teacher_fn(obs, scaffold=None):
        cycle = obs.get("cycle", "?") if isinstance(obs, dict) else "?"
        step = obs.get("step", "?") if isinstance(obs, dict) else "?"
        sr = (obs.get("valid_seen") or {}).get("avg") if isinstance(obs, dict) else None
        preamble = (f"Cycle {cycle} just finished training (now at step {step}). "
                    + (f"Held-out scaffold-free score: {sr}. " if sr is not None else "")
                    + "Investigate as you see fit, then decide.")
        try:
            data = RunData(os.path.join(cfg["work"], "rollouts.jsonl"), tail_rows=tail_rows)
            data.scaffold = scaffold or {}
            data.state = {"decision_history": (obs.get("decision_history") if isinstance(obs, dict) else None) or []}
            decision, transcript = investigate_and_propose(
                _client(), data, model=T1.MODEL, user_preamble=preamble,
                tools=KD, system=system)
        except Exception as e:
            if T1.teacher_unreachable(e):
                return dict(T1.NOOP), f"teacher unreachable ({str(e)[:150]}) -> no-op"
            return dict(T1.NOOP), f"v2 investigation failed ({str(e)[:150]}) -> no-op"
        try:
            with open(os.path.join(tdir, f"c{cycle}.json"), "w") as f:
                json.dump({"decision": decision, "transcript": transcript}, f,
                          ensure_ascii=False, indent=1)
        except OSError:
            pass
        if decision is None:
            return dict(T1.NOOP), "v2 malformed final output -> no-op"
        n_calls = len([t for t in transcript if t.get("tool")])
        action, note = T1.normalize(decision, domain=cfg["domain"], scaffold=scaffold)
        return action, f"v2({n_calls} calls) {note}"

    return teacher_fn
