"""The Teacher (GPT-5.5): propose an action from the observation.

The GPT call is injected (``call_fn``) so the loop and tests do not depend on the
network. ``propose`` normalizes and PHYSICALLY validates the returned action; on any
malformed output it falls back to a no-op (fail-safe: never crash the run, never apply
garbage). It does NOT judge whether the content is good — that is the A/B gate's job.
"""
from __future__ import annotations

import json
import os

from . import observation as O
from . import scaffold as S

MODEL = "gpt-5.5"
DEFAULT_KEY_FILE = "/mnt/data1/zha00175/tool-agent-secrets/openai.env"
NOOP = {"diagnosis": "", "text_ops": [], "p_ops": []}


def normalize(raw, domain=None):
    """Coerce a raw Teacher dict into a validated action, or a no-op on any problem.
    Returns (action, note). Physical validation only."""
    if not isinstance(raw, dict):
        return dict(NOOP), "non-dict output -> no-op"
    action = {
        "diagnosis": str(raw.get("diagnosis", "") or "")[:2000],
        "text_ops": raw.get("text_ops") or [],
        "p_ops": raw.get("p_ops") or [],
    }
    ok, reason = S.validate_action(action, domain)
    if not ok:
        return dict(NOOP), f"invalid action ({reason}) -> no-op"
    return action, "ok"


def _read_key(key_file):
    """Return the API key from either an env-style file (OPENAI_API_KEY=...) or a raw key file."""
    txt = open(key_file).read()
    for line in txt.splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            return line.strip().split("=", 1)[1].strip()
    return txt.strip()


def openai_call(system, user, key_file=DEFAULT_KEY_FILE, model=MODEL, max_tokens=6000):
    """Real GPT call. Imported lazily so tests never need the openai package/network."""
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY") or _read_key(key_file)
    cli = OpenAI(api_key=key, timeout=300, max_retries=2)
    r = cli.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, max_completion_tokens=max_tokens)
    return json.loads(r.choices[0].message.content)


def propose(obs, call_fn=openai_call, domain=None, priors=False):
    """Render prompt -> call_fn(system, user) -> normalized, validated action.
    Any exception in the call becomes a no-op (the run continues on the current scaffold)."""
    system = O.render_system_prompt(domain, priors=priors)
    user = O.render_user_prompt(obs)
    try:
        raw = call_fn(system, user)
    except Exception as e:  # network / parse / auth — degrade to no-op, never crash
        return dict(NOOP), f"teacher call failed ({str(e)[:150]}) -> no-op"
    return normalize(raw, domain)


# Errors that mean the Teacher ITSELF cannot be reached, as opposed to a Teacher that answered
# badly. Matched on the message because the openai SDK raises the same RateLimitError class for
# "slow down, retry later" (transient, worth measuring for) and "you have no credits"
# (persistent, nothing downstream can run).
_UNREACHABLE = ("insufficient_quota", "credit_balance_exhausted", "no credits remaining",
                "invalid_api_key", "incorrect api key", "authentication", "401",
                "apiconnectionerror", "connection error", "no_key")


def teacher_unreachable(err):
    """True when `err` means no Teacher call can succeed this cycle, however long we wait."""
    return any(s in str(err).lower() for s in _UNREACHABLE)


def triage(obs, call_fn=openai_call):
    """Ask whether this cycle is worth measuring, BEFORE paying for the signals pass.

    Returns (intervene: bool, why: str).

    Fails OPEN on a Teacher that is reachable but unhelpful — malformed JSON, a missing verdict,
    a transient rate limit — because a broken pre-check should degrade to the previous behaviour
    (measure every cycle) rather than silently freezing the scaffold for the rest of the run.

    Fails CLOSED when the Teacher is unreachable for a reason that will not clear on its own
    (2026-08-01: the OpenAI account hit credit_balance_exhausted mid-run). The fail-open rule is
    predicated on someone being able to ACT on the measurement, and triage and propose share one
    call_fn: if this call died on quota or auth, the propose() right after it dies the same way
    and returns a no-op. Measuring anyway would spend a full bare+injected pass (~85 min of GPU
    here) producing signals with no consumer, every cycle, for as long as the outage lasts.
    Declining instead keeps training and eval running — which is the empty-scaffold control arm —
    and costs nothing, because the loop re-asks next cycle and resumes the moment the key works.
    """
    try:
        raw = call_fn(O.render_triage_prompt(), json.dumps(obs, ensure_ascii=False)[:60000])
    except Exception as e:
        if teacher_unreachable(e):
            return False, (f"teacher unreachable ({str(e)[:120]}) -> skipping the signals pass; "
                           f"training and eval continue on the current scaffold")
        return True, f"triage call failed ({str(e)[:120]}) -> measuring anyway"
    if not isinstance(raw, dict) or not isinstance(raw.get("intervene"), bool):
        return True, "triage returned no usable verdict -> measuring anyway"
    return raw["intervene"], str(raw.get("why", ""))[:500]
