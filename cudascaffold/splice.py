"""Build the parquet a training cycle reads, with the current scaffold spliced in.

Why a parquet rewrite rather than a runtime hook: StitchCUDA is stock verl driven by a
`custom_reward_function` — there is no environment manager to intercept, and the prompt comes
straight out of the dataset. Regenerating the file each cycle keeps core verl untouched (the
same property that made this repo preferable to KernelGYM) and costs milliseconds on 400 rows.

Injection is decided ONCE PER TASK, not per row: several rows can share a `task_name`, and the
policy is scored in groups, so letting some rows of a task see the text and others not would
put the scaffold on both sides of the comparison the advantage is computed from.
"""
from __future__ import annotations

import hashlib
import json
import os

_MARK_OPEN = "[STRATEGY — injected during training only]"
_MARK_CLOSE = "[END STRATEGY]"
# Splice ahead of the block that tells the model what to emit, so the guidance is read before the
# output contract rather than after it. Which phrase opens that block depends on the dataset, so
# the anchors are tried in order and the first one present wins.
#
# A single hard-coded anchor is not safe here: splice() falls back to appending at the END of the
# prompt when the anchor is missing, which does not fail loudly. Pointing the CUDA anchor at the
# Triton set put every injected block after "Let's think step by step." — still in the prompt, but
# after the instruction it was meant to qualify, and nothing reported anything wrong.
#
# "Optimize the architecture named" is deliberately the invariant PREFIX: the class name varies
# (Model, TripleMarginLoss, ...), so the full sentence misses rows.
_ANCHORS = (
    "OUTPUT RULES (STRICT)",              # CudaForge
    "Optimize the architecture named",    # drkernel-rl-data (Triton)
)


def level_of(extra_info, data_source=None):
    """The scaffold category a row belongs to, or None if it fits none.

    Categories cross the task mode with the level: improvement rows carry a level and become
    `improve_l{1,2,3}`; from-scratch rows carry no level and become `scratch`. Every row of
    train_new.parquet lands in exactly one, which is what makes per-category text reach all of
    the data rather than half of it.
    """
    ei = _as_dict(extra_info)
    # An explicit category wins over anything derived. The Triton set carries `category` directly
    # and has level==0 on every row, so deriving from level there produced "improve_l0" — a label
    # in no domain's category list, which meant splice() matched nothing and the scaffold was
    # never injected at all. Silently: no error, no injected rows, every cycle.
    cat = ei.get("category")
    if isinstance(cat, str) and cat.strip():
        return cat.strip()
    lv = ei.get("level")
    if lv is None:
        # No level => from-scratch. Confirm with data_source when the caller has it, so a future
        # dump that drops `level` from improvement rows fails loudly instead of silently
        # relabelling them as scratch.
        if data_source is None or str(data_source) == "CudaForge":
            return "scratch"
        return None
    try:
        return f"improve_l{int(float(lv))}"
    except (TypeError, ValueError):
        return None


def task_key(extra_info, fallback_index=None):
    """Grouping key for the injection coin flip. `task_name` when present (rows of the same
    kernel task share it); otherwise the row index, which degrades to per-row grouping."""
    ei = _as_dict(extra_info)
    return str(ei.get("task_name") or ei.get("index") or fallback_index)


def _as_dict(x):
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}


def render_block(scaffold, level, instance=None):
    """The text this row receives: general + its category's + its own instance's, in that order.

    Ordering is general -> category -> instance, i.e. widest to narrowest, so the most specific
    statement is the last thing read before the output contract.
    """
    parts = []
    g = (scaffold.get("general_skill") or "").strip()
    if g:
        parts.append(g)
    sk = (scaffold.get("skills") or {}).get(level) if level else None
    sk = (sk if isinstance(sk, str) else (sk or {}).get("text", "")) if sk else ""
    if sk and sk.strip():
        parts.append(sk.strip())
    inst = (scaffold.get("instances") or {}).get(instance) if instance else None
    if inst and str(inst).strip():
        parts.append(str(inst).strip())
    return "\n\n".join(parts)


def splice(prompt, block):
    """Insert `block` before the output contract. Pure; returns `prompt` unchanged if empty."""
    if not block or not block.strip():
        return prompt
    wrapped = f"\n{_MARK_OPEN}\n{block.strip()}\n{_MARK_CLOSE}\n\n"
    for anchor in _ANCHORS:
        if anchor in prompt:
            return prompt.replace(anchor, wrapped + anchor, 1)
    return prompt.rstrip() + "\n" + wrapped


def _fires(task, level, scaffold, seed):
    """Group-level Bernoulli(p). Deterministic in (seed, task) so the same cycle is reproducible
    and so a paired A/B can replay the identical assignment under a different scaffold."""
    p = float((scaffold.get("p_task") or {}).get(level, scaffold.get("default_p", 0.0)) or 0.0)
    if p <= 0:
        return False
    if p >= 1:
        return True
    h = hashlib.sha1(f"{seed}|{task}".encode()).digest()
    return (int.from_bytes(h[:8], "big") / 2 ** 64) < p


def build(src_parquet, dst_parquet, scaffold, seed=0, mode="full"):
    """Write `dst_parquet` = `src_parquet` with the scaffold spliced into the chosen rows.

    mode: 'full'     — inject per p_task (training)
          'none'     — inject nothing (the bare condition; the eval anchor and the A/B's control)
          'force'    — inject into every row regardless of p (the A/B's injected condition)
    Returns {n_rows, n_injected, per_level: {level: (injected, total)}}.
    """
    import pandas as pd

    df = pd.read_parquet(src_parquet)
    prompts = df["prompt"].tolist()
    extras = df["extra_info"].tolist()
    out, n_inj = [], 0
    per_level = {}
    for i, (p, ei) in enumerate(zip(prompts, extras)):
        lv = level_of(ei, df["data_source"].iloc[i] if "data_source" in df.columns else None)
        stat = per_level.setdefault(lv or "unlabelled", [0, 0])
        stat[1] += 1
        block = render_block(scaffold, lv, task_key(ei, i)) if mode != "none" else ""
        fire = bool(block) and (mode == "force" or _fires(task_key(ei, i), lv, scaffold, seed))
        if fire:
            p = _splice_row(p, block)
            n_inj += 1
            stat[0] += 1
        out.append(p)
    df["prompt"] = out
    os.makedirs(os.path.dirname(os.path.abspath(dst_parquet)), exist_ok=True)
    df.to_parquet(dst_parquet, index=False)
    return {"n_rows": len(df), "n_injected": n_inj,
            "per_level": {k: tuple(v) for k, v in per_level.items()},
            "dst": dst_parquet}


def _splice_row(prompt_cell, block):
    """verl stores the prompt as a chat list of {role, content}; splice into the user turn.
    Handles the plain-string form too so this does not silently no-op on a different dump."""
    import numpy as np

    if isinstance(prompt_cell, str):
        return splice(prompt_cell, block)
    seq = list(prompt_cell) if isinstance(prompt_cell, (list, tuple, np.ndarray)) else None
    if seq is None:
        return prompt_cell
    for i in range(len(seq) - 1, -1, -1):          # last user turn is the one being answered
        m = seq[i]
        if isinstance(m, dict) and m.get("role") == "user":
            m = dict(m)
            m["content"] = splice(m.get("content", ""), block)
            seq[i] = m
            return np.array(seq, dtype=object) if isinstance(prompt_cell, np.ndarray) else seq
    return prompt_cell
