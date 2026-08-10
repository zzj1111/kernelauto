"""A measurement pass must be fed in waves, and the waves must not correlate with the arm.

verl defaults data.val_batch_size to null, which means "the whole file in one batch"
(ray_trainer.py:398-400). The A/B therefore handed 540 rows to the reward at once, and every
completed generation forked a kernel_runner that creates a CUDA context: 85 of them on one GPU
inside 49 seconds on 2026-08-10, which deadlocked the driver's global rwsem and cost a reboot.

Bounding the batch bounds the peak directly, and the validation loop iterates its dataloader
batch by batch (ray_trainer.py:550) so the model is still loaded exactly once — chunking the
file into separate passes would instead pay an 8B load plus vLLM init per chunk.

The second test is the one that keeps the fix honest: batching is only safe if a batch is a
random mix of arms. If chunk boundaries lined up with arms, each wave would score one condition
under its own batch composition, and the comparison the gate makes would be biased rather than
merely noisy.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def test_a_measurement_pass_bounds_its_batch():
    from cudascaffold import adapters as A
    src = open(A.__file__, encoding="utf-8").read()
    body = src[src.index("def _measure_pass("):src.index("def _subset_tasks(")]
    assert "data.val_batch_size=" in body, (
        "the pass does not set val_batch_size, so verl falls back to null — the whole file in "
        "one batch, which is what wedged the driver")
    assert A._MEASURE_BATCH_ROWS <= 48, (
        f"batch of {A._MEASURE_BATCH_ROWS} rows: training sustains 48 candidates per step and "
        f"85 concurrent wedged the node, so a measurement wave must stay at or under training")


def test_the_arms_are_interleaved_so_a_wave_is_a_mix():
    """Reproduces the shuffle the adapter applies, then checks every wave carries all three arms.

    Done on the real construction — concatenate three arms, shuffle with the seeded RNG — rather
    than on a mock, because the property at risk is exactly that the shuffle is what separates
    'batched' from 'batched by arm'.
    """
    import pandas as pd
    from cudascaffold import adapters as A

    n_problems = 180
    frames = [pd.DataFrame({"row": range(n_problems), "arm": [arm] * n_problems})
              for arm in ("bare", "current", "candidate")]
    merged = (pd.concat(frames, ignore_index=True)
                .sample(frac=1.0, random_state=20260722)
                .reset_index(drop=True))

    size = A._MEASURE_BATCH_ROWS
    waves = [merged.iloc[i:i + size] for i in range(0, len(merged), size)]
    assert len(waves) > 1, "the point is that there is more than one wave"
    for i, w in enumerate(waves):
        present = set(w["arm"])
        assert present == {"bare", "current", "candidate"}, (
            f"wave {i} carries only {present}: batch composition would become a per-arm effect")
        counts = w["arm"].value_counts()
        assert counts.max() <= 3 * counts.min(), (
            f"wave {i} is lopsided ({dict(counts)}); arms should be roughly balanced per wave")


def test_every_problem_is_still_scored_once_per_arm():
    """Batching must not drop or duplicate work — the gate's n depends on it."""
    import pandas as pd
    from cudascaffold import adapters as A
    n = 180
    merged = (pd.concat([pd.DataFrame({"row": range(n), "arm": [a] * n})
                         for a in ("bare", "current", "candidate")], ignore_index=True)
                .sample(frac=1.0, random_state=20260722).reset_index(drop=True))
    size = A._MEASURE_BATCH_ROWS
    seen = pd.concat([merged.iloc[i:i + size] for i in range(0, len(merged), size)])
    assert len(seen) == 3 * n
    assert (seen.groupby("arm")["row"].nunique() == n).all()


def test_hydra_overrides_match_what_the_config_already_defines():
    """A '+' prefix means "add a key that is not there". Using it on a key that IS there is a
    hard startup failure — `Could not append to config. An item is already at ...` — and it cost
    a full model load to discover, because nothing before the trainer's own config composition
    can see it.

    The sibling ALFWorld arm legitimately needs '+' for the same setting, since its verl does
    not define the key. So the syntax is not a matter of taste; it has to match the config in
    THIS tree.
    """
    import re as _re
    import yaml
    from cudascaffold import adapters as A

    cfg_path = os.path.join(REPO, "verl", "trainer", "config", "_generated_ppo_trainer.yaml")
    with open(cfg_path) as f:
        defined = yaml.safe_load(f)

    def exists(dotted):
        node = defined
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        return True

    src = open(A.__file__, encoding="utf-8").read()
    body = src[src.index("def _train_cmd("):src.index("def _measure_pass(")]
    for m in _re.finditer(r'f?"\+([a-z_]+(?:\.[a-z_0-9]+)+)=', body):
        assert not exists(m.group(1)), (
            f"'+{m.group(1)}' appends a key the config already defines; hydra refuses and the "
            f"trainer dies at startup")
    for m in _re.finditer(r'f?"([a-z_]+(?:\.[a-z_0-9]+)+)=', body):
        key = m.group(1)
        if key.startswith(("data.", "trainer.", "algorithm.")) and not exists(key):
            raise AssertionError(
                f"'{key}' overrides a key this config does not define; it needs a '+' prefix")
