"""The domain-independent core must stay byte-identical across the two arms.

Six files — scaffold, splice, observation, teacher, gates, loop — carry no domain knowledge
beyond what a Domain object supplies, and both the Triton/CUDA arm and this one run them. They
exist as two copies only because the arms live in different repositories.

This test exists because the copies silently diverged for two days. The CUDA arm was rewritten
around items with ids, a per-cycle edit budget, zero-gradient measured on reward variance, and an
A/B moved to the held-out split; ALFWorld kept running the older whole-scope `text_ops` design
with the all-fail statistic that had already been shown wrong. Nothing failed — both suites were
green against their own copy — so the divergence was only found by comparing modification times.

A hash mismatch here means one arm was edited and the other was not. Fix by copying, not by
editing this test: a change worth making to the core is worth making to both arms, and a change
that is only right for one arm belongs in adapters.py or on the Domain object.

Skips (rather than fails) when the other repository is absent, so the suite still runs on a
machine that only has one of them checked out.
"""
from __future__ import annotations

import hashlib
import os

import pytest

CORE = ("scaffold.py", "splice.py", "observation.py", "teacher.py", "gates.py", "loop.py")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The peer checkout is machine-specific by nature, so the default is only a convenience for the
# box this was written on; every other machine sets AUTOSCAFFOLD_PEER or gets a skip. It points
# at the ALFWorld arm that was FROZEN and rebuilt elsewhere, so a mismatch here now means "the
# peer moved on", not necessarily "this arm drifted" — check which side changed before copying.
OTHER = os.environ.get("AUTOSCAFFOLD_PEER",
                       "/mnt/data1/zha00175/verl-agent/agent_system/skill_opt/autoscaffold")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# Diverged from the frozen peer ON PURPOSE, with the date and the reason. The test's own rule
# is "fix by copying, not by editing this test" — but the peer tree is frozen by decree and
# cannot receive the copy, and the rebuilt ALFWorld arm no longer carries these files at all.
# 2026-08-11 observation.py: single-turn failure evidence (fail_kinds / sample_error /
# speedup_when_correct) and the held-out per-category breakdown in the valid_seen description.
# If the peer ever unfreezes, copy these files THERE and empty this set.
# 2026-08-11 scaffold.py: the target-GPU domain fact is now driver-detected (was a hardcoded
# H200/sm_90, wrong on the incoming B200s).
DIVERGED_FROM_FROZEN_PEER = {"observation.py", "scaffold.py"}


@pytest.mark.parametrize("name", CORE)
def test_core_file_matches_the_other_arm(name):
    peer = os.path.join(OTHER, name)
    if not os.path.exists(peer):
        pytest.skip(f"peer arm not present at {OTHER}")
    if name in DIVERGED_FROM_FROZEN_PEER:
        pytest.skip(f"{name} deliberately ahead of the frozen peer (see DIVERGED_FROM_FROZEN_PEER)")
    mine = os.path.join(HERE, name)
    assert _sha(mine) == _sha(peer), (
        f"{name} differs between the two arms.\n"
        f"  here: {mine}\n  peer: {peer}\n"
        f"Diff them and copy whichever is current — do not edit this test. A core change that is "
        f"only right for one arm belongs in adapters.py or on the Domain object.")


def test_the_domain_specific_files_are_NOT_expected_to_match():
    """Stated so the rule above is not read as "everything must match". adapters.py and
    run_arm.py are where the arms legitimately differ: one drives a Triton kernel benchmark,
    the other verl's GiGPO ALFWorld trainer, and their signal sources have nothing in common."""
    for name in ("adapters.py", "run_arm.py"):
        peer = os.path.join(OTHER, name)
        if not os.path.exists(peer):
            pytest.skip(f"peer arm not present at {OTHER}")
        assert _sha(os.path.join(HERE, name)) != _sha(peer), (
            f"{name} is byte-identical across arms, which means one arm is running the other's "
            f"plumbing")
