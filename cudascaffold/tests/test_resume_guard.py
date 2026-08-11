"""Resume must fail loudly when state.json and the checkpoints disagree.

verl resume_mode=auto silently retrains from BASE weights when the checkpoint dir is empty;
paired with a state.json that says step=N this produces a run whose journal lies about what
the weights are. The guard turns that five-hours-later mystery into a launch-time refusal.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from cudascaffold import adapters as A


def _ckpt(tmp_path, latest):
    if latest is not None:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(str(latest))
    return str(tmp_path)


def test_matching_state_and_checkpoint_resume(tmp_path):
    A.check_resume_consistency(20, _ckpt(tmp_path, 20))   # no raise


def test_fresh_state_needs_no_checkpoint(tmp_path):
    A.check_resume_consistency(0, str(tmp_path))          # no raise


def test_state_ahead_of_checkpoints_refuses(tmp_path):
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        A.check_resume_consistency(20, _ckpt(tmp_path, None))


def test_foreign_newer_checkpoints_refuse(tmp_path):
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        A.check_resume_consistency(4, _ckpt(tmp_path, 180))


def test_the_escape_hatch_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("ARM_ALLOW_STATE_CKPT_MISMATCH", "1")
    A.check_resume_consistency(20, _ckpt(tmp_path, None))  # warns, proceeds
