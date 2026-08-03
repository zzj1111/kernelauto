"""Auto-scaffold harness (redesigned).

A weak Student is trained with GiGPO on ALFWorld; a Teacher (GPT-5.5) reads the
STANDALONE training signals and edits a text scaffold injected during training only.
The harness's ONLY jobs are the two gates (accept a proposed text change via a
frozen-policy A/B on train; revert weights+scaffold on a sustained held-out regression)
plus the two fixed rules it may never delegate: what counts as success (the held-out
standalone anchor) and the safety net. Everything else — what to change and why — is
the Teacher's.
"""
from . import scaffold, gates, observation, teacher  # noqa: F401
