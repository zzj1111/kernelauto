"""Auto-scaffold harness for the CUDA/Triton kernel arm.

A Student is trained with GRPO on kernel generation and optimization; a Teacher (GPT-5.5)
reads the training signals and edits a text scaffold that is injected into TRAINING prompts
only. There is no environment manager to intercept here, so the scaffold reaches the policy by
rewriting the training parquet each cycle (splice.py) — core verl stays untouched, and the
held-out eval is bare by construction because splice never touches test.parquet.

The harness's own jobs are deliberately few: run the A/B that decides whether a proposed text
change is kept, and hold the one rule it may never delegate — what counts as success (the
held-out standalone anchor). What to change and why is the Teacher's.

One thing this docstring used to promise and the code does not have: a revert gate. It was
removed 2026-07-29, so an accepted scaffold is permanent for the rest of the run and only the
held-out curve will show that it was noise. Say so plainly here rather than leave a reader
believing there is a safety net.
"""
from . import scaffold, gates, observation, teacher  # noqa: F401
