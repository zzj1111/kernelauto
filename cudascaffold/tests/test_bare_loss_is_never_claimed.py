"""The Teacher must never be told this arm has a bare-prompt loss.

observation.py is shared byte-for-byte with the ALFWorld arm, where `algorithm.bare_prompt_loss`
is a real trainer option; in THIS repo's verl it does not exist at all. The shared module renders
its description of the loss from ARM_BARE_LOSS, so the arm-specific run_arm.py pins that variable
— and the pin has to be an assignment, not a setdefault, or an inherited value wins and the
Teacher reasons correctly about a mechanism that is not there. Every judgement it makes about
what injected text can buy follows from that description, so the failure is silent and total.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_this_verl_really_has_no_such_option():
    """If upstream ever grows the option, this test fails and the pin can be reconsidered."""
    hit = subprocess.run(["grep", "-rl", "bare_prompt_loss", os.path.join(REPO, "verl")],
                         capture_output=True, text=True)
    assert hit.returncode != 0, \
        f"verl now defines bare_prompt_loss ({hit.stdout.strip()}) — revisit the pin in run_arm.py"


def test_an_inherited_true_does_not_survive_the_import():
    out = subprocess.run(
        [sys.executable, "-c",
         "import cudascaffold.run_arm as R; import os; print(os.environ['ARM_BARE_LOSS'])"],
        cwd=REPO, env=dict(os.environ, ARM_BARE_LOSS="True"),
        capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "False", (
        f"ARM_BARE_LOSS survived as {out.stdout.strip()!r} — setdefault is back, and a stale "
        f"export now decides what the Teacher is told about the loss")


def test_the_pin_is_an_assignment_not_a_setdefault():
    """Stated structurally too: the subprocess check above needs an interpreter that can import
    the package, and this one holds even where that is unavailable."""
    src = open(os.path.join(REPO, "cudascaffold", "run_arm.py"), encoding="utf-8").read()
    assert re.search(r'os\.environ\[\s*["\']ARM_BARE_LOSS["\']\s*\]\s*=\s*["\']False["\']', src), \
        "the ARM_BARE_LOSS pin is no longer an unconditional assignment"


def test_the_trainer_command_never_passes_the_option():
    from cudascaffold import adapters as A
    src = open(A.__file__, encoding="utf-8").read()
    assert "bare_prompt_loss" not in src, \
        "the trainer command references an option this verl does not define"
