"""The arch list every GPU subprocess inherits must match the GPU it runs on.

base_env exports TORCH_CUDA_ARCH_LIST, and kernel_runner treats the env as authoritative over
its own stdin/detection fallbacks — so this one constant decides what architecture every
generated kernel is compiled for. It used to be hardcoded "9.0", which is only ever right on
sm_90 machines: a cubin does not run across major compute-capability versions, so on a B200
every candidate would fail to load, score 0.0, and read as a model that cannot write CUDA.
Nothing would error; the training signal would just be uniformly wrong.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from cudascaffold import adapters as A


class _Result:
    def __init__(self, stdout, returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def test_detection_reports_what_the_driver_says(monkeypatch):
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: _Result("10.0\n10.0\n"))
    assert A._detect_arch_list() == "10.0"


def test_a_mixed_node_compiles_for_every_capability_present(monkeypatch):
    """REWARD_CUDA_VISIBLE_DEVICES can point the benchmark at a different GPU than training,
    so the safe list is the union, not the first row."""
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: _Result("9.0\n10.0\n"))
    assert A._detect_arch_list() == "10.0;9.0" or A._detect_arch_list() == "9.0;10.0"


def test_detection_failure_falls_back_loudly_not_silently(monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("no nvidia-smi")
    monkeypatch.setattr(A.subprocess, "run", boom)
    assert A._detect_arch_list() == "9.0"
    assert "WARNING" in capsys.readouterr().out, (
        "the fallback must announce itself: a silent 9.0 on a non-sm_90 box zeroes every "
        "reward with no error anywhere")


def test_garbage_from_the_driver_is_not_exported_into_compilers(monkeypatch, capsys):
    monkeypatch.setattr(A.subprocess, "run",
                        lambda *a, **k: _Result("NVIDIA-SMI has failed\n", returncode=9))
    assert A._detect_arch_list() == "9.0"


def test_the_operator_override_beats_detection():
    """Source-level: the module must consult ARM_TORCH_CUDA_ARCH_LIST before detecting, so a
    cross-compiling site can still pin the list without editing code."""
    src = open(A.__file__, encoding="utf-8").read()
    m = re.search(r"^TORCH_CUDA_ARCH_LIST = (.+)$", src, re.M)
    assert m, "TORCH_CUDA_ARCH_LIST is no longer assigned at module level"
    assert "ARM_TORCH_CUDA_ARCH_LIST" in m.group(1) and "_detect_arch_list" in m.group(1), (
        f"assignment is {m.group(1)!r}; it must try the ARM_TORCH_CUDA_ARCH_LIST override "
        "first and fall back to detection")
