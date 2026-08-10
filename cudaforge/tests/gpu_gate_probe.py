"""Real-hardware check of the correctness gate — run it, do not import it.

The NaN and shape fixes are pinned without hardware elsewhere in this suite, by stubbing the
runner. That proves the logic; it does not prove that a kernel which actually emits NaN on a
card is rejected, because the gate is a float comparison over tensors that only exist at
runtime and the failure mode was that the comparison silently did nothing.

Measured on an H200 (sm_90), same kernel and payload against both runners:

    pre-fix (6b269ef)   correct: True,  speedup 0.745    <- every output element is NaN
    fixed               correct: False, kind correctness_error

Not collected by pytest (it compiles four kernels with nvcc and takes minutes). Run it:

    cd <repo>
    python cudaforge/tests/gpu_gate_probe.py $PWD/cudaforge/kernel_runner.py $(which python)

and point GATE_TEST_DEVICE at a free GPU. Exit status is 0 only if every case is judged as
expected, so it works as a release check.
"""
import json
import os
import subprocess
import sys

RUNNER = sys.argv[1] if len(sys.argv) > 1 else "cudaforge/kernel_runner.py"
PY = sys.argv[2] if len(sys.argv) > 2 else sys.executable
DEV = int(os.environ.get("GATE_TEST_DEVICE", "0"))

REF = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x + 1.0

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(1024, device="cuda")]
'''

CUDA_SRC_TEMPLATE = '''
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

src = r"""
#include <torch/extension.h>
__global__ void k(const float* x, float* y, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{ y[i] = {expr}; }}
}}
torch::Tensor run(torch::Tensor x) {{
    auto y = torch::empty_like(x);
    int n = x.numel();
    k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}}
"""
mod = load_inline(name="gate_probe_{tag}", cpp_sources="torch::Tensor run(torch::Tensor x);",
                  cuda_sources=src, functions=["run"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = mod.run(x)
        {post}
        return out
'''

CASES = {
    "correct": CUDA_SRC_TEMPLATE.format(expr="x[i] + 1.0f", tag="ok", post="pass"),
    "nan":     CUDA_SRC_TEMPLATE.format(expr="x[i] / 0.0f - x[i] / 0.0f", tag="nan", post="pass"),
    "shape":   CUDA_SRC_TEMPLATE.format(expr="x[i] + 1.0f", tag="shp", post="out = out[:16]"),
    "wrong":   CUDA_SRC_TEMPLATE.format(expr="x[i] * 2.0f", tag="wrg", post="pass"),
}
EXPECT = {"correct": True, "nan": False, "shape": False, "wrong": False}

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = str(DEV)
env["TORCH_CUDA_ARCH_LIST"] = "9.0"
# per-case dir: mirrors the hashed layout the reward now uses, and keeps one case's
# abandoned build from wedging the next — the very failure this run reproduced
import hashlib as _h

print(f"runner={RUNNER}  device={DEV}\n")
failures = []
for name, code in CASES.items():
    payload = {"test_code": code, "ref_code": REF, "device_idx": 0, "warmup": 2,
               "repeat": 5, "tol": 1e-3, "seed": 0, "num_inputs": 3,
               "torch_cuda_arch_list": "9.0"}
    env["TORCH_EXTENSIONS_DIR"] = "/tmp/gate_probe/" + _h.sha1(code.encode()).hexdigest()[:12]
    p = subprocess.run([PY, RUNNER], input=json.dumps(payload).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=900)
    try:
        res = json.loads(p.stdout.decode() or "{}")
    except Exception:
        res = {"_parse_error": p.stdout.decode()[:200], "_stderr": p.stderr.decode()[-400:]}
    ok = bool(res.get("ok"))
    correct = bool(res.get("correct"))
    sp = res.get("speedup")
    verdict = "PASS" if correct == EXPECT[name] else "FAIL"
    if verdict == "FAIL":
        failures.append(name)
    print(f"[{verdict}] {name:8s} ok={ok!s:5s} correct={correct!s:5s} speedup={sp} "
          f"kind={res.get('kind')}")
    if not ok and res.get("message"):
        print(f"          message: {str(res['message'])[:150]}")
    if res.get("_parse_error") is not None:
        print(f"          stdout: {res['_parse_error']}\n          stderr: {res.get('_stderr')}")

print()
print("RESULT:", "全部符合预期" if not failures else f"不符合预期: {failures}")
sys.exit(1 if failures else 0)
