"""Downsample hkust-nlp/drkernel-rl-data into a train/held-out split this harness can train on.

Why this dataset with this model: drkernel-8b-coldstart was cold-start SFT'd on Triton kernel
generation, and every one of these 71,996 prompts asks for Triton. Until now the model was being
asked for CUDA C++ by a CUDA dataset — measured at step 10 on the CUDA eval, its apparent 22.3%
correctness was 12.5 points of answers computing the same thing as the PyTorch reference at the
same speed, leaving 9.8% that actually wrote a kernel. Aligning model, data and task removes that
confound.

Three format differences from the CudaForge set, each handled here rather than in the adapters:

  reference code   reward_bench_rubric reads extra_info["answer"]; this dataset carries it in
                   reward_model["ground_truth"]. Copied across, leaving the original in place.

  category axis    `level` is 0 and `type` is empty for all 71,996 rows, so the scratch/improve_l*
                   axis does not exist here. `ops` does: a JSON list of the PyTorch operators the
                   task fuses, over a 454-operator vocabulary. Categories are built from operator
                   FAMILIES, which is a better axis than level anyway — every observation so far
                   has separated fusion-friendly work from library-backed work (conv, matmul),
                   and that is what this splits on.

  splice anchor    the CudaForge prompts end with "OUTPUT RULES (STRICT)"; these end with
                   "Optimize the architecture named <Class> with custom Triton operators!".
                   The class name is NOT always Model — a first pass anchored on the full
                   sentence missed the rows where the reference class is named after its
                   operator (e.g. TripleMarginLoss). Anchoring on the invariant prefix
                   "Optimize the architecture named" covers every row.

Sampling is uniform at random within each category, from the full 71,996 rows.

Sizing. The held-out set is deliberately much larger than the 50 rows used before: that eval had
a three-draw standard error of 0.025-0.08 on a mean of 0.11-0.41, wide enough that no plausible
per-cycle change could be established, and it was the single biggest reason the Teacher spent
twenty cycles unable to confirm a plateau. Held-out rows are cheap here — 72k available, no
overlap to engineer — so the split spends them.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re

import pandas as pd

SRC = "/mnt/data1/zha00175/datasets_hf/drkernel_rl.parquet"
OUT = "/mnt/data1/zha00175/StitchCUDA/dataset/Triton"
SEED = 20260803
N_TRAIN_PER_CAT = 100          # 6 categories -> 600 training rows, 75 steps/epoch at batch 8
N_TEST_PER_CAT = 30            # -> 180 held-out rows, 3.6x the old eval

# Rows are sampled at random from the full pool — no size filter.
#
# A filter was tried and removed. The reasoning behind it: the raw median task is 8,192 elements,
# where a Triton launch costs more than the whole PyTorch op, so a genuine kernel measures SLOWER
# than doing nothing (measured: a real fused silu+erf kernel scored speedup 0.576 while renaming
# the reference class scored 1.024). Restricting to >=1e6 elements was meant to make speedup a
# usable signal.
#
# It was dropped for two reasons. First, the objective here is to learn to write a working kernel
# at all; whether that kernel beats a library call on a given shape is a second-order question,
# and the rubric — not the speedup term — is what has to hold the line against no-op answers
# (measured on the same pair: rubric total 15/20 for the real kernel vs 8/20 for the rename, with
# bottleneck_coverage and perf_quality both scoring 1). Second, the filter had no ceiling and
# admitted tasks of 4.29e9 elements — 17.6 GiB per fp32 tensor — which fail to run at all, even
# for the reference implementation. Sizing the window correctly is its own piece of work and not
# worth doing to serve a secondary signal.
#
# n_elements is still recorded per row, so the question can be revisited from the data later.

_ALLOC = re.compile(r"torch\.(?:randn|rand|zeros|ones|empty|randint)\(([^)]*)\)")


def n_elements(answer: str) -> int:
    """Total elements the reference allocates in get_inputs(). Zero when it cannot be parsed —
    those rows are dropped rather than guessed at."""
    m = re.search(r"def get_inputs\(\):(.*?)(?:\ndef |\Z)", answer, re.S)
    if not m:
        return 0
    env = {k: int(v) for k, v in re.findall(r"^(\w+)\s*=\s*(\d+)\s*$", answer, re.M)}
    tot = 0
    for g in _ALLOC.finditer(m.group(1)):
        n, ok = 1, False
        for t in [t.strip() for t in g.group(1).split(",")]:
            if t.isdigit():
                n *= int(t); ok = True
            elif t in env:
                n *= env[t]; ok = True
            elif re.fullmatch(r"\d+\s*\*\s*\d+", t):
                a, b = map(int, t.split("*")); n *= a * b; ok = True
        if ok:
            tot += n
    return tot

FAMILY = (
    ("conv", ("conv",)),
    ("matmul", ("matmul", "bmm", "linear", "einsum", "mm")),
    ("norm_softmax", ("norm", "softmax", "logsoftmax")),
    ("reduce", ("sum", "mean", "max", "min", "prod", "cumsum", "argmax", "argmin",
                "logsumexp", "std", "var", "median", "count")),
    ("loss", ("loss", "divergence", "entropy", "similarity")),
)


def family_of(op: str) -> str:
    s = op.lower()
    for name, keys in FAMILY:
        if any(k in s for k in keys):
            return name
    return "elementwise"


def category_of(ops) -> str:
    """The heaviest family present. An elementwise tail rides along with almost every task, so
    naming a task by its heaviest operator is what separates 'a conv with a GELU' from 'a chain
    of elementwise ops' — which is the distinction that has actually predicted success so far."""
    fams = {family_of(o) for o in ops}
    for name, _ in FAMILY:                      # FAMILY is ordered heaviest-first
        if name in fams:
            return name
    return "elementwise"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(SRC)
    print(f"  源数据 {len(df):,} 行")

    ops, cats = [], []
    for x in df.extra_info:
        try:
            o = json.loads(x["ops"])
        except Exception:
            o = []
        ops.append(o)
        cats.append(category_of(o))
    df = df.assign(_cat=cats, _ops=ops)
    print("  按最重算子族分类:")
    for k, v in collections.Counter(cats).most_common():
        print(f"    {k:14s} {v:6,}")

    # Sample train and test disjointly per category, from a shuffled order so the two sides are
    # drawn from the same distribution rather than from different regions of the file.
    rng = random.Random(SEED)
    train_idx, test_idx = [], []
    for cat, g in df.groupby("_cat"):
        idx = list(g.index)
        rng.shuffle(idx)
        need = N_TRAIN_PER_CAT + N_TEST_PER_CAT
        if len(idx) < need:
            raise SystemExit(f"category {cat} has {len(idx)} rows,需要 {need}")
        test_idx += idx[:N_TEST_PER_CAT]
        train_idx += idx[N_TEST_PER_CAT:need]

    def build(idx, split):
        sub = df.loc[idx].copy()
        rows = []
        for _, r in sub.iterrows():
            e = dict(r.extra_info)
            # reward_bench_rubric.compute_score returns 0.0 unless extra_info carries "answer".
            e["answer"] = str(r.reward_model["ground_truth"])
            e["category"] = r._cat
            e["ops_list"] = list(r._ops)
            e["split"] = split
            e["n_elements"] = n_elements(e["answer"])
            # task_name enables the per-instance scaffold channel; uuid is the only stable id.
            e["task_name"] = str(e.get("uuid", ""))[:16]
            rows.append(e)
        sub["extra_info"] = rows
        sub["data_source"] = "TritonKernel"
        return sub.drop(columns=["_cat", "_ops"]).reset_index(drop=True)

    tr, te = build(train_idx, "train"), build(test_idx, "test")
    tr.to_parquet(f"{OUT}/train.parquet", index=False)
    te.to_parquet(f"{OUT}/test.parquet", index=False)

    # ---- verify what was written ----
    def report(name, d):
        c = collections.Counter(x["category"] for x in d.extra_info)
        print(f"\n  {name}: {len(d)} 行  {dict(sorted(c.items()))}")
        return {str(x["answer"]).strip() for x in d.extra_info}

    print("\n=== 写出 ===")
    a_tr, a_te = report("train.parquet", tr), report("test.parquet", te)
    print(f"\n  参考实现交集: {len(a_tr & a_te)}  (必须为 0)")
    assert not (a_tr & a_te)
    anchor = "Optimize the architecture named"
    miss = sum(1 for p in list(tr.prompt) + list(te.prompt) if anchor not in p[0]["content"])
    print(f"  缺少 splice 锚点的行: {miss}  (必须为 0)")
    assert miss == 0
    noans = sum(1 for x in list(tr.extra_info) + list(te.extra_info) if not x.get("answer"))
    print(f"  缺少 answer 的行: {noans}  (必须为 0)")
    assert noans == 0
    print("  ✅")


if __name__ == "__main__":
    main()
