"""Rebuild the CudaForge TRAINING set so no test problem appears in it. test.parquet is untouched.

The 80/20 split itself was always correct: `train.parquet` covers 200 KernelBench L1-L3 problems
and `test.parquet` the other 50, stratified per level (80/20, 80/20, 40/10), with a zero-problem
intersection.

What leaked was the improvement half. `train_stitchCUDA_skill.parquet` carries its own
`extra_info.split` field and every one of its 200 rows is labelled `"train"` — but 44 of them are
improvement trajectories for problems that live in the TEST split. Nobody checked the label
against the split, so training on `train_new.parquet` (= train + skill) meant 34 of the 50 test
architectures had been seen during training, in improvement form. Held-out numbers were measuring
a mixture of generalisation and recall.

This drops those 44 rows from training. They are not relocated: the held-out set stays exactly as
it is, so every held-out number ever recorded remains comparable to the ones measured after this
rebuild.

Membership is decided by matching each row's reference implementation (`extra_info.answer`)
against the KernelBench source files, not by trusting any stored label — the stored label is what
was wrong.

Writes (originals left untouched):
  train_new_clean.parquet   200 scratch + 156 improvement = 356 rows, over the 200 train problems

Every row gains `kb_level` / `kb_task` / `kb_split` in extra_info. `level` is deliberately NOT set
on scratch rows: splice.level_of derives the scaffold category from it, and giving scratch rows a
level would silently relabel them as improve_l*.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

KB = "/mnt/data1/zha00175/KernelBench/KernelBench"
D = "/mnt/data1/zha00175/StitchCUDA/dataset/CudaForge"


def norm(s: str) -> str:
    """Whitespace-insensitive form for matching. Three train rows differ from the upstream file
    only in blank lines / trailing spaces; comparing raw text would drop them."""
    return "\n".join(l.rstrip() for l in s.strip().splitlines() if l.strip())


def ei(x):
    return dict(x) if isinstance(x, dict) else json.loads(x)


def kb_index():
    idx = {}
    for lv in (1, 2, 3):
        for p in sorted(glob.glob(f"{KB}/level{lv}/*.py")):
            name = os.path.basename(p)[:-3]
            idx[norm(open(p, errors="ignore").read())] = (lv, int(name.split("_")[0]), name)
    return idx


def main():
    KBI = kb_index()
    assert len(KBI) == 250, f"expected 250 KernelBench L1-L3 problems, found {len(KBI)}"

    scratch = pd.read_parquet(f"{D}/train.parquet")
    skill = pd.read_parquet(f"{D}/train_stitchCUDA_skill.parquet")
    test = pd.read_parquet(f"{D}/test.parquet")

    def ids_of(df):
        return [KBI.get(norm(ei(x).get("answer", "")), (None, None, None)) for x in df.extra_info]

    s_id, k_id, t_id = ids_of(scratch), ids_of(skill), ids_of(test)
    TEST = {(lv, ix) for lv, ix, _ in t_id if lv}
    assert len(TEST) == 50, f"test must be 50 problems, got {len(TEST)}"

    # The train split is "every KernelBench problem that is not in TEST". Deriving it this way
    # rather than from train.parquet's own matches keeps the three format-drifted rows on the
    # train side instead of silently dropping them.
    TRAIN = {(lv, ix) for lv, ix, _ in KBI.values()} - TEST
    assert len(TRAIN) == 200, f"train must be 200 problems, got {len(TRAIN)}"

    def annotate(df, ids, split):
        rows = []
        for x, (lv, ix, name) in zip(df.extra_info, ids):
            e = ei(x)
            e["kb_level"] = lv
            e["kb_task"] = name
            e["kb_split"] = split
            rows.append(e)
        out = df.copy()
        out["extra_info"] = rows
        return out

    def belongs(ids, target):
        return [i for i, (lv, ix, _) in enumerate(ids) if (lv, ix) in target]

    bad = belongs(s_id, TEST)
    assert not bad, f"train.parquet contains test problems at rows {bad}"

    keep = belongs(k_id, TRAIN)
    drop = belongs(k_id, TEST)
    assert len(keep) + len(drop) == len(skill), "some improvement rows matched neither split"
    print(f"=== 改进集 {len(skill)} 行 ===")
    print(f"  保留(属于 train 划分): {len(keep)}")
    print(f"  剔除(属于 test  划分): {len(drop)}   <- 泄漏源,仅从训练集移除,测试集不动")
    dropped_tasks = sorted({k_id[i][2] for i in drop})
    print(f"  被剔除的题目 {len(dropped_tasks)} 个,前 8: {dropped_tasks[:8]}")

    train_out = pd.concat([
        annotate(scratch, s_id, "train"),
        annotate(skill.iloc[keep].reset_index(drop=True), [k_id[i] for i in keep], "train"),
    ], ignore_index=True)
    train_out.to_parquet(f"{D}/train_new_clean.parquet", index=False)

    # --- verify what was written, not what was intended ---
    def report(name, df):
        ids = ids_of(df)
        probs = {(lv, ix) for lv, ix, _ in ids if lv}
        cats, lvls = {}, {}
        for x in df.extra_info:
            lv = ei(x).get("level")
            c = "scratch" if lv is None else f"improve_l{int(float(lv))}"
            cats[c] = cats.get(c, 0) + 1
        for lv, _, _ in ids:
            lvls[f"L{lv}"] = lvls.get(f"L{lv}", 0) + 1
        print(f"  {name:26s} {len(df):3d} 行  {len(probs):3d} 题")
        print(f"      scaffold 类别 {dict(sorted(cats.items()))}")
        print(f"      KernelBench level {dict(sorted(lvls.items()))}")
        return probs

    print("\n=== 写出结果 ===")
    ptr = report("train_new_clean.parquet", train_out)
    pte = report("test.parquet (未改动)", test)

    print("\n=== 泄漏检查(按 KernelBench 题目) ===")
    inter = ptr & pte
    print(f"  train ∩ test = {len(inter)}   (必须为 0)")
    assert not inter, f"LEAK: {sorted(inter)[:5]}"
    print(f"  train ∪ test = {len(ptr | pte)} / 250")

    # Belt and braces: the reference implementations themselves must not coincide, which is the
    # form the original leak actually took.
    ta = {norm(ei(x).get("answer", "")) for x in train_out.extra_info}
    ea = {norm(ei(x).get("answer", "")) for x in test.extra_info}
    print(f"  参考实现交集 = {len(ta & ea)}   (必须为 0;旧 train_new 是 34)")
    assert not (ta & ea), "LEAK: identical reference implementation on both sides"
    print("  ✅ 无泄漏")


if __name__ == "__main__":
    main()
