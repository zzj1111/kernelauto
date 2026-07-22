# `cudaforge/` — the custom RL-for-CUDA layer

Everything in this repo *except* this folder, `experiments/`, `dataset/`, and `tools/`
is upstream [**verl**](https://github.com/volcengine/verl) (ByteDance's RL framework),
unmodified. This folder is the entire project-specific customization: an RL reward that
trains an LLM to write fast, correct CUDA kernels.

It plugs into verl through verl's built-in
`custom_reward_function.path=...` config hook (verl loads the file by path via
`importlib.util.spec_from_file_location` and calls its `compute_score(...)`), so **core
verl is never touched**. These files import only the stdlib (+ `torch` in the runner).

## Files

| File | Role |
|------|------|
| `reward_bench_rubric.py` | **Main reward.** Real benchmark signal (correctness gate + measured speedup) × LLM rubric. Entry point: `compute_score(data_source, solution_str, ground_truth, extra_info=None)`. |
| `reward_rubric_ablation.py` | **Ablation reward.** *No* benchmark is run; an LLM judges correctness + speedup from the code text (6 rubric categories instead of 4). Same `compute_score` interface. |
| `kernel_runner.py` | **Isolated subprocess** invoked by the reward. Reads a JSON payload on stdin, sets `TORCH_CUDA_ARCH_LIST`, `nvcc`-compiles the candidate CUDA extension, checks correctness against the PyTorch reference, and times the speedup. Runs out-of-process so a bad kernel (compile error / segfault) can't crash the trainer. |

> Renamed from the original `CudaForge.py` / `CudaForge_new.py` / `cudaforge_runner.py`.

## Reward formula (in `reward_bench_rubric.py`)

```
reward = (clip(speedup, ≤5) + 0.3) · (1 + λ·(rubric_norm − 0.5))
if major_hacking detected  ->  reward = 0        # anti reward-hacking gate
```

- **Benchmark signal** — correctness is a hard gate; speedup (clipped at 5×) is the main magnitude.
- **Rubric signal** — an LLM-as-judge (a separate vLLM server) scores anti-hacking, bottleneck
  coverage, CUDA-perf quality, and multi-component focus (1–5 each), and flags obvious
  reward-hacking (fake speedups, keeping the heavy op in PyTorch, etc.).
- `λ = 0.7` for the `cudaforgeimprovement` data source, else `1.0`.

## Runtime dependencies (set by the launch scripts, see `../experiments/`)

- **Rubric LLM judge** — an OpenAI-compatible vLLM endpoint, read from env:
  `RUBRIC_VLLM_URL`, `RUBRIC_MODEL_NAME`, `RUBRIC_VLLM_TIMEOUT_SEC`.
- **Benchmark GPU** — the launch scripts reserve a GPU for `kernel_runner.py`
  (`REWARD_CUDA_VISIBLE_DEVICES`).
- Kernel compile/benchmark logs are written under `cudaforge_logs/` (git-ignored).
