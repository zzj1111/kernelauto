import subprocess
import json
import re, os, sys, time
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Same cap, same variable, same reason as reward_bench_rubric.py: each runner creates its own
# CUDA context, and on 2026-08-04 243 concurrent ones wedged the GPU driver for every user of
# the node. compute_score() here never calls bench(), so this guards the offline/debug path the
# file deliberately keeps — the copy that was made before the cap existed and never got it.
_MAX_CONCURRENT = int(os.environ.get("CUDAFORGE_MAX_CONCURRENT_RUNNERS", "12"))
_RUNNER_SLOTS = threading.Semaphore(_MAX_CONCURRENT)

try:
    import requests
except Exception:
    requests = None


_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


# ============================================================
# Rubric Prompts (NOW: 6 categories, each 1-5, + major_hacking)
# - correctness_estimate: infer from code text
# - speedup_estimate: infer from code text
# ============================================================

RUBRIC_PROMPT_FROM_SCRATCH = r"""
You are a strict rubric judge for GRPO training on CUDA code generation.

IMPORTANT CHANGE (ABLATION):
- You MUST judge BOTH functional correctness and speedup potential by inspecting the code itself.
- There will be NO external benchmark signal used in reward computation.
- Your scoring must therefore include:
  (A) correctness_estimate (1-5)
  (B) speedup_estimate (1-5)

You are given:
- REFERENCE CODE (PyTorch baseline implementation)
- CANDIDATE CODE (CUDA / extension-based implementation)

Score SIX rubric aspects (1-5 integers):
- anti_hacking (1-5)
- bottleneck_coverage (1-5)
- cuda_perf_quality (1-5)
- multi_component_focus (1-5)
- correctness_estimate (1-5)
- speedup_estimate (1-5)

If major_hacking=true, set anti_hacking=1 and total=0.

============================================================
(1) anti_hacking (1-5)
5: No signs of hacking. Forward relies on custom CUDA for core compute. No suspicious shortcuts.
4: Mostly clean. Minor specialization (e.g., limited shapes) but still genuine CUDA acceleration path.
3: Some suspicious patterns (custom CUDA exists but impact unclear; heavy ops partially left to PyTorch).
2: Strong suspicion of gaming (decorative CUDA; most heavy compute still PyTorch), but not fully blatant.
1: Obvious hacking => MUST set major_hacking=true.

Set major_hacking=true if any obvious:
- hardcoded/constant outputs unrelated to inputs
- caching/memoization keyed on input patterns to cheat
- branching on specific input values to cheat
- Python/CPU tricks as the primary compute path
- fake CUDA acceleration (keeps real heavy ops unchanged while CUDA is decorative)

============================================================
(2) correctness_estimate (1-5)
Estimate the probability that the CANDIDATE is functionally correct relative to the REFERENCE.

Evidence to consider:
- Output tensor shapes match reference
- Dtype/device handling correct
- Numerical behavior plausible (e.g., correct reduction axes, correct padding/stride for conv)
- Handles batch/channel dimensions correctly
- No obvious missing operations or wrong semantics
- No illegal assumptions about fixed shapes unless clearly aligned with reference constraints

Scoring:
5: Very likely correct. Clear 1-to-1 mapping with reference logic; careful shape/dtype handling.
4: Likely correct. Minor uncertainty, but overall consistent and complete.
3: Unclear / moderate risk. Some parts missing or ambiguous, but could be correct.
2: Likely incorrect. Noticeable semantic mismatch, missing critical computation.
1: Almost certainly incorrect. Major wrong outputs, missing key ops, inconsistent dimensions.

============================================================
(3) speedup_estimate (1-5)
Estimate whether the CANDIDATE likely provides speedup over the REFERENCE in real GPU execution.

Evidence to consider:
- Uses CUDA custom kernel for core compute (not just wrapper)
- Avoids falling back to PyTorch heavy ops
- Fuses multiple ops or reduces memory traffic
- Uses efficient layouts / fewer kernel launches
- Minimizes host-device sync, avoids Python loops in forward

Scoring:
5: Very likely meaningful speedup (fusion/tiling/shared memory/tensor core usage or major traffic reduction).
4: Likely speedup (reasonable CUDA kernel, avoids heavy PyTorch ops).
3: Some speedup possible but uncertain (basic kernel exists but may be naive).
2: Unlikely speedup (decorative kernel, heavy ops remain in PyTorch, too many launches).
1: No speedup or likely slower than reference.

============================================================
(4) bottleneck_coverage (1-5)
Judge whether the candidate chooses to optimize the REAL bottlenecks in the reference model.
5: Optimizes/fuses core bottlenecks (Conv/Matmul/Attention/Norm/Softmax), or meaningful fusion pipeline (e.g., Conv+ReLU+Bias).
4: Optimizes at least one major expensive operator; decisions are reasonable.
3: Some bottleneck effort exists but mixed with superficial targets; not clearly focused on the dominant cost.
2: Mostly optimizes cheap ops only (relu/add/bias) while main bottleneck remains unchanged.
1: No meaningful bottleneck coverage.

============================================================
(5) cuda_perf_quality (1-5)
Judge the performance-quality of CUDA implementation (beyond "a CUDA kernel exists").
Evaluate evidence of hardware-aware optimization practices, including:
- Coalesced & aligned global memory access patterns (avoid scattered loads/stores)
- Use of shared memory to reduce global traffic; avoid redundant global reads
- Tiling strategy to improve parallelism; appropriate grid/block sizes
- Optional/advanced: async shared-memory prefetching / pipelining (cp.async style), overlap compute & memory
- Optional/advanced: tensor cores / mma instructions when applicable
- Optional/advanced: staging/pipeline design for multi-stage kernels; use CUDA cores & tensor cores concurrently when possible
- Optional/advanced: PTX-level optimizations

Scoring:
5: Expert-grade intent. Clear tiling + shared memory reuse + coalesced access. Possibly pipelining/async copy and/or tensor cores/mma/PTX when applicable.
4: Good performance awareness. Implements at least one major optimization (tiling/shared memory/coalescing/vectorization) reasonably.
3: Basic CUDA kernel present and plausible, but mostly naive. Limited tiling/shared memory usage.
2: Very naive/high-overhead design. Poor memory access patterns and little reuse.
1: No meaningful CUDA performance reasoning visible; essentially minimal CUDA wrapper.

============================================================
(6) multi_component_focus (1-5)
Judge whether the candidate appropriately targets MULTI-COMPONENT models.
Some reference architectures include multiple components (e.g., several conv/pool/linear blocks).
We want to encourage optimizing MORE and/or MORE BOTTLENECK components, not just the last trivial op.

Scoring:
5: If model has multiple heavy components, candidate optimizes multiple or the dominant bottleneck(s); avoids only touching the last minor op.
4: Optimizes at least one dominant bottleneck; may not cover all, but clearly targets key cost.
3: Optimizes something meaningful, but coverage is limited; might focus on one part when multiple exist.
2: Focuses mostly on minor/tail ops while leaving main bottlenecks unchanged (e.g., only last add/relu).
1: No evidence of appropriate focus; essentially ignores multi-component bottlenecks.

Notes:
- If the reference model is truly single-op / single-bottleneck, do NOT penalize for not optimizing multiple ops:
  In that case, assign at least 3 if it targets the single bottleneck reasonably.

============================================================
Output ONLY JSON:
{
  "anti_hacking": <int 1-5>,
  "correctness_estimate": <int 1-5>,
  "speedup_estimate": <int 1-5>,
  "bottleneck_coverage": <int 1-5>,
  "cuda_perf_quality": <int 1-5>,
  "multi_component_focus": <int 1-5>,
  "major_hacking": <true/false>,
  "total": <int>
}

Constraints:
- Integers only (1-5).
- If major_hacking=false:
  total = anti_hacking + correctness_estimate + speedup_estimate
          + bottleneck_coverage + cuda_perf_quality + multi_component_focus.
- If major_hacking=true: total=0 and anti_hacking=1.
- Output JSON only, no extra text.
"""


RUBRIC_PROMPT_REFINEMENT = r"""
You are a strict rubric judge for GRPO training on iterative CUDA code refinement.

IMPORTANT CHANGE (ABLATION):
- You MUST judge BOTH functional correctness and speedup potential by inspecting the code itself.
- There will be NO external benchmark signal used in reward computation.
- Your scoring must therefore include:
  (A) correctness_estimate (1-5)
  (B) speedup_estimate (1-5)

You receive an improvement-context TXT that may include:
- GPU info, previous kernels, architecture file
- optimization instructions / modification plan
- output rules
- sometimes ERROR_LOG + previous CUDA candidate to audit

Score SIX rubric aspects (1-5 integers):
- anti_hacking (1-5)
- correctness_estimate (1-5)
- speedup_estimate (1-5)
- bottleneck_effectiveness (1-5)
- instruction_alignment (1-5)
- optimization_scope_focus (1-5)

If major_hacking=true, set anti_hacking=1 and total=0.

============================================================
(1) anti_hacking (1-5)
5: Clean refinement, no cheating patterns, genuine CUDA acceleration path.
4: Mostly clean; minor suspicious specialization but still genuine.
3: Some suspicious patterns (decorative CUDA / heavy ops partially unchanged) but not clear cheating.
2: Strong suspicion of gaming (heavy compute largely PyTorch, suspicious shortcuts).
1: Obvious hacking => MUST set major_hacking=true.

major_hacking=true if any obvious:
- hardcoded/constant outputs unrelated to inputs
- caching/memoization keyed on inputs to cheat
- branching on specific input values to cheat
- Python/CPU tricks as main compute path
- fake CUDA acceleration (keeps real heavy ops unchanged but claims speedup)
- newly introduced cheating patterns not present previously (if described in TXT)

============================================================
(2) correctness_estimate (1-5)
Estimate the probability that the NEW candidate is functionally correct.
Use evidence:
- The change preserves semantics vs reference
- Fixes described ERROR_LOG without breaking computation
- Handles shapes/dtypes/devices correctly
- No missing critical ops

Scoring:
5: Very likely correct; fixes issues cleanly without semantic regressions.
4: Likely correct; small uncertainties but consistent.
3: Unclear; medium risk of hidden bug.
2: Likely incorrect; mismatch in semantics/shapes.
1: Almost certainly incorrect.

============================================================
(3) speedup_estimate (1-5)
Estimate whether the NEW candidate likely speeds up execution.
Evidence:
- Fewer global loads / shared memory reuse
- Better tiling / occupancy / fewer launches
- Eliminates heavy PyTorch ops
- Uses tensor cores / pipelining when applicable

Scoring:
5: Very likely meaningful speedup.
4: Likely speedup.
3: Uncertain.
2: Unlikely speedup.
1: No speedup or likely slower.

============================================================
(4) bottleneck_effectiveness (1-5)
Judge whether the refinement improves the stated bottleneck/perf limiter OR fixes the ERROR_LOG root cause effectively.

Use CUDA expert criteria as evidence of effectiveness:
- Improved global memory access pattern: coalesced/aligned access, fewer scattered loads/stores
- Reduced global memory traffic: reuse via shared memory, fewer redundant loads
- Better parallelism: tiling strategy, appropriate grid/block sizing, better occupancy awareness
- Optional/advanced: async copy / pipelined prefetch; overlap compute & memory
- Optional/advanced: tensor core utilization (mma) when applicable
- Optional/advanced: multi-stage pipeline design, concurrent use of CUDA cores & tensor cores
- Optional/advanced: PTX-level micro-optimizations

Scoring:
5: Directly resolves bottleneck OR fixes ERROR_LOG root cause cleanly; improvement is clear and substantial.
4: Meaningful improvement addressing bottleneck/root cause; likely improves performance materially.
3: Partial/moderate improvement; addresses bottleneck somewhat but incomplete.
2: Mostly superficial edits; bottleneck/root cause not truly improved.
1: No meaningful improvement over previous design.

============================================================
(5) instruction_alignment (1-5)
Judge compliance with the improvement TXT request and output rules.
5: Implements the request clearly and directly; changes match optimization instructions OR resolve the ERROR_LOG root issue when present; respects output rules.
4: Implements most key parts; minor omissions.
3: Partial compliance; some required items missing or unclear.
2: Weak alignment; changes mostly unrelated.
1: Ignores the request.

============================================================
(6) optimization_scope_focus (1-5)
We want refinement to focus on the MOST IMPORTANT bottleneck or the TRUE root cause, not only minor/tail changes.

Scoring:
5: Targets dominant hotspot / true root cause; avoids trivial edits.
4: Mostly targets important parts; minor distractions.
3: Mixed focus.
2: Focuses mainly on minor pieces.
1: Irrelevant.

============================================================
Output ONLY JSON:
{
  "anti_hacking": <int 1-5>,
  "correctness_estimate": <int 1-5>,
  "speedup_estimate": <int 1-5>,
  "bottleneck_effectiveness": <int 1-5>,
  "instruction_alignment": <int 1-5>,
  "optimization_scope_focus": <int 1-5>,
  "major_hacking": <true/false>,
  "total": <int>
}

Constraints:
- Integers only (1-5).
- If major_hacking=false:
  total = anti_hacking + correctness_estimate + speedup_estimate
          + bottleneck_effectiveness + instruction_alignment + optimization_scope_focus.
- If major_hacking=true: total=0 and anti_hacking=1.
- Output JSON only, no extra text.
"""


RUBRIC_JSON_ONLY_ENFORCER = r"""
IMPORTANT:
- Output MUST be a single JSON object starting with '{' and ending with '}'.
- No explanations, no reasoning, no extra tokens.
- Do NOT wrap with ``` or any other formatting.
"""


# ============================================================
# Common Helpers
# ============================================================

def _extract_python_code(solution_str: str) -> str:
    m = _CODEBLOCK_RE.search(solution_str)
    return (m.group(1) if m else solution_str).strip()

def _safe_tail(s: str, n: int) -> str:
    if not s:
        return ""
    return s[-n:] if len(s) > n else s

def _write_jsonl(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _decode_maybe_bytes(x, limit: int) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        s = x.decode("utf-8", errors="replace")
    else:
        s = str(x)
    return _safe_tail(s, limit)

def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


# ============================================================
# Improvement TXT safe-trim (token saving) — dual-format compatible
# ============================================================

def safe_trim_improvement_txt(txt: str, max_chars: int = 14000) -> str:
    if not txt:
        return ""

    raw = txt.strip()
    if len(raw) <= max_chars:
        return raw

    def _grab(pattern: str, flags=0) -> str:
        m = re.search(pattern, raw, flags)
        return m.group(0).strip() if m else ""

    def _cap(s: str, cap: int) -> str:
        if not s:
            return ""
        s = s.strip()
        return s[:cap] if len(s) > cap else s

    prefix = _cap(raw, 1200)

    error_log = _grab(
        r"ERROR_LOG:\s*[\s\S]*?(?=\n\s*\n|\nPyTorch reference|\nCUDA candidate|\n#\s*-+\s*Previously generated kernels|\n\[ARCHITECTURE FILE\]|\Z)",
        flags=re.IGNORECASE,
    )
    error_log = _cap(error_log, 3500)

    pyt_ref = _grab(
        r"PyTorch reference\s*\(ground truth\)\s*:\s*[\s\S]*?(?=\n\s*\n|\nCUDA candidate|\n\[ARCHITECTURE FILE\]|\n\[optimization instructions\]|\Z)",
        flags=re.IGNORECASE,
    )
    pyt_ref = _cap(pyt_ref, 3000)

    cuda_cand_full = _grab(
        r"CUDA candidate\s*\(to audit\)\s*:\s*[\s\S]*?(?=\n\s*\nFollow the Rules|\n\s*\nOUTPUT RULES|\Z)",
        flags=re.IGNORECASE,
    )
    cuda_cand = ""
    if cuda_cand_full:
        cand_codeblock = _grab(r"```python[\s\S]*?```", flags=re.IGNORECASE)
        if cand_codeblock:
            cuda_cand = "CUDA candidate (to audit):\n" + cand_codeblock.strip()
        else:
            cuda_cand = "CUDA candidate (to audit):\n" + _cap(cuda_cand_full, 6500)
        cuda_cand = _cap(cuda_cand, 6500)

    prev_kernels = _grab(
        r"#\s*-+\s*Previously generated kernels\s*-+[\s\S]*?(?=\n\s*\n|\n\[ARCHITECTURE FILE\]|\n\[optimization instructions\]|\Z)",
        flags=re.IGNORECASE,
    )
    prev_kernels = _cap(prev_kernels, 1200)

    opt_instr = _grab(
        r"\[optimization instructions\][\s\S]*?\{[\s\S]*?\}\s*",
        flags=re.IGNORECASE,
    )
    opt_instr = _cap(opt_instr, 1800)

    arch_file = ""
    arch_section = _grab(
        r"\[ARCHITECTURE FILE\][\s\S]*?(?=\n\s*\[optimization instructions\]|\n\s*GOAL|\n\s*OUTPUT RULES|\Z)",
        flags=re.IGNORECASE,
    )
    if arch_section:
        codeblock = _grab(r"```python[\s\S]*?```", flags=re.IGNORECASE)
        arch_file = (codeblock.strip() if codeblock else _cap(arch_section, 6500))
    else:
        codeblock = _grab(r"```python[\s\S]*?```", flags=re.IGNORECASE)
        arch_file = _cap(codeblock, 6500)

    goal = _grab(r"\nGOAL[\s\S]*?(?=\n\s*OUTPUT RULES|\Z)", flags=re.IGNORECASE)
    goal = _cap(goal, 1400)

    output_rules = _grab(r"OUTPUT RULES[\s\S]*?(?=\n```python|\Z)", flags=re.IGNORECASE)
    output_rules = _cap(output_rules, 1800)

    chunks = [prefix]

    if error_log:
        chunks.append("\n\n" + error_log)
    if pyt_ref:
        chunks.append("\n\n" + pyt_ref)
    if cuda_cand:
        chunks.append("\n\n" + cuda_cand)

    if prev_kernels:
        chunks.append("\n\n" + prev_kernels)
    if opt_instr:
        chunks.append("\n\n" + opt_instr)
    if arch_file and "[ARCHITECTURE FILE]" in raw:
        chunks.append("\n\n[ARCHITECTURE FILE]\n" + arch_file)
    if goal:
        chunks.append("\n\n" + goal)
    if output_rules:
        chunks.append("\n\n" + output_rules)

    out = "\n".join([c for c in chunks if c.strip()]).strip()

    if len(out) > max_chars:
        head = out[: int(max_chars * 0.75)]
        tail = out[-int(max_chars * 0.20):]
        out = head.rstrip() + "\n\n...[TRIMMED]...\n\n" + tail.lstrip()

    return out


# ============================================================
# vLLM Rubric Judge Client (OpenAI-compatible)
# ============================================================

def _call_vllm_chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: Optional[int] = None,
    max_tokens: int = 512,
) -> str:
    url = url or os.environ.get("RUBRIC_VLLM_URL", "").strip()
    model = model or os.environ.get("RUBRIC_MODEL_NAME", "").strip()

    if not url or not model:
        raise RuntimeError("Rubric vLLM not configured. Please set RUBRIC_VLLM_URL and RUBRIC_MODEL_NAME.")

    if timeout_sec is None:
        timeout_sec = int(os.environ.get("RUBRIC_VLLM_TIMEOUT_SEC", "30"))

    if requests is None:
        raise RuntimeError("requests is not available. Please install requests in your env.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected vLLM response format: {data}")


def _parse_rubric_json(text: str) -> Dict[str, Any]:
    raw_all = (text or "").strip()
    if not raw_all:
        raise ValueError("Rubric JSON parse failed: empty response")

    raw = raw_all
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    def _extract_last_balanced_json(s: str) -> Optional[str]:
        opens = [i for i, ch in enumerate(s) if ch == "{"]
        if not opens:
            return None

        for start in reversed(opens):
            depth = 0
            in_str = False
            esc = False
            for j in range(start, len(s)):
                ch = s[j]

                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == '"':
                        in_str = False
                    continue

                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : j + 1]
        return None

    candidate = _extract_last_balanced_json(raw)
    if candidate is None:
        candidate = _extract_last_balanced_json(raw_all)

    if candidate is None:
        raise ValueError(f"Rubric JSON parse failed: no JSON object found | raw_tail={_safe_tail(raw_all, 500)}")

    try:
        obj = json.loads(candidate)
    except Exception as ex:
        raise ValueError(
            f"Rubric JSON parse failed: extracted JSON invalid: {repr(ex)} | extracted_tail={_safe_tail(candidate, 500)}"
        )

    if not isinstance(obj, dict):
        raise ValueError(f"Rubric output is not a JSON object: type={type(obj)}")

    return obj


# ============================================================
# Rubric-only Reward Mapping
# - total in [6,30] -> normalized in [0,1]
# - final reward = 3 * normalized
# ============================================================

def _normalize_rubric_total_6_to_30(total: int) -> float:
    total = int(total)
    if total < 6:
        total = 6
    if total > 30:
        total = 30
    return (total - 6) / 24.0


def _default_neutral_rubric(data_source: str) -> Dict[str, Any]:
    """
    Neutral rubric for fallback:
      all 3s => total=18 => normalized=(18-6)/24=0.5 => final=1.5
    """
    ds = str(data_source).lower()
    if ds == "cudaforgeimprovement":
        return {
            "anti_hacking": 3,
            "correctness_estimate": 3,
            "speedup_estimate": 3,
            "bottleneck_effectiveness": 3,
            "instruction_alignment": 3,
            "optimization_scope_focus": 3,
            "major_hacking": False,
            "total": 18,
        }
    else:
        return {
            "anti_hacking": 3,
            "correctness_estimate": 3,
            "speedup_estimate": 3,
            "bottleneck_coverage": 3,
            "cuda_perf_quality": 3,
            "multi_component_focus": 3,
            "major_hacking": False,
            "total": 18,
        }


def _compute_rubric_only_reward(
    *,
    rubric_obj: Optional[Dict[str, Any]],
    data_source: str,
) -> Tuple[float, Dict[str, Any]]:
    """
    Reward is computed ONLY from rubric model output.

    Policy:
    - If rubric_obj missing => fallback neutral used by caller (should not happen)
    - If major_hacking => reward=0
    - Else: reward = 3 * normalize(total in [6,30])
    """
    dbg: Dict[str, Any] = {}

    if rubric_obj is None:
        dbg["gate"] = "no_rubric"
        return 0.0, dbg

    major_hacking = bool(rubric_obj.get("major_hacking", False))
    dbg["major_hacking"] = major_hacking
    if major_hacking:
        dbg["gate"] = "major_hacking"
        return 0.0, dbg

    total = rubric_obj.get("total", None)

    # If missing total, sum known keys
    if total is None:
        total = 0
        for k in (
            "anti_hacking",
            "correctness_estimate",
            "speedup_estimate",
            "bottleneck_coverage",
            "cuda_perf_quality",
            "multi_component_focus",
            "bottleneck_effectiveness",
            "instruction_alignment",
            "optimization_scope_focus",
        ):
            if k in rubric_obj:
                try:
                    total += int(rubric_obj[k])
                except Exception:
                    pass

    try:
        total_int = int(total)
    except Exception:
        total_int = 18  # neutral fallback

    norm = _normalize_rubric_total_6_to_30(total_int)
    reward = 3.0 * float(norm)

    dbg["rubric_total"] = total_int
    dbg["rubric_norm_0_1"] = norm
    dbg["rubric_reward"] = reward
    dbg["data_source"] = str(data_source)
    return reward, dbg


def _build_rubric_user_prompt_from_scratch(reference_code: str, candidate_code: str) -> str:
    return (
        "REFERENCE CODE:\n```python\n"
        f"{reference_code}\n"
        "```\n\n"
        "CANDIDATE CODE:\n```python\n"
        f"{candidate_code}\n"
        "```\n"
    )


def _build_rubric_user_prompt_refinement_from_txt(
    improvement_txt: str,
    candidate_code: str,
    reference_code: Optional[str] = None,
) -> str:
    parts = []
    parts.append("IMPROVEMENT CONTEXT TXT:\n")
    parts.append((improvement_txt or "").strip() + "\n")

    if reference_code:
        parts.append("\nREFERENCE (optional):\n```python\n")
        parts.append(reference_code.strip() + "\n```\n")

    parts.append("\nCANDIDATE (new):\n```python\n")
    parts.append(candidate_code.strip() + "\n```\n")

    return "".join(parts)


def _rubric_log_path(log_dir: str = "./cudaforge_logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "rubric_logs", "rubric_judge.jsonl")


def _run_rubric_judge(
    data_source: str,
    *,
    reference_code: str,
    candidate_code: str,
    extra_info: Optional[Dict[str, Any]] = None,
    log_dir: str = "./cudaforge_logs",
) -> Dict[str, Any]:
    """
    Enforces JSON-only output (system prompt + enforcer).
    Retry once; fallback to neutral rubric.
    """
    ts = _now_ts()
    pid = os.getpid()
    log_path = _rubric_log_path(log_dir)

    def _log(obj: Dict[str, Any]) -> None:
        obj.setdefault("ts", ts)
        obj.setdefault("pid", pid)
        obj.setdefault("data_source", data_source)
        _write_jsonl(log_path, obj)

    extra_info = extra_info or {}
    ds = str(data_source).lower()

    if ds == "cudaforgeimprovement":
        base_system_prompt = RUBRIC_PROMPT_REFINEMENT.strip() + "\n\n" + RUBRIC_JSON_ONLY_ENFORCER.strip()
        improvement_txt_raw = str(extra_info.get("question", "") or "")
        improvement_txt = safe_trim_improvement_txt(improvement_txt_raw, max_chars=14000)
        base_user_prompt = _build_rubric_user_prompt_refinement_from_txt(
            improvement_txt=improvement_txt,
            candidate_code=candidate_code,
            reference_code=reference_code,
        )
    else:
        base_system_prompt = RUBRIC_PROMPT_FROM_SCRATCH.strip() + "\n\n" + RUBRIC_JSON_ONLY_ENFORCER.strip()
        base_user_prompt = _build_rubric_user_prompt_from_scratch(
            reference_code=reference_code,
            candidate_code=candidate_code,
        )

    def _call_and_parse(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw = _call_vllm_chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=320,
        )
        return _parse_rubric_json(raw)

    try:
        obj = _call_and_parse(base_system_prompt, base_user_prompt)
        _log({"ok": True, "attempt": 1, "rubric_obj": obj})
        return obj
    except Exception as ex1:
        _log({"ok": False, "attempt": 1, "error": repr(ex1), "traceback": traceback.format_exc()})

    try:
        obj = _call_and_parse(base_system_prompt, base_user_prompt)
        _log({"ok": True, "attempt": 2, "rubric_obj": obj})
        return obj
    except Exception as ex2:
        _log({"ok": False, "attempt": 2, "error": repr(ex2), "traceback": traceback.format_exc()})

    default_obj = _default_neutral_rubric(data_source)
    _log({"ok": False, "attempt": "fallback", "fallback_default_used": True, "fallback_rubric_obj": default_obj})
    return default_obj


# ============================================================
# (Optional) Keep bench for offline evaluation/debug
# NOTE: In ablation, compute_score() will NOT call it.
# ============================================================

def bench(
    solution_str,
    reference_str,
    device_idx=0,
    warmup=5,
    repeat=20,
    tol=1e-3,
    timeout_sec=600,
    *,
    log_dir: str = "./cudaforge_logs",
    log_on_success: bool = False,
    max_code_chars: int = 8000,
    max_io_chars: int = 20000,
    num_inputs: int = 5,
    ninja_jobs: int = 16,
    max_jobs: int = 16,
    ext_dir_mode: str = "shared",
    torch_cuda_arch_list: str = "9.0",
):
    t_start = time.time()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    log_path = os.path.join(log_dir, f"bench_{ts}_pid{pid}.jsonl")

    def _log(record: dict) -> None:
        record.setdefault("ts", ts)
        record.setdefault("pid", pid)
        record.setdefault("elapsed_sec", round(time.time() - t_start, 6))
        record.setdefault("timeout_sec", timeout_sec)
        _write_jsonl(log_path, record)

    try:
        test_code = _extract_python_code(solution_str)
    except Exception as ex:
        _log({
            "phase": "error",
            "ok": False,
            "kind": "code_extract_error",
            "message": f"Failed to extract python code: {repr(ex)}",
            "traceback": traceback.format_exc(),
        })
        return 0, 0.0

    if "class ModelNew" not in test_code and "class Model(" in test_code:
        test_code = test_code.replace("class Model(", "class ModelNew(", 1)

    payload = {
        "ref_code": reference_str,
        "test_code": test_code,
        "warmup": int(warmup),
        "repeat": int(repeat),
        "tol": float(tol),
        "seed": 100,
        "device_idx": 0,
        "debug_dir": None,
        "num_inputs": int(num_inputs),
        "torch_cuda_arch_list": str(torch_cuda_arch_list),
    }

    runner = os.path.abspath("./cudaforge/kernel_runner.py")
    cmd = [sys.executable, runner]

    env = os.environ.copy()
    reward_vis = env.get("REWARD_CUDA_VISIBLE_DEVICES", None)
    if reward_vis is not None:
        env["CUDA_VISIBLE_DEVICES"] = reward_vis

    env["MAX_JOBS"] = str(max_jobs)
    env["NINJA_NUM_JOBS"] = str(ninja_jobs)
    env["TORCH_CUDA_ARCH_LIST"] = str(torch_cuda_arch_list)

    if ext_dir_mode == "shared":
        vis = env.get("CUDA_VISIBLE_DEVICES", "unknown")
        env["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_ext_cache_reward_cuda{vis}"
    else:
        env["TORCH_EXTENSIONS_DIR"] = f"/dev/shm/torch_ext_{pid}_{ts}"

    runner_debug_dir = os.path.join(log_dir, "runner_debug", f"{ts}_pid{pid}")
    payload["debug_dir"] = runner_debug_dir

    payload_for_log = {
        **payload,
        "ref_code": _safe_tail(str(payload.get("ref_code", "")), max_code_chars),
        "test_code": _safe_tail(str(payload.get("test_code", "")), max_code_chars),
    }

    _log({
        "phase": "start",
        "ok": True,
        "cmd": cmd,
        "env": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "REWARD_CUDA_VISIBLE_DEVICES": os.environ.get("REWARD_CUDA_VISIBLE_DEVICES"),
            "TORCH_EXTENSIONS_DIR": env.get("TORCH_EXTENSIONS_DIR"),
            "MAX_JOBS": env.get("MAX_JOBS"),
            "NINJA_NUM_JOBS": env.get("NINJA_NUM_JOBS"),
            "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST"),
        },
        "payload_meta": {k: payload.get(k) for k in (
            "device_idx", "warmup", "repeat", "tol", "seed",
            "num_inputs", "debug_dir", "torch_cuda_arch_list"
        )},
        "payload_tail": payload_for_log,
    })

    res = None
    out = ""
    err = ""
    try:
      with _RUNNER_SLOTS:                      # bounded GPU-context creation; see _MAX_CONCURRENT
        p = subprocess.run(
            cmd,
            input=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=timeout_sec,
            check=False,
        )

        out = p.stdout.decode("utf-8", errors="replace").strip()
        err = p.stderr.decode("utf-8", errors="replace").strip()

        if out:
            try:
                res = json.loads(out)
            except json.JSONDecodeError:
                res = {"ok": False, "kind": "bad_json", "message": "Runner stdout is not JSON."}
        else:
            res = {"ok": False, "kind": "no_output", "message": "Runner returned empty stdout."}

        res.setdefault("returncode", p.returncode)
        if err:
            res.setdefault("stderr_tail", _safe_tail(err, 2000))

        ok = bool(res.get("ok", False))
        if (not ok) or log_on_success:
            _log({
                "phase": "finish",
                "timeout": False,
                "runner_returncode": p.returncode,
                "runner_stdout_tail": _safe_tail(out, max_io_chars),
                "runner_stderr_tail": _safe_tail(err, max_io_chars),
                "runner_json": res,
                "runner_dump_path": res.get("dump_path"),
                "runner_debug_dir": runner_debug_dir,
            })

    except subprocess.TimeoutExpired as e:
        partial_out = _decode_maybe_bytes(getattr(e, "stdout", None), max_io_chars)
        partial_err = _decode_maybe_bytes(getattr(e, "stderr", None), max_io_chars)

        _log({
            "phase": "timeout",
            "timeout": True,
            "message": "Runner timed out.",
            "cmd": cmd,
            "payload_tail": payload_for_log,
            "partial_stdout_tail": partial_out,
            "partial_stderr_tail": partial_err,
            "runner_debug_dir": runner_debug_dir,
        })
        return 0, 0.0

    except Exception as ex:
        _log({
            "phase": "error",
            "ok": False,
            "kind": "bench_exception",
            "message": f"bench() exception: {repr(ex)}",
            "traceback": traceback.format_exc(),
            "cmd": cmd,
            "payload_tail": payload_for_log,
            "runner_debug_dir": runner_debug_dir,
        })
        return 0, 0.0

    if not res or not res.get("ok", False):
        return 0, 0.0
    if not res.get("correct", False):
        return 0, 0.0

    speedup = float(res.get("speedup", 0.0))
    return 1, speedup


# ============================================================
# FINAL: Rubric-only reward for Ablation
# ============================================================

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    ABLATION MODE:
    - Final reward is computed ONLY from rubric model output.
    - No bench() call is used in reward computation.

    Still feeds reference code into rubric judge for correctness/speedup inference.
    """
    if extra_info is None or "answer" not in extra_info:
        return 0.0

    reference_code = extra_info["answer"]

    # candidate code for judge
    try:
        candidate_code = _extract_python_code(solution_str)
    except Exception:
        candidate_code = solution_str

    if "class ModelNew" not in candidate_code and "class Model(" in candidate_code:
        candidate_code = candidate_code.replace("class Model(", "class ModelNew(", 1)

    rubric_obj = _run_rubric_judge(
        data_source=str(data_source),
        reference_code=reference_code,
        candidate_code=candidate_code,
        extra_info=extra_info,
    )

    reward, dbg = _compute_rubric_only_reward(
        rubric_obj=rubric_obj,
        data_source=str(data_source),
    )

    # Keep your historical cap behavior
    final = min(float(reward), 3.0)

    print(f"[rubric-only] obj={rubric_obj} dbg={dbg} final={final}")
    return final
