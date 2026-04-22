"""
Adaptive pairwise A/B judge — the "4th score family" beyond DeepEval / trace
assertions / keyword matching.

Design (see prior design doc in conversation):

1. Judge is given BOTH outputs + all DeepEval scores + golden metadata in one
   prompt and asked to:
   a) derive task-relevant QUALITY DIMENSIONS from the prompt + reference_output
      (flagging applicable_here=false for dimensions that don't fit the task —
      e.g. "test coverage" for read-only tasks)
   b) score both sides on each applicable dimension with QUOTED evidence
   c) produce a DeepEval "gap analysis" — for every sub-metric, say whether it
      pointed the right way, wrong way, or was silent on the real issue
   d) declare winner only with delta >= 0.10 AND confidence >= 0.6

2. temperature=0 + strict JSON schema → reproducible. Cached by the calling
   endpoint on (run_a, run_b, golden_id, judge_model).

3. Uses `get_rubric_judge_llm()` which defaults to `claude-opus-4.7` via Poe.
   Does NOT change the existing G-Eval / DeepEval judge model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import get_rubric_judge_llm
from src.orchestrator import _extract_text

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_DELAY_S = 1.5

_SYSTEM_PROMPT = """You are a senior principal engineer evaluating two agent outputs for a software
engineering task. You MUST be ruthlessly grounded: never invent evidence, never
reward verbosity, never score a dimension without an exact quote from the actual
output.

Your job has four stages. Follow them in order.

STAGE 1 — DERIVE TASK-RELEVANT DIMENSIONS
Read the TASK and REFERENCE OUTPUT. List the quality dimensions that actually
matter for THIS specific task. Flag each with `applicable_here=true/false`. A
dimension like "test coverage" is applicable_here=false for read-only tasks.
Keep to 4-8 dimensions; omit dimensions that do not affect correctness.

STAGE 2 — SCORE BOTH SIDES PER DIMENSION
For every applicable dimension, produce:
  side_a.score in [0,1], side_a.evidence (an EXACT quote or factual observation
                                          from side A's output — no paraphrase)
  side_b.score in [0,1], side_b.evidence (same rules for side B)
  delta — e.g. "+0.40 A" or "0.00 tie"
If evidence genuinely cannot be found in the output, set score=null and say so.

STAGE 3 — DEEPEVAL GAP ANALYSIS (MOST IMPORTANT)
For each DeepEval metric provided, explain what the metric says vs what you
actually see:
  - pointed="right"  → metric aligns with your per-dimension scoring
  - pointed="wrong"  → metric points the opposite direction of quality reality
  - pointed="silent" → metric captures nothing the user cares about here
Provide a corrective_signal telling the user when to trust/discount this metric
in future runs of similar tasks.

STAGE 4 — VERDICT
Compute rubric_score per side = average of all non-null applicable dimension
scores (unweighted for now; all dimensions weighted equally). Declare:
  winner = "A" | "B" | "tie"
  A winner requires: |rubric_score_a - rubric_score_b| >= 0.10 AND confidence >= 0.6.
  Otherwise "tie".

Output STRICTLY valid JSON matching the schema in the user message. No preamble.
No trailing prose. No markdown code fences. JSON only.
"""


_SCHEMA_REMINDER = """
Return this exact shape:

{
  "task_dimensions": [
    {"dim": "<slug>", "why_it_matters": "<one sentence>", "applicable_here": true}
  ],
  "per_dimension": {
    "<slug>": {
      "side_a": {"score": <0-1 or null>, "evidence": "<exact quote or observation>"},
      "side_b": {"score": <0-1 or null>, "evidence": "<exact quote or observation>"},
      "delta":  "<e.g. '+0.40 A' or '0.00 tie'>"
    }
  },
  "rubric_score": {"side_a": <0-1>, "side_b": <0-1>, "delta": "<signed>"},
  "deepeval_gap_analysis": [
    {
      "metric": "<deepeval metric name>",
      "deepeval_verdict": "A=<x> B=<y>",
      "actual_story": "<what you see>",
      "pointed": "right" | "wrong" | "silent",
      "corrective_signal": "<when to trust or discount this metric>"
    }
  ],
  "winner": "A" | "B" | "tie",
  "confidence": <0-1>,
  "verdict": "<2-4 sentence summary grounded in the scores above>",
  "key_defects_in_loser": ["<specific defect 1>", "<specific defect 2>"]
}
"""


# ── Input data shapes ───────────────────────────────────────────────

@dataclass
class SideInput:
    label: str                         # "A" or "B"
    team_id: str
    run_id: str
    model: str
    prompt_versions: dict              # role -> version
    actual_output: str
    actual_agent: str
    delegation_pattern: list
    tools_called: list
    llm_calls: int
    tool_calls: int
    latency_ms: float
    cost: float
    tokens_in: int
    tokens_out: int
    deepeval_scores: dict              # from RegressionResult.deepeval_scores (with _reason keys)


@dataclass
class GoldenInput:
    id: str
    name: str
    prompt: str
    reference_output: str
    expected_tools: list = field(default_factory=list)
    expected_delegation: list = field(default_factory=list)
    expected_keywords: list = field(default_factory=list)
    quality_thresholds: dict = field(default_factory=dict)
    complexity: str = ""


# ── Prompt rendering ────────────────────────────────────────────────

def _format_deepeval_block(de: dict) -> str:
    """Render the deepeval_scores JSON into a compact, judge-readable block."""
    if not de:
        return "  (no DeepEval scores recorded)"
    # Split score vs _reason keys
    metrics: dict[str, dict] = {}
    for k, v in de.items():
        if k.endswith("_reason"):
            metrics.setdefault(k[: -len("_reason")], {})["reason"] = v
        else:
            metrics.setdefault(k, {})["score"] = v
    lines = []
    for name, entry in metrics.items():
        s = entry.get("score")
        r = (entry.get("reason") or "").strip()
        if isinstance(s, (int, float)):
            score_str = f"{s:.3f}"
        elif s is None:
            score_str = "null (judge call failed)"
        else:
            score_str = str(s)
        line = f"  - {name}: {score_str}"
        if r:
            r_short = r.replace("\n", " ").strip()
            if len(r_short) > 240:
                r_short = r_short[:237] + "..."
            line += f"   // reason: {r_short}"
        lines.append(line)
    return "\n".join(lines)


def _format_side(s: SideInput) -> str:
    pv = ", ".join(f"{k}:{v}" for k, v in (s.prompt_versions or {}).items()) or "(unknown)"
    delegation = " -> ".join(s.delegation_pattern or [])
    tools = ", ".join(s.tools_called or [])
    return f"""## SIDE {s.label} — team={s.team_id} | run={s.run_id}
Model: {s.model}
Prompt versions: {pv}
Trajectory: {delegation or '(no agents recorded)'}
Tools called ({len(s.tools_called or [])}): {tools or '(none)'}
LLM calls: {s.llm_calls}   Tool calls: {s.tool_calls}
Latency: {s.latency_ms/1000:.1f}s   Cost: ${s.cost:.4f}   Tokens in/out: {s.tokens_in} / {s.tokens_out}

DeepEval metrics (with reasons):
{_format_deepeval_block(s.deepeval_scores)}

ACTUAL_OUTPUT ({len(s.actual_output)} chars):
\"\"\"
{s.actual_output}
\"\"\"
"""


def _render_user_prompt(g: GoldenInput, a: SideInput, b: SideInput) -> str:
    return f"""# TASK
{g.prompt}

# REFERENCE OUTPUT (intent grounding — NOT a grading key)
\"\"\"
{g.reference_output or '(no reference output stored for this golden)'}
\"\"\"

# GOLDEN METADATA
- golden_id: {g.id}
- name: {g.name}
- complexity: {g.complexity or 'unknown'}
- expected_tools: {g.expected_tools}
- expected_delegation: {g.expected_delegation}
- expected_keywords: {g.expected_keywords}
- quality_thresholds: {g.quality_thresholds}

{_format_side(a)}
{_format_side(b)}

# RETURN SCHEMA (strict — JSON only, no prose before or after)
{_SCHEMA_REMINDER}
"""


# ── JSON extraction + schema validation ─────────────────────────────

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Extract the first top-level JSON object from the LLM response."""
    # Strip markdown fences if any
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```\s*$", "", txt)
    match = _JSON_BLOCK.search(txt)
    if not match:
        raise ValueError(f"No JSON object found in judge response (first 300 chars): {raw[:300]}")
    return json.loads(match.group(0))


_REQUIRED_TOP_KEYS = (
    "task_dimensions", "per_dimension", "rubric_score",
    "deepeval_gap_analysis", "winner", "confidence", "verdict", "key_defects_in_loser",
)


def _validate_payload(payload: dict) -> None:
    missing = [k for k in _REQUIRED_TOP_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Judge payload missing required keys: {missing}")

    if payload["winner"] not in ("A", "B", "tie"):
        raise ValueError(f"Invalid winner value: {payload['winner']!r}")

    rs = payload["rubric_score"]
    for side in ("side_a", "side_b"):
        v = rs.get(side)
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError(f"rubric_score.{side} must be number, got {type(v).__name__}")

    if not isinstance(payload["task_dimensions"], list):
        raise ValueError("task_dimensions must be a list")
    if not isinstance(payload["per_dimension"], dict):
        raise ValueError("per_dimension must be a dict")
    if not isinstance(payload["deepeval_gap_analysis"], list):
        raise ValueError("deepeval_gap_analysis must be a list")


async def _retry_async(coro_fn):
    last_exc: Optional[BaseException] = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                logger.warning(
                    "Rubric-judge call failed (attempt %d/%d): %s: %s — retrying in %.1fs",
                    attempt + 1, _MAX_ATTEMPTS, type(exc).__name__, exc,
                    _RETRY_DELAY_S * (attempt + 1),
                )
                await asyncio.sleep(_RETRY_DELAY_S * (attempt + 1))
    assert last_exc is not None
    raise last_exc


# ── Public entry point ─────────────────────────────────────────────

async def judge_ab(golden: GoldenInput, a: SideInput, b: SideInput) -> dict:
    """
    One-shot pairwise rubric judgment. Returns the validated JSON payload.

    Caller is responsible for caching by (run_a, run_b, golden_id, judge_model).
    Raises on unrecoverable parse/schema failure AFTER retries.
    """
    user_prompt = _render_user_prompt(golden, a, b)
    llm = get_rubric_judge_llm(temperature=0.0)

    async def _call():
        resp = await llm.ainvoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        raw = _extract_text(resp.content)
        payload = _extract_json(raw)
        _validate_payload(payload)
        return payload

    payload = await _retry_async(_call)
    logger.info(
        "rubric-judge: golden=%s winner=%s conf=%.2f rubric(A=%.2f B=%.2f)",
        golden.id,
        payload.get("winner"),
        payload.get("confidence", 0.0),
        payload.get("rubric_score", {}).get("side_a") or 0.0,
        payload.get("rubric_score", {}).get("side_b") or 0.0,
    )
    return payload


# ── Convenience: build inputs from ORM rows ─────────────────────────

def build_side_input(label: str, reg_result, team_id: str) -> SideInput:
    """Convert a RegressionResult ORM row into a SideInput."""
    # reconstruct prompt_versions from the row (stored as a scalar string or dict
    # depending on how the run was recorded; normalise to dict for display)
    pv_raw = getattr(reg_result, "prompt_version", None) or "v1"
    pv: dict = {"__overall": pv_raw} if isinstance(pv_raw, str) else (pv_raw or {})
    router_pv = getattr(reg_result, "router_prompt_version", None)
    if router_pv:
        pv["router"] = router_pv

    return SideInput(
        label=label,
        team_id=team_id,
        run_id=reg_result.run_id,
        model=reg_result.model_used or "unknown",
        prompt_versions=pv,
        actual_output=reg_result.actual_output or "",
        actual_agent=reg_result.actual_agent or "",
        delegation_pattern=reg_result.actual_delegation_pattern or [],
        tools_called=reg_result.actual_tools or [],
        llm_calls=reg_result.actual_llm_calls or 0,
        tool_calls=reg_result.actual_tool_calls or 0,
        latency_ms=reg_result.actual_latency_ms or 0.0,
        cost=reg_result.actual_cost or 0.0,
        tokens_in=reg_result.actual_tokens_in or 0,
        tokens_out=reg_result.actual_tokens_out or 0,
        deepeval_scores=reg_result.deepeval_scores or {},
    )


def build_golden_input(golden) -> GoldenInput:
    """Convert a GoldenTestCase ORM row into a GoldenInput."""
    return GoldenInput(
        id=golden.id,
        name=golden.name or golden.id,
        prompt=golden.prompt or "",
        reference_output=golden.reference_output or "",
        expected_tools=list(golden.expected_tools or []),
        expected_delegation=list(golden.expected_delegation_pattern or []),
        expected_keywords=list(golden.expected_output_keywords or []),
        quality_thresholds=dict(golden.quality_thresholds or {}),
        complexity=golden.complexity or "",
    )
