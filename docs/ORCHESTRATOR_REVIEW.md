# Orchestrator Maintainability Review

**File:** `src/orchestrator.py`  
**Lines:** 811  
**Reviewed:** 2024  
**Verdict:** needs-work

---

## Issue 1: Monolithic File Structure

### Description
The `orchestrator.py` file is 811 lines long and handles multiple distinct responsibilities:
- Strategy selection (meta-router logic, lines 75-162)
- Graph building for 4 different strategies (router, sequential, parallel, supervisor)
- Agent execution and handoff coordination
- Database queries and ORM operations (lines 368-414)
- Prompt templates and routing logic
- State management and synthesis

This violates the Single Responsibility Principle and makes the file difficult to navigate, test, and maintain. Changes to one strategy risk breaking others, and the cognitive load for understanding the entire orchestration system is high.

**Affected Lines:** Entire file (1-811)

**Severity:** WARNING

### Suggested Fix
Refactor into a modular structure:

```
src/orchestrator/
├── __init__.py
├── state.py              # OrchestratorState, reducers
├── strategies/
│   ├── __init__.py
│   ├── router.py         # _build_router_graph
│   ├── sequential.py     # _build_sequential_graph
│   ├── parallel.py       # _build_parallel_graph
│   └── supervisor.py     # _build_supervisor_graph
├── synthesis.py          # _synthesize_outputs (unified)
├── meta_router.py        # select_strategy_auto, _heuristic_strategy
├── prompts.py            # All prompt templates
└── builder.py            # build_orchestrator_from_team (DB logic)
```

**Benefits:**
- Each file has a single, clear responsibility
- Easier to test individual strategies in isolation
- Reduces merge conflicts in team environments
- Improves code discoverability

---

## Issue 2: Duplicated Synthesis Logic

### Description
Three separate functions implement nearly identical synthesis logic:
1. `_synthesize_outputs()` (lines 515-571) — used by sequential and supervisor
2. `_merge()` inside `_build_parallel_graph()` (lines 611-646)
3. Wrapper functions `_seq_synthesize()` and `_sup_synthesize()` (lines 587-588, 771-772)

All three:
- Extract the original user request from messages
- Collect AI agent outputs
- Call an LLM with a synthesis prompt
- Return a unified response

The only differences are minor prompt wording variations ("sequential" vs "parallel" vs "supervisor"). This violates the DRY (Don't Repeat Yourself) principle and creates maintenance burden: bug fixes or improvements must be applied in three places.

**Affected Lines:** 515-571, 587-588, 611-646, 771-772

**Severity:** WARNING

### Suggested Fix
Create a single, parameterized synthesis function:

```python
async def synthesize_agent_outputs(
    state: OrchestratorState,
    mode: str = "multi-agent",
    custom_instructions: str = ""
) -> OrchestratorState:
    """Unified synthesis for all orchestration strategies.
    
    Args:
        state: Current orchestrator state with messages.
        mode: Strategy label for logging ("sequential", "parallel", "supervisor").
        custom_instructions: Optional extra guidance for the synthesis LLM.
    """
    msgs = _ensure_messages(state["messages"])
    
    user_request = next(
        (m.content for m in msgs if isinstance(m, HumanMessage)), None
    )
    if not user_request:
        logger.debug("%s synthesizer: no user request found, skipping.", mode)
        return {}
    
    agent_outputs = [
        m for m in msgs
        if isinstance(m, AIMessage) and len(_extract_text(m.content).strip()) > 50
    ]
    if len(agent_outputs) <= 1:
        return {}
    
    base_instructions = (
        f"You are a synthesizer for a {mode} multi-agent workflow. "
        "Multiple agents have completed their parts of a task. "
        "Combine their outputs into ONE final, coherent response that directly "
        "and completely answers the user's original request. "
        "Preserve all important details (file paths, code, results, URLs). "
        "Remove redundant preamble and agent self-introductions. "
        "Structure the response clearly with headers if it spans multiple topics."
    )
    
    synthesis_system = base_instructions + ("\n\n" + custom_instructions if custom_instructions else "")
    
    try:
        llm = get_llm()
        response = await llm.ainvoke([
            SystemMessage(content=synthesis_system),
            *msgs,
            HumanMessage(content=(
                f"Original request: {user_request}\n\n"
                "Produce the single unified final answer now."
            )),
        ])
        logger.info("%s synthesizer: produced unified response (%d chars).",
                    mode, len(_extract_text(response.content)))
        return {"messages": [response]}
    except Exception as exc:
        logger.warning("%s synthesizer LLM call failed (%s); returning raw outputs.", mode, exc)
        return {}
```

Then replace all three implementations with calls to this function:

```python
# In _build_sequential_graph:
async def _seq_synthesize(state: OrchestratorState) -> OrchestratorState:
    return await synthesize_agent_outputs(state, mode="sequential")

# In _build_parallel_graph:
async def _merge(state: OrchestratorState) -> OrchestratorState:
    return await synthesize_agent_outputs(
        state, 
        mode="parallel",
        custom_instructions="If agents produced conflicting information, note the discrepancy."
    )

# In _build_supervisor_graph:
async def _sup_synthesize(state: OrchestratorState) -> OrchestratorState:
    return await synthesize_agent_outputs(state, mode="supervisor")
```

**Benefits:**
- Single source of truth for synthesis logic
- Bug fixes apply everywhere automatically
- Easier to add new synthesis features (e.g., structured output, citations)
- Reduces code size by ~80 lines

---

## Issue 3: Hardcoded Prompts Scattered Throughout Code

### Description
Large, multi-line prompt templates are embedded directly in the code at multiple locations:
- `_META_ROUTER_PROMPT` (lines 41-72) — 32 lines
- `STRATEGY_INSTRUCTIONS` dict (lines 200-212) — 13 lines
- `_build_router_prompt()` function (lines 235-285) — 51 lines
- `supervisor_prompt` in `_build_supervisor_graph()` (lines 680-701) — 22 lines

**Problems:**
1. **Prompt engineering is difficult** — Iterating on prompts requires navigating Python code, not dedicated prompt files
2. **Version control noise** — Prompt tweaks create large diffs mixed with code changes
3. **No prompt reusability** — Can't easily A/B test prompts or load them from external sources
4. **Localization impossible** — No path to multi-language support
5. **Hard to review** — Prompt quality reviews require reading Python source

**Affected Lines:** 41-72, 200-212, 235-285, 680-701

**Severity:** INFO

### Suggested Fix
Extract all prompts to a dedicated prompts module or configuration files:

**Option A: Python module (immediate, low-risk)**

Create `src/orchestrator/prompts.py`:

```python
"""Prompt templates for orchestrator strategies."""

META_ROUTER_PROMPT = """\
You are an orchestration meta-router. Given a user task and available agents, choose the best \
multi-agent execution strategy.

Available agents:
{agent_descs}

Strategy options and when to use each:
- "router_decides" — Route to exactly ONE agent. Use when the task is a single, self-contained
  action (read a file, run a search, check git status). One agent can do it all alone.
...
"""

STRATEGY_INSTRUCTIONS = {
    "react": """## Decision Strategy: ReAct (Reason + Act)
Think step by step: 1) Reason about what to do next 2) Take ONE action using a tool 3) Observe the result 4) Repeat until done. Always act, never just describe.""",
    # ... etc
}

def build_router_prompt(agents_config: list[dict]) -> str:
    """Generate the router prompt from agent configuration."""
    agent_descs = "\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config)
    agent_names = ", ".join(f'"{a["role"]}"' for a in agents_config)
    return f"""You are a routing agent. Select the SINGLE best-fit agent for the FIRST step of the task.
...
Respond with ONLY the agent name from: {agent_names}."""

def build_supervisor_prompt(agents_config: list[dict]) -> str:
    """Generate the supervisor prompt from agent configuration."""
    # ... implementation
```

Then in `orchestrator.py`:

```python
from src.orchestrator.prompts import (
    META_ROUTER_PROMPT,
    STRATEGY_INSTRUCTIONS,
    build_router_prompt,
    build_supervisor_prompt,
)
```

**Option B: External YAML/JSON files (better for non-engineers)**

Create `prompts/orchestrator.yaml`:

```yaml
meta_router:
  system: |
    You are an orchestration meta-router. Given a user task and available agents, 
    choose the best multi-agent execution strategy.
    
    Available agents:
    {agent_descs}
    
    Strategy options and when to use each:
    ...

strategies:
  react: |
    ## Decision Strategy: ReAct (Reason + Act)
    Think step by step: 1) Reason about what to do next...
  
  plan_execute: |
    ## Decision Strategy: Plan-and-Execute
    ALWAYS create a plan first...

router:
  system: |
    You are a routing agent. Select the SINGLE best-fit agent...
```

Load with:

```python
import yaml
from pathlib import Path

_PROMPTS = yaml.safe_load(Path("prompts/orchestrator.yaml").read_text())

def get_prompt(key: str, **kwargs) -> str:
    """Retrieve and format a prompt template."""
    template = _PROMPTS
    for part in key.split("."):
        template = template[part]
    return template.format(**kwargs)
```

**Benefits:**
- Prompts are first-class artifacts, easy to review and version
- Non-engineers (product managers, prompt engineers) can iterate on prompts
- A/B testing and experimentation become trivial
- Cleaner git diffs (prompt changes separate from code changes)
- Enables future features: prompt versioning, user-customizable prompts, localization

**Recommended approach:** Start with Option A (Python module) for immediate improvement, then migrate to Option B (YAML) if prompt iteration velocity becomes a bottleneck.

---

## Summary

These three issues share a common theme: **separation of concerns**. The orchestrator currently mixes:
- Business logic (strategy selection, graph building)
- Data access (database queries)
- Presentation (prompt templates)
- Infrastructure (synthesis, state management)

Addressing these issues will:
1. Reduce cognitive load for developers
2. Enable parallel development (multiple engineers can work on different strategies)
3. Improve testability (unit test individual strategies, mock synthesis)
4. Accelerate prompt engineering (non-engineers can contribute)
5. Reduce bug surface area (changes are localized)

**Recommended priority:**
1. **Issue 2 (Duplicated synthesis)** — Quick win, immediate code reduction, low risk
2. **Issue 3 (Hardcoded prompts)** — Medium effort, high long-term value
3. **Issue 1 (Monolithic structure)** — Larger refactor, plan carefully, do incrementally

All three can be addressed incrementally without breaking existing functionality by using the Strangler Fig pattern: create new modules alongside the existing code, migrate one function at a time, then remove the old code once all references are updated.
