# Orchestrator Maintainability Review

> **File reviewed:** `src/orchestrator.py`  
> **Review scope:** Top 3 maintainability issues  
> **Lines of code:** 610

---

## Issue 1 — God Function: `build_orchestrator_from_team` Does Too Much

### Description

`build_orchestrator_from_team` (lines 322–423) is a single ~100-line async function that
is responsible for at least **five distinct concerns**:

1. Opening and querying the database (Teams, Agents, AgentToolMappings)
2. Resolving the orchestration strategy
3. Iterating over agents to assemble tool lists per agent
4. Applying HITL wrappers (`wrap_dangerous_tool`, `wrap_reviewable_tool`, `ask_human`)
5. Building LLM instances and `create_react_agent` objects
6. Dispatching to one of four graph-builder functions

Because all of this lives in one function, any single change — e.g. adding a new HITL
wrapper type, changing how tools are loaded, or altering strategy resolution — requires
reading and reasoning about the entire function. It is also effectively untestable in
isolation: you cannot unit-test tool assembly without also triggering a real database
query, and you cannot test DB loading without also triggering LLM construction.

**Specific symptoms:**
- The function has at least 3 levels of nested loops (`for ac in agents_config` → `for group
  in ac["tool_groups"]` → `for t in agent_tools`).
- The `try/except/finally` DB block and the tool-assembly loop are at the same indentation
  level with no visual or logical boundary between them.
- The `model_override` mutation loop (lines 378–380) is a side-effectful pass that alters
  shared config dicts in-place.

### Suggested Fix

Decompose the function into focused, single-responsibility helpers:

```python
# 1. Pure DB concern — returns plain dicts, no LLM objects
async def _load_team_config(team_id: str) -> tuple[str, list[dict]]:
    """Return (strategy, agents_config) from the database."""
    ...

# 2. Pure tool concern — returns a list of (possibly-wrapped) StructuredTools
def _build_agent_tools(
    ac: dict,
    tool_map: dict,
) -> list[StructuredTool]:
    """Assemble and HITL-wrap tools for a single agent config dict."""
    ...

# 3. Pure LLM / agent concern
def _build_react_agent(ac: dict, tools: list, model_override=None):
    """Construct the LLM and create_react_agent for one agent."""
    ...

# 4. Thin orchestrator — composes the above
async def build_orchestrator_from_team(
    team_id: str = "default",
    model_override=None,
    strategy_override: str = None,
):
    strategy, agents_config = await _load_team_config(team_id)
    if strategy_override and strategy_override in VALID_STRATEGIES:
        strategy = strategy_override
    tool_map = await get_all_tools()
    built_agents, exec_agents = {}, {}
    for ac in agents_config:
        tools = _build_agent_tools(ac, tool_map)
        built_agents[ac["role"]] = _build_react_agent(ac, tools, model_override)
        ...
    return builders[strategy](agents_config, built_agents, ...)
```

This structure means each helper can be unit-tested independently with mocks, and future
changes (e.g. a new wrapping strategy) are localised to one small function.

---

## Issue 2 — Prompt Strings Embedded as Raw f-strings Inside Logic Functions

### Description

Two large natural-language prompt templates are constructed inline inside logic functions:

- **Router prompt** — `_build_router_prompt` (lines 230–280): a 50-line f-string that
  mixes routing rules, agent descriptions, and formatting directly in a Python function.
- **Supervisor prompt** — inside `_build_supervisor_graph` (lines 525–530): a multi-line
  f-string constructed at graph-build time inside a graph-builder function.

This pattern creates several problems:

1. **Untestable in isolation.** The prompt content cannot be verified without calling the
   full builder function. There is no way to assert "the router prompt contains rule 3"
   without constructing a full `agents_config` fixture.
2. **No versioning or swapping.** Changing from one prompt style to another (e.g. for a
   different LLM provider that needs a different format) requires editing the middle of a
   logic function rather than swapping a template.
3. **Mixed abstraction levels.** `_build_router_prompt` is named as if it returns a string,
   but it also encodes business rules (Jira routing, file access priority, etc.) that belong
   in a configuration layer, not in a Python string literal.
4. **The supervisor prompt** is not even extracted to a named constant or builder — it is
   an anonymous f-string on lines 525–530, making it invisible to a reader scanning the
   module's top-level symbols.

### Suggested Fix

Extract all prompt templates into a dedicated module (e.g. `src/prompts/orchestrator.py`)
as named template strings, and use `.format()` or a lightweight templating approach:

```python
# src/prompts/orchestrator.py

ROUTER_PROMPT_TEMPLATE = """\
You are a routing agent. Select the SINGLE best-fit agent ...

Available agents:
{agent_descs}

Routing rules:
...

Respond with ONLY the agent name from: {agent_names}.\
"""

SUPERVISOR_PROMPT_TEMPLATE = """\
You are a supervisor agent. Decide which agent handles the task, or if it's complete.

Available agents:
{agent_descs}

Respond with ONLY "DONE" or an agent name ({agent_names}).\
"""
```

Then in `orchestrator.py`:

```python
from src.prompts.orchestrator import ROUTER_PROMPT_TEMPLATE, SUPERVISOR_PROMPT_TEMPLATE

def _build_router_prompt(agents_config: list[dict]) -> str:
    return ROUTER_PROMPT_TEMPLATE.format(
        agent_descs=...,
        agent_names=...,
    )
```

Benefits: prompts become independently testable, can be loaded from files or a database,
and changes to routing rules no longer touch the graph-building logic.

---

## Issue 3 — Silent Exception Swallowing in the Database Block

### Description

The database query block in `build_orchestrator_from_team` (lines 369–373) uses:

```python
except Exception:
    session.rollback()
    raise
```

While `raise` does re-propagate the exception, this pattern has three maintainability
problems:

1. **No log entry at the point of failure.** When this path is hit in production, the
   only signal is whatever the caller (potentially a LangGraph node or an HTTP handler)
   decides to log — which may be far removed from the DB context. There is no log line
   that says *"DB query for team X failed with error Y"*, making it hard to distinguish
   a misconfigured team ID from a transient connection error from a schema mismatch.

2. **Bare `except Exception` is too broad.** It catches everything from
   `sqlalchemy.exc.OperationalError` (transient, retriable) to `ValueError` (programmer
   error, not retriable) to `KeyboardInterrupt`-adjacent exceptions. These have very
   different recovery strategies but are handled identically.

3. **Inconsistency with the rest of the file.** `select_strategy_auto` (lines 108–116)
   logs every retry attempt with structured fields (`type(exc).__name__`, `exc`). The DB
   block provides none of that structure, making log correlation harder.

**Contrast with the meta-router's error handling** (lines 108–116), which logs:
```
Meta-router attempt 1/3 failed (OpenAIError: rate limit); retrying…
```
versus the DB block, which logs: *(nothing)*.

### Suggested Fix

Add a structured `logger.exception` call before re-raising, and narrow the caught
exception types where possible:

```python
from sqlalchemy.exc import SQLAlchemyError

try:
    team = session.query(Team).filter_by(id=team_id).first()
    if not team:
        raise ValueError(f"Team '{team_id}' not found")
    ...
except SQLAlchemyError as exc:
    logger.exception(
        "Database error while loading team %r: %s", team_id, exc
    )
    session.rollback()
    raise
except ValueError:
    # Configuration errors (missing team, no agents) — log at WARNING, not ERROR
    logger.warning("Configuration error for team %r", team_id, exc_info=True)
    raise
finally:
    session.close()
```

This ensures:
- Every DB failure produces a structured log entry with `team_id` and the full traceback.
- `SQLAlchemyError` and `ValueError` are handled at appropriate severity levels.
- The `finally: session.close()` is preserved unchanged.

---

## Summary

| # | Issue | Location | Severity |
|---|-------|----------|----------|
| 1 | God function — `build_orchestrator_from_team` mixes DB, tool assembly, HITL wrapping, LLM construction, and graph dispatch | `lines 322–423` | **High** — blocks unit testing and increases change risk |
| 2 | Prompt templates embedded as raw f-strings inside logic functions | `lines 230–280`, `525–530` | **Medium** — prompts are untestable and unversioned |
| 3 | Bare `except Exception` with no logging in the DB block | `lines 369–373` | **Medium** — makes production debugging significantly harder |
