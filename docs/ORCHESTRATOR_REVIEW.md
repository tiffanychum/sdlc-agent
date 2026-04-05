# Orchestrator Maintainability Review

> **File reviewed:** `src/orchestrator.py`
> **Review date:** 2025
> **Issues identified:** 3

---

## Issue 1 — Monolithic God Function: `build_orchestrator_from_team`

### Description

`build_orchestrator_from_team` (lines 322–423) is a single ~100-line async function that
conflates at least five distinct responsibilities:

1. **Database access** — opens a session, queries `Team`, `Agent`, and `AgentToolMapping`,
   and builds `agents_config`.
2. **Tool resolution** — calls `get_all_tools()` and maps tool groups to per-agent lists.
3. **HITL wrapping** — iterates over every tool and conditionally wraps it with
   `wrap_dangerous_tool` / `wrap_reviewable_tool`.
4. **Agent construction** — assembles prompts, calls `get_llm()`, and invokes
   `create_react_agent` for every agent in the team.
5. **Graph dispatch** — selects the correct `_build_*_graph` builder and compiles the
   final LangGraph graph.

Because all of this lives in one function, a change to any single concern (e.g., switching
the ORM, adding a new HITL category, or changing how prompts are assembled) requires
reading and modifying the entire function. It also makes unit-testing any individual step
impossible without running the whole pipeline.

### Suggested Fix

Extract each responsibility into its own focused helper function and reduce
`build_orchestrator_from_team` to an orchestrating coordinator:

```python
# Step 1 – pure DB concern
async def _load_agents_config(team_id: str) -> tuple[str, list[AgentConfig]]:
    """Query DB and return (strategy, agents_config). No tool or LLM logic."""
    ...

# Step 2 – pure tool concern
def _resolve_agent_tools(
    ac: AgentConfig, tool_map: dict
) -> list[StructuredTool]:
    """Expand tool_groups → flat list of StructuredTool objects."""
    ...

# Step 3 – pure HITL concern
def _apply_hitl_wrappers(
    tools: list[StructuredTool], role: str
) -> list[StructuredTool]:
    """Wrap dangerous / reviewable tools and ensure ask_human is present."""
    ...

# Step 4 – pure agent-construction concern
def _build_agent(ac: AgentConfig, tools: list[StructuredTool]) -> CompiledGraph:
    """Assemble prompt, get LLM, and create the react agent."""
    ...

# Coordinator – thin glue only
async def build_orchestrator_from_team(
    team_id: str = "default",
    model_override=None,
    strategy_override: str = None,
):
    strategy, agents_config = await _load_agents_config(team_id)
    if strategy_override and strategy_override in VALID_STRATEGIES:
        strategy = strategy_override
    tool_map = await get_all_tools()
    built_agents = {}
    for ac in agents_config:
        tools = _apply_hitl_wrappers(_resolve_agent_tools(ac, tool_map), ac.role)
        built_agents[ac.role] = _build_agent(ac, tools)
    return _build_graph(strategy, agents_config, built_agents)
```

Each helper can now be unit-tested in isolation with straightforward mocks.

---

## Issue 2 — Prompt Templates Hardcoded Inside Closures and Builder Functions

### Description

Two large, multi-line prompt strings are defined inline inside function bodies rather than
as module-level constants or in an external template layer:

- **Router prompt** — built and returned by `_build_router_prompt()` (lines 230–280) as a
  plain f-string that embeds routing rules directly. Any change to routing logic requires
  editing deep inside a function.
- **Supervisor prompt** — constructed inside `_build_supervisor_graph()` (lines 525–530)
  via an inline f-string that captures `agent_descs` and `agent_names` from the enclosing
  scope. It is coupled to the graph-builder closure and cannot be read, tested, or iterated
  on without also building a graph.

Contrast this with `_META_ROUTER_PROMPT` (lines 40–67), which is correctly defined as a
module-level constant with `{placeholder}` substitution. The router and supervisor prompts
should follow the same pattern. The inconsistency makes it hard to find all prompts,
compare them, or apply a common formatting / versioning strategy.

### Suggested Fix

Define all prompt templates as module-level constants with named placeholders, mirroring
the existing `_META_ROUTER_PROMPT` pattern:

```python
# At module level, alongside _META_ROUTER_PROMPT

_ROUTER_PROMPT_TEMPLATE = """\
You are a routing agent. Select the SINGLE best-fit agent for the FIRST step of the task.
...
Available agents:
{agent_descs}

Routing rules (apply in strict priority order):
...
Respond with ONLY the agent name from: {agent_names}."""

_SUPERVISOR_PROMPT_TEMPLATE = """\
You are a supervisor agent. Decide which agent handles the task, or if it's complete.

Available agents:
{agent_descs}

Respond with ONLY "DONE" or an agent name ({agent_names})."""
```

Then the builder functions become simple one-liners:

```python
def _build_router_prompt(agents_config: list) -> str:
    return _ROUTER_PROMPT_TEMPLATE.format(
        agent_descs="\n".join(f'- "{a["role"]}": {a["description"]}' for a in agents_config),
        agent_names=", ".join(f'"{a["role"]}"' for a in agents_config),
    )
```

Benefits:
- All prompt text is findable at the top of the file.
- Prompts can be unit-tested without instantiating a graph.
- A future move to external `.jinja2` / `.txt` template files requires only changing the
  constant's assignment, not hunting through closures.

---

## Issue 3 — Untyped `dict` Used as the Agent Config Contract

### Description

Throughout the file, agent configuration is passed as an untyped `list[dict]` under the
name `agents_config`. Every consumer accesses fields via ad-hoc patterns with silent
fallbacks:

| Location | Pattern | Risk |
|---|---|---|
| `build_orchestrator_from_team` line 358 | `getattr(a, "model", "") or ""` | Silently hides a missing ORM column |
| `build_orchestrator_from_team` line 359 | `getattr(a, "decision_strategy", "react") or "react"` | Same |
| `_build_router_graph` line 431 | `agents_config[0]["role"]` | `KeyError` if `"role"` absent |
| `_build_sequential_graph` line 483 | `roles[i]` iterating list of strings | No traceability back to source dict |
| `_get_executor` / every builder | `ac.get("model") or None` | Conflates absent key with empty string |

Because the shape of `agents_config` is never formally declared, a typo in a key name
(e.g., `"desciption"` instead of `"description"`) silently produces empty strings or
`None` values that surface only at LLM call time. There is no IDE auto-complete, no static
analysis, and no centralised place to see what fields are required vs. optional.

### Suggested Fix

Introduce a lightweight `dataclass` (or `TypedDict`) to act as the contract between the
DB-loading step and the graph-building step:

```python
from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    id: str
    name: str
    role: str
    description: str
    system_prompt: str
    tool_groups: list[str] = field(default_factory=list)
    model: str = ""
    decision_strategy: str = "react"
```

Constructing it from the ORM row becomes an explicit, validated operation:

```python
AgentConfig(
    id=a.id,
    name=a.name,
    role=a.role,
    description=a.description or "",
    system_prompt=a.system_prompt or "",
    tool_groups=mapping_by_agent[a.id],
    model=getattr(a, "model", "") or "",
    decision_strategy=getattr(a, "decision_strategy", "react") or "react",
)
```

All downstream builder functions then receive `list[AgentConfig]` and access
`ac.role`, `ac.model`, etc. — getting IDE support, static type checking, and an
`AttributeError` (not silent `None`) if a field is accessed incorrectly.
The `getattr` fallback noise is confined to the one construction site instead of
being duplicated across every builder.

---

## Summary

| # | Issue | Affected Lines | Severity |
|---|---|---|---|
| 1 | Monolithic god function `build_orchestrator_from_team` | 322–423 | High |
| 2 | Prompt templates hardcoded inside closures / builder functions | 230–280, 525–530 | Medium |
| 3 | Untyped `dict` used as the agent config contract | Throughout | Medium |

Addressing these three issues in order will make the orchestrator significantly easier to
test, extend, and reason about without altering any runtime behaviour.
