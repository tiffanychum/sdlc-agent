"""
Unit tests for orchestrator.py pure functions.

These tests have NO dependency on a database, LLM, or LangGraph runtime — they
cover the stateless helpers and routing logic that are easiest to regress.
"""

import pytest

# ---------------------------------------------------------------------------
# Import the module under test — works without any external services
# ---------------------------------------------------------------------------
from src.orchestrator import (
    _extract_text,
    _ensure_messages,
    _strategy_instruction,
    _build_router_prompt,
    STRATEGY_INSTRUCTIONS,
    MAX_SUPERVISOR_ITERATIONS,
    OrchestratorState,
    _take_last,
    _add_int,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ===========================================================================
# _extract_text
# ===========================================================================

class TestExtractText:
    def test_plain_string(self):
        assert _extract_text("hello") == "hello"

    def test_empty_string(self):
        assert _extract_text("") == ""

    def test_list_of_strings(self):
        assert _extract_text(["foo", "bar"]) == "foo\nbar"

    def test_list_of_dicts_with_text_key(self):
        blocks = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
        assert _extract_text(blocks) == "first\nsecond"

    def test_list_mixed_str_and_dict(self):
        blocks = ["alpha", {"text": "beta"}, {"no_text": "ignored"}]
        assert _extract_text(blocks) == "alpha\nbeta"

    def test_list_dict_without_text_key_is_skipped(self):
        blocks = [{"type": "image", "url": "http://example.com"}]
        assert _extract_text(blocks) == ""

    def test_non_string_non_list_falls_back_to_str(self):
        assert _extract_text(42) == "42"
        assert _extract_text(None) == "None"
        assert _extract_text({"key": "val"}) == "{'key': 'val'}"

    def test_empty_list(self):
        assert _extract_text([]) == ""

    def test_multiblock_preserves_newline_separation(self):
        result = _extract_text(["line one", "line two", "line three"])
        assert "line one" in result
        assert "line two" in result
        assert "\n" in result


# ===========================================================================
# _ensure_messages
# ===========================================================================

class TestEnsureMessages:
    def test_passthrough_existing_base_messages(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        result = _ensure_messages(msgs)
        assert result == msgs

    def test_dict_user_role(self):
        result = _ensure_messages([{"role": "user", "content": "question"}])
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "question"

    def test_dict_assistant_role(self):
        result = _ensure_messages([{"role": "assistant", "content": "answer"}])
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "answer"

    def test_dict_system_role(self):
        result = _ensure_messages([{"role": "system", "content": "instructions"}])
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "instructions"

    def test_dict_unknown_role_falls_back_to_human(self):
        result = _ensure_messages([{"role": "unknown_role", "content": "text"}])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "text"

    def test_dict_missing_role_defaults_to_user(self):
        result = _ensure_messages([{"content": "no role key"}])
        assert isinstance(result[0], HumanMessage)

    def test_dict_missing_content_defaults_to_empty(self):
        result = _ensure_messages([{"role": "user"}])
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == ""

    def test_non_dict_non_message_coerced_to_human(self):
        result = _ensure_messages(["raw string", 42])
        assert all(isinstance(m, HumanMessage) for m in result)
        assert result[0].content == "raw string"
        assert result[1].content == "42"

    def test_empty_list(self):
        assert _ensure_messages([]) == []

    def test_mixed_types(self):
        msgs = [
            HumanMessage(content="real"),
            {"role": "assistant", "content": "dict"},
            "plain string",
        ]
        result = _ensure_messages(msgs)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)


# ===========================================================================
# _strategy_instruction
# ===========================================================================

class TestStrategyInstruction:
    def test_known_strategies_return_non_empty(self):
        for strategy in ("react", "plan_execute", "reflexion", "cot"):
            result = _strategy_instruction(strategy)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_strategy_falls_back_to_react(self):
        result = _strategy_instruction("nonexistent_strategy")
        assert result == STRATEGY_INSTRUCTIONS["react"]

    def test_empty_string_falls_back_to_react(self):
        assert _strategy_instruction("") == STRATEGY_INSTRUCTIONS["react"]

    def test_all_strategies_are_distinct(self):
        texts = [STRATEGY_INSTRUCTIONS[s] for s in STRATEGY_INSTRUCTIONS]
        assert len(texts) == len(set(texts)), "Each strategy should have a unique instruction"


# ===========================================================================
# _build_router_prompt
# ===========================================================================

class TestBuildRouterPrompt:
    def _make_agents(self, roles_descs):
        return [{"role": r, "description": d} for r, d in roles_descs]

    def test_contains_agent_roles(self):
        agents = self._make_agents([("coder", "writes code"), ("runner", "runs commands")])
        prompt = _build_router_prompt(agents)
        assert '"coder"' in prompt
        assert '"runner"' in prompt

    def test_contains_agent_descriptions(self):
        agents = self._make_agents([("coder", "writes code")])
        prompt = _build_router_prompt(agents)
        assert "writes code" in prompt

    def test_no_duplicate_rule_numbers(self):
        import re
        agents = self._make_agents([("coder", "desc"), ("runner", "desc")])
        prompt = _build_router_prompt(agents)
        numbers = re.findall(r"^\d+\.", prompt, re.MULTILINE)
        assert len(numbers) == len(set(numbers)), f"Duplicate rule numbers found: {numbers}"

    def test_single_agent_still_valid(self):
        agents = self._make_agents([("coder", "sole agent")])
        prompt = _build_router_prompt(agents)
        assert "coder" in prompt

    def test_respond_instruction_includes_all_roles(self):
        agents = self._make_agents([("coder", "x"), ("researcher", "y"), ("runner", "z")])
        prompt = _build_router_prompt(agents)
        last_line = [l for l in prompt.splitlines() if "Respond with ONLY" in l]
        assert last_line, "Missing 'Respond with ONLY' instruction"
        assert "coder" in last_line[0]
        assert "researcher" in last_line[0]
        assert "runner" in last_line[0]


# ===========================================================================
# Supervisor routing helper (pure function, no LLM needed)
# ===========================================================================

class TestSupervisorRouter:
    """Test the supervisor_router closure extracted from _build_supervisor_graph."""

    def _make_router(self, valid_roles):
        """Replicate the supervisor_router logic from _build_supervisor_graph."""
        def supervisor_router(state) -> str:
            sel = state.get("selected_agent", "__done__")
            if sel == "__done__" or sel not in valid_roles:
                return "__end__"
            return sel
        return supervisor_router

    def test_done_routes_to_end(self):
        router = self._make_router({"coder", "runner"})
        assert router({"selected_agent": "__done__"}) == "__end__"

    def test_valid_role_routes_to_that_role(self):
        router = self._make_router({"coder", "runner"})
        assert router({"selected_agent": "coder"}) == "coder"
        assert router({"selected_agent": "runner"}) == "runner"

    def test_unknown_role_routes_to_end(self):
        router = self._make_router({"coder"})
        assert router({"selected_agent": "hallucinated_agent"}) == "__end__"

    def test_missing_key_defaults_to_end(self):
        router = self._make_router({"coder"})
        assert router({}) == "__end__"


# ===========================================================================
# Reducer functions
# ===========================================================================

class TestReducers:
    def test_take_last_returns_second_value(self):
        assert _take_last("first", "second") == "second"
        assert _take_last("", "new") == "new"
        assert _take_last("a", "") == ""

    def test_add_int_accumulates(self):
        assert _add_int(0, 1) == 1
        assert _add_int(3, 1) == 4
        assert _add_int(9, 1) == 10


# ===========================================================================
# OrchestratorState schema
# ===========================================================================

class TestOrchestratorState:
    def test_has_required_keys(self):
        annotations = OrchestratorState.__annotations__
        for key in ("messages", "selected_agent", "agent_trace", "supervisor_iterations"):
            assert key in annotations, f"Missing key: {key}"

    def test_supervisor_iterations_present(self):
        assert "supervisor_iterations" in OrchestratorState.__annotations__

    def test_max_supervisor_iterations_positive(self):
        assert MAX_SUPERVISOR_ITERATIONS > 0
