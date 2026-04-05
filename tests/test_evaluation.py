"""
Unit tests for the evaluation pipeline.

Validates metric calculations, regression detection,
and scenario execution without requiring LLM calls.
"""

import pytest
from datetime import datetime, timedelta

from src.evaluation.metrics import (
    EvalRunMetric,
    TaskMetric,
    ToolCallMetric,
)
from src.evaluation.evaluator import AgentEvaluator
from src.evaluation.scenarios import SCENARIOS


class TestTaskMetrics:
    """Unit tests for individual task metric calculations."""

    def test_routing_correct(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba", actual_agent="ba",
        )
        assert task.routing_correct is True

    def test_routing_incorrect(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba", actual_agent="developer",
        )
        assert task.routing_correct is False

    def test_latency_calculation(self):
        now = datetime.now()
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba",
            start_time=now, end_time=now + timedelta(milliseconds=500),
        )
        assert abs(task.latency_ms - 500.0) < 1.0

    def test_tool_call_accuracy_all_correct(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba",
            expected_tools=["jira_search", "jira_create_ticket"],
            tool_calls=[
                ToolCallMetric(tool_name="jira_search", arguments={}, was_correct=True),
                ToolCallMetric(tool_name="jira_create_ticket", arguments={}, was_correct=True),
            ],
        )
        assert task.tool_call_accuracy == 1.0

    def test_tool_call_accuracy_partial(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba",
            expected_tools=["jira_search"],
            tool_calls=[
                ToolCallMetric(tool_name="jira_search", arguments={}, was_correct=True),
                ToolCallMetric(tool_name="kb_search", arguments={}, was_correct=False),
            ],
        )
        assert task.tool_call_accuracy == 0.5

    def test_tool_call_accuracy_no_calls_no_expected(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba", expected_tools=[], tool_calls=[],
        )
        assert task.tool_call_accuracy == 1.0

    def test_failure_recovery_rate(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba",
            tool_calls=[
                ToolCallMetric(tool_name="jira_get_ticket", arguments={},
                               error="Not found", recovered=True),
                ToolCallMetric(tool_name="jira_search", arguments={},
                               error="Timeout", recovered=False),
            ],
        )
        assert task.failure_recovery_rate == 0.5

    def test_failure_recovery_no_failures(self):
        task = TaskMetric(
            task_id="t1", scenario_name="test", prompt="test",
            expected_agent="ba",
            tool_calls=[
                ToolCallMetric(tool_name="jira_search", arguments={}, was_correct=True),
            ],
        )
        assert task.failure_recovery_rate == 1.0


class TestEvalRunMetrics:
    """Unit tests for aggregated evaluation run metrics."""

    def _make_run(self, tasks: list[TaskMetric]) -> EvalRunMetric:
        return EvalRunMetric(
            run_id="test-run", model="test-model",
            prompt_version="v1", tasks=tasks,
        )

    def test_task_completion_rate(self):
        tasks = [
            TaskMetric(task_id="1", scenario_name="s1", prompt="p1",
                       expected_agent="ba", completed=True),
            TaskMetric(task_id="2", scenario_name="s2", prompt="p2",
                       expected_agent="qa", completed=False),
            TaskMetric(task_id="3", scenario_name="s3", prompt="p3",
                       expected_agent="ba", completed=True),
        ]
        run = self._make_run(tasks)
        assert abs(run.task_completion_rate - 0.667) < 0.01

    def test_routing_accuracy(self):
        tasks = [
            TaskMetric(task_id="1", scenario_name="s1", prompt="p1",
                       expected_agent="ba", actual_agent="ba"),
            TaskMetric(task_id="2", scenario_name="s2", prompt="p2",
                       expected_agent="qa", actual_agent="developer"),
        ]
        run = self._make_run(tasks)
        assert run.routing_accuracy == 0.5

    def test_summary_structure(self):
        tasks = [
            TaskMetric(task_id="1", scenario_name="s1", prompt="p1",
                       expected_agent="ba", actual_agent="ba", completed=True),
        ]
        run = self._make_run(tasks)
        summary = run.summary()
        expected_keys = {
            "run_id", "model", "prompt_version", "timestamp",
            "num_tasks", "task_success_rate", "tool_accuracy",
            "reasoning_quality", "step_efficiency", "faithfulness",
            "routing_accuracy", "failure_recovery", "safety_compliance",
            "avg_latency_ms",
        }
        assert set(summary.keys()) == expected_keys


class TestRegressionDetection:
    """Tests for detecting performance regressions between eval runs."""

    def test_detects_regression(self):
        run_a = EvalRunMetric(
            run_id="a", model="m", prompt_version="v1",
            tasks=[TaskMetric(
                task_id="1", scenario_name="s", prompt="p",
                expected_agent="ba", actual_agent="ba", completed=True,
            )],
        )
        run_b = EvalRunMetric(
            run_id="b", model="m", prompt_version="v2",
            tasks=[TaskMetric(
                task_id="1", scenario_name="s", prompt="p",
                expected_agent="ba", actual_agent="developer", completed=False,
            )],
        )
        comparison = AgentEvaluator.compare_runs(run_a, run_b)
        assert comparison["task_success_rate"]["regression"] is True
        assert comparison["routing_accuracy"]["regression"] is True

    def test_no_regression(self):
        tasks = [TaskMetric(
            task_id="1", scenario_name="s", prompt="p",
            expected_agent="ba", actual_agent="ba", completed=True,
        )]
        run_a = EvalRunMetric(run_id="a", model="m", prompt_version="v1", tasks=tasks)
        run_b = EvalRunMetric(run_id="b", model="m", prompt_version="v2", tasks=tasks)
        comparison = AgentEvaluator.compare_runs(run_a, run_b)
        assert all(not v["regression"] for v in comparison.values())


class TestEvalScenarios:
    """Validate scenario definitions are well-formed."""

    def test_all_scenarios_have_required_fields(self):
        for s in SCENARIOS:
            assert s.name, "Scenario missing name"
            assert s.prompt, "Scenario missing prompt"
            valid_agents = {
                "coder", "tester", "devops", "researcher",
                "reviewer", "planner", "project_manager", "business_analyst",
            }
            assert s.expected_agent in valid_agents, \
                f"Invalid expected_agent: {s.expected_agent} (must be one of {sorted(valid_agents)})"

    def test_scenario_names_unique(self):
        names = [s.name for s in SCENARIOS]
        assert len(names) == len(set(names)), "Duplicate scenario names found"

    def test_minimum_scenario_count(self):
        assert len(SCENARIOS) >= 10, "Need at least 10 eval scenarios for meaningful coverage"
