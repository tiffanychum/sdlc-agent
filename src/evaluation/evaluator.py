"""
Automated evaluation pipeline for agent performance assessment.

Supports:
- Single-model evaluation runs
- Multi-LLM comparison (run same scenarios with different models)
- Regression detection between runs
- Persisting results to DB and JSON
"""

import json
import os
import uuid
from datetime import datetime

from src.evaluation.metrics import EvalRunMetric, TaskMetric, ToolCallMetric
from src.evaluation.scenarios import EvalScenario, SCENARIOS
from src.config import config


class AgentEvaluator:

    def __init__(self, model: str = None, prompt_version: str = "v1",
                 base_url: str = None, api_key: str = None):
        self.model = model or config.llm.model
        self.base_url = base_url or config.llm.base_url
        self.api_key = api_key or config.llm.api_key
        self.prompt_version = prompt_version
        self.results_dir = config.eval_output_dir

    async def run_evaluation(
        self,
        scenarios: list[EvalScenario] = None,
        team_id: str = "default",
    ) -> EvalRunMetric:
        """Execute all scenarios and return aggregated metrics."""
        from src.orchestrator import build_orchestrator_from_team
        from src.db.database import init_db, seed_defaults

        init_db()
        seed_defaults()

        scenarios = scenarios or SCENARIOS
        orchestrator = await build_orchestrator_from_team(team_id)

        run = EvalRunMetric(
            run_id=str(uuid.uuid4())[:8],
            model=self.model,
            prompt_version=self.prompt_version,
        )

        for scenario in scenarios:
            task_metric = await self._run_scenario(orchestrator, scenario)
            run.tasks.append(task_metric)

        self._save_results(run)
        self._save_to_db(run, team_id)
        return run

    async def run_comparison(
        self,
        model_configs: list[dict],
        scenarios: list[EvalScenario] = None,
        team_id: str = "default",
    ) -> list[EvalRunMetric]:
        """Run the same scenarios with multiple LLM configs and return all results."""
        results = []
        for mc in model_configs:
            evaluator = AgentEvaluator(
                model=mc.get("model", self.model),
                base_url=mc.get("base_url", self.base_url),
                api_key=mc.get("api_key", self.api_key),
                prompt_version=mc.get("prompt_version", self.prompt_version),
            )
            run = await evaluator.run_evaluation(scenarios=scenarios, team_id=team_id)
            results.append(run)
        return results

    async def _run_scenario(self, orchestrator, scenario: EvalScenario) -> TaskMetric:
        task = TaskMetric(
            task_id=str(uuid.uuid4())[:8],
            scenario_name=scenario.name,
            prompt=scenario.prompt,
            expected_agent=scenario.expected_agent,
            expected_tools=scenario.expected_tools,
        )
        task.start_time = datetime.now()

        try:
            result = await orchestrator.ainvoke({
                "messages": [{"role": "user", "content": scenario.prompt}],
                "selected_agent": "",
                "agent_trace": [],
            })

            task.end_time = datetime.now()

            if trace := result.get("agent_trace", []):
                for entry in trace:
                    if entry.get("step") == "routing":
                        task.actual_agent = entry.get("selected_agent")
                    elif entry.get("step") == "execution":
                        for tc in entry.get("tool_calls", []):
                            tool_metric = ToolCallMetric(
                                tool_name=tc["tool"],
                                arguments=tc.get("args", {}),
                                expected_tool=tc["tool"] if tc["tool"] in scenario.expected_tools else None,
                                was_correct=tc["tool"] in scenario.expected_tools,
                            )
                            task.tool_calls.append(tool_metric)

            last_msg = result["messages"][-1] if result.get("messages") else None
            if last_msg:
                task.final_response = (
                    last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                )

            task.completed = self._check_success(task.final_response, scenario)

        except Exception as e:
            task.end_time = datetime.now()
            task.error = str(e)
            task.completed = False

        return task

    def _check_success(self, response: str, scenario: EvalScenario) -> bool:
        if not scenario.success_keywords:
            return True
        if isinstance(response, list):
            response = " ".join(str(r) for r in response)
        response_lower = str(response).lower()
        return any(kw.lower() in response_lower for kw in scenario.success_keywords)

    def _save_results(self, run: EvalRunMetric) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, f"eval_{run.run_id}_{run.timestamp[:10]}.json")

        data = {
            "summary": run.summary(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "scenario": t.scenario_name,
                    "prompt": t.prompt,
                    "expected_agent": t.expected_agent,
                    "actual_agent": t.actual_agent,
                    "routing_correct": t.routing_correct,
                    "completed": t.completed,
                    "tool_call_accuracy": round(t.tool_call_accuracy, 3),
                    "failure_recovery_rate": round(t.failure_recovery_rate, 3),
                    "latency_ms": round(t.latency_ms, 1) if t.latency_ms else None,
                    "tool_calls": [
                        {"tool": tc.tool_name, "correct": tc.was_correct,
                         "error": tc.error, "recovered": tc.recovered}
                        for tc in t.tool_calls
                    ],
                    "error": t.error,
                }
                for t in run.tasks
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _save_to_db(self, run: EvalRunMetric, team_id: str = None):
        """Persist evaluation run summary to the database."""
        try:
            from src.db.database import get_session
            from src.db.models import EvalRun

            session = get_session()
            summary = run.summary()
            eval_run = EvalRun(
                id=run.run_id,
                model=run.model,
                prompt_version=run.prompt_version,
                team_id=team_id,
                num_tasks=summary["num_tasks"],
                task_completion_rate=summary["task_completion_rate"],
                routing_accuracy=summary["routing_accuracy"],
                avg_tool_call_accuracy=summary["avg_tool_call_accuracy"],
                avg_failure_recovery_rate=summary["avg_failure_recovery_rate"],
                avg_latency_ms=summary.get("avg_latency_ms"),
                results_json=summary,
            )
            session.add(eval_run)
            session.commit()
            session.close()
        except Exception:
            pass

    @staticmethod
    def compare_runs(run_a: EvalRunMetric, run_b: EvalRunMetric) -> dict:
        a, b = run_a.summary(), run_b.summary()
        comparison = {}
        for key in ["task_completion_rate", "routing_accuracy", "avg_tool_call_accuracy", "avg_failure_recovery_rate"]:
            val_a, val_b = a.get(key, 0), b.get(key, 0)
            delta = val_b - val_a
            comparison[key] = {
                "before": val_a, "after": val_b,
                "delta": round(delta, 3), "regression": delta < -0.05,
            }
        return comparison
