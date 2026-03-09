"""
Automated evaluation pipeline for agent performance.

Evaluation approaches:
1. Rule-based metrics (task success, tool accuracy, step efficiency, safety)
2. LLM-as-Judge / G-Eval (correctness, relevance, coherence, tool quality)
3. Trajectory/Step evaluation (score each decision in the agent trace)
4. DeepEval integration (answer relevancy, faithfulness)
5. Langfuse integration (trace export for observability)
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
                 base_url: str = None, api_key: str = None,
                 use_llm_judge: bool = True, use_deepeval: bool = False):
        self.model = model or config.llm.model
        self.base_url = base_url or config.llm.base_url
        self.api_key = api_key or config.llm.api_key
        self.prompt_version = prompt_version
        self.results_dir = config.eval_output_dir
        self.use_llm_judge = use_llm_judge
        self.use_deepeval = use_deepeval

    async def run_evaluation(
        self,
        scenarios: list[EvalScenario] = None,
        team_id: str = "default",
        skip_init: bool = False,
    ) -> EvalRunMetric:
        from src.orchestrator import build_orchestrator_from_team

        if not skip_init:
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
        self._export_to_langfuse(run)
        return run

    async def run_comparison(
        self,
        model_configs: list[dict],
        scenarios: list[EvalScenario] = None,
        team_id: str = "default",
    ) -> list[EvalRunMetric]:
        results = []
        for mc in model_configs:
            evaluator = AgentEvaluator(
                model=mc.get("model", self.model),
                base_url=mc.get("base_url", self.base_url),
                api_key=mc.get("api_key", self.api_key),
                prompt_version=mc.get("prompt_version", self.prompt_version),
                use_llm_judge=self.use_llm_judge,
                use_deepeval=self.use_deepeval,
            )
            run = await evaluator.run_evaluation(scenarios=scenarios, team_id=team_id, skip_init=True)
            results.append(run)
        return results

    async def _run_scenario(self, orchestrator, scenario: EvalScenario) -> TaskMetric:
        task = TaskMetric(
            task_id=str(uuid.uuid4())[:8],
            scenario_name=scenario.name,
            prompt=scenario.prompt,
            expected_agent=scenario.expected_agent,
            expected_tools=scenario.expected_tools,
            expected_min_steps=scenario.expected_min_steps,
        )
        task.start_time = datetime.now()

        try:
            result = await orchestrator.ainvoke({
                "messages": [{"role": "user", "content": scenario.prompt}],
                "selected_agent": "", "agent_trace": [],
            })
            task.end_time = datetime.now()

            trace_steps = result.get("agent_trace", [])
            for entry in trace_steps:
                if entry.get("step") == "routing":
                    task.actual_agent = entry.get("selected_agent")
                elif entry.get("step") == "execution":
                    for tc in entry.get("tool_calls", []):
                        task.tool_calls.append(ToolCallMetric(
                            tool_name=tc["tool"],
                            arguments=tc.get("args", {}),
                            expected_tool=tc["tool"] if tc["tool"] in scenario.expected_tools else None,
                            was_correct=tc["tool"] in scenario.expected_tools,
                        ))

            for msg in result.get("messages", []):
                if hasattr(msg, "content") and hasattr(msg, "type") and msg.type == "tool":
                    task.tool_outputs.append(str(msg.content)[:500])

            last_msg = result["messages"][-1] if result.get("messages") else None
            if last_msg:
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                task.final_response = str(content) if not isinstance(content, str) else content

            task.completed = self._check_success(task.final_response, scenario)

            # ── LLM-as-Judge (G-Eval) ──
            if self.use_llm_judge and task.final_response:
                try:
                    from src.evaluation.llm_judge import judge_response, judge_trajectory

                    judge_scores = await judge_response(
                        user_prompt=scenario.prompt,
                        agent_response=task.final_response,
                        tool_calls=[{"tool": tc.tool_name, "args": tc.arguments} for tc in task.tool_calls],
                        tool_outputs=task.tool_outputs,
                    )
                    task.llm_judge_scores = judge_scores

                    trajectory_result = await judge_trajectory(
                        user_prompt=scenario.prompt,
                        trace_steps=trace_steps,
                        final_response=task.final_response,
                    )
                    task.trajectory_scores = trajectory_result

                except Exception:
                    task.llm_judge_scores = {}
                    task.trajectory_scores = {}

            # ── DeepEval Metrics ──
            if self.use_deepeval and task.final_response:
                try:
                    from src.evaluation.integrations import run_deepeval_metrics
                    task.deepeval_scores = run_deepeval_metrics(
                        user_prompt=scenario.prompt,
                        agent_response=task.final_response,
                        tool_outputs=task.tool_outputs,
                    )
                except Exception:
                    task.deepeval_scores = {}

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
        return any(kw.lower() in str(response).lower() for kw in scenario.success_keywords)

    def _save_results(self, run: EvalRunMetric) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        filepath = os.path.join(self.results_dir, f"eval_{run.run_id}_{run.timestamp[:10]}.json")
        data = {"summary": run.summary(), "tasks": [
            {
                "task_id": t.task_id, "scenario": t.scenario_name,
                "completed": t.completed, "routing_correct": t.routing_correct,
                "tool_call_accuracy": round(t.tool_call_accuracy, 3),
                "step_efficiency": round(t.step_efficiency, 3),
                "hallucination_score": round(t.hallucination_score, 3),
                "safety_score": round(t.safety_score, 3),
                "reasoning_quality": round(t.reasoning_quality, 3),
                "latency_ms": round(t.latency_ms, 1) if t.latency_ms else None,
                "llm_judge": getattr(t, "llm_judge_scores", {}),
                "trajectory": getattr(t, "trajectory_scores", {}),
                "deepeval": getattr(t, "deepeval_scores", {}),
                "error": t.error,
            }
            for t in run.tasks
        ]}
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _save_to_db(self, run: EvalRunMetric, team_id: str = None):
        try:
            from src.db.database import get_session
            from src.db.models import EvalRun
            session = get_session()
            summary = run.summary()
            eval_run = EvalRun(
                id=run.run_id, model=run.model, prompt_version=run.prompt_version,
                team_id=team_id, num_tasks=summary["num_tasks"],
                task_completion_rate=summary["task_success_rate"],
                routing_accuracy=summary["routing_accuracy"],
                avg_tool_call_accuracy=summary["tool_accuracy"],
                avg_failure_recovery_rate=summary["failure_recovery"],
                avg_latency_ms=summary.get("avg_latency_ms"),
                results_json=summary,
            )
            session.add(eval_run)
            session.commit()
            session.close()
        except Exception:
            pass

    def _export_to_langfuse(self, run: EvalRunMetric):
        try:
            from src.evaluation.integrations import export_eval_run_to_langfuse
            summary = run.summary()
            tasks = [{"scenario": t.scenario_name, "completed": t.completed} for t in run.tasks]
            export_eval_run_to_langfuse(summary, tasks)
        except Exception:
            pass

    @staticmethod
    def compare_runs(run_a: EvalRunMetric, run_b: EvalRunMetric) -> dict:
        a, b = run_a.summary(), run_b.summary()
        comparison = {}
        for key in ["task_success_rate", "tool_accuracy", "reasoning_quality", "step_efficiency",
                     "faithfulness", "safety_compliance", "routing_accuracy"]:
            val_a, val_b = a.get(key, 0), b.get(key, 0)
            delta = val_b - val_a
            comparison[key] = {"before": val_a, "after": val_b, "delta": round(delta, 3), "regression": delta < -0.05}
        return comparison
