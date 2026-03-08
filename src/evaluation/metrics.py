"""
Evaluation metrics for agent performance assessment.

Tracks:
- Task completion rate: did the agent successfully complete the requested task?
- Tool-call accuracy: did the agent call the right tools with correct arguments?
- Failure recovery: did the agent handle errors gracefully and retry?
- Latency: how long did the agent take to complete the task?
- Token usage: how many tokens were consumed?
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ToolCallMetric:
    tool_name: str
    arguments: dict
    expected_tool: str | None = None
    was_correct: bool | None = None
    error: str | None = None
    recovered: bool = False


@dataclass
class TaskMetric:
    task_id: str
    scenario_name: str
    prompt: str
    expected_agent: str
    actual_agent: str | None = None
    completed: bool = False
    tool_calls: list[ToolCallMetric] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    final_response: str = ""

    @property
    def routing_correct(self) -> bool:
        return self.actual_agent == self.expected_agent

    @property
    def latency_ms(self) -> float | None:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    @property
    def tool_call_accuracy(self) -> float:
        if not self.tool_calls:
            return 1.0 if not self.expected_tools else 0.0
        correct = sum(1 for tc in self.tool_calls if tc.was_correct)
        return correct / len(self.tool_calls)

    @property
    def failure_recovery_rate(self) -> float:
        failed = [tc for tc in self.tool_calls if tc.error]
        if not failed:
            return 1.0
        recovered = sum(1 for tc in failed if tc.recovered)
        return recovered / len(failed)


@dataclass
class EvalRunMetric:
    run_id: str
    model: str
    prompt_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tasks: list[TaskMetric] = field(default_factory=list)

    @property
    def task_completion_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.completed) / len(self.tasks)

    @property
    def routing_accuracy(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.routing_correct) / len(self.tasks)

    @property
    def avg_tool_call_accuracy(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.tool_call_accuracy for t in self.tasks) / len(self.tasks)

    @property
    def avg_failure_recovery_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.failure_recovery_rate for t in self.tasks) / len(self.tasks)

    @property
    def avg_latency_ms(self) -> float | None:
        latencies = [t.latency_ms for t in self.tasks if t.latency_ms]
        return sum(latencies) / len(latencies) if latencies else None

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp,
            "num_tasks": len(self.tasks),
            "task_completion_rate": round(self.task_completion_rate, 3),
            "routing_accuracy": round(self.routing_accuracy, 3),
            "avg_tool_call_accuracy": round(self.avg_tool_call_accuracy, 3),
            "avg_failure_recovery_rate": round(self.avg_failure_recovery_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1) if self.avg_latency_ms else None,
        }
