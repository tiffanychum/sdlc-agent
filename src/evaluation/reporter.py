"""
Evaluation report generator.

Produces human-readable reports from evaluation runs,
highlighting regressions and areas for improvement.
"""

from rich.console import Console
from rich.table import Table

from src.evaluation.metrics import EvalRunMetric


console = Console()


def print_eval_report(run: EvalRunMetric) -> None:
    """Print a formatted evaluation report to the console."""
    console.print(f"\n[bold]Evaluation Report — Run {run.run_id}[/bold]")
    console.print(f"Model: {run.model} | Prompt Version: {run.prompt_version}")
    console.print(f"Timestamp: {run.timestamp}\n")

    summary = run.summary()
    summary_table = Table(title="Summary Metrics")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Tasks Run", str(summary["num_tasks"]))
    summary_table.add_row("Task Completion Rate", f"{summary['task_completion_rate']:.1%}")
    summary_table.add_row("Routing Accuracy", f"{summary['routing_accuracy']:.1%}")
    summary_table.add_row("Avg Tool-Call Accuracy", f"{summary['avg_tool_call_accuracy']:.1%}")
    summary_table.add_row("Avg Failure Recovery", f"{summary['avg_failure_recovery_rate']:.1%}")
    if summary["avg_latency_ms"]:
        summary_table.add_row("Avg Latency", f"{summary['avg_latency_ms']:.0f}ms")

    console.print(summary_table)

    detail_table = Table(title="\nPer-Scenario Results")
    detail_table.add_column("Scenario", style="cyan")
    detail_table.add_column("Routing", style="green")
    detail_table.add_column("Completed", style="green")
    detail_table.add_column("Tool Accuracy", style="yellow")
    detail_table.add_column("Latency", style="blue")

    for task in run.tasks:
        routing_ok = "✓" if task.routing_correct else f"✗ (got: {task.actual_agent})"
        completed = "✓" if task.completed else "✗"
        tool_acc = f"{task.tool_call_accuracy:.0%}" if task.tool_calls else "N/A"
        latency = f"{task.latency_ms:.0f}ms" if task.latency_ms else "N/A"

        detail_table.add_row(task.scenario_name, routing_ok, completed, tool_acc, latency)

    console.print(detail_table)


def print_regression_report(comparison: dict) -> None:
    """Print a regression comparison between two evaluation runs."""
    console.print("\n[bold]Regression Analysis[/bold]\n")

    table = Table(title="Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Before", style="blue")
    table.add_column("After", style="blue")
    table.add_column("Delta", style="green")
    table.add_column("Status", style="bold")

    for metric, data in comparison.items():
        status = "[red]⚠ REGRESSION[/red]" if data["regression"] else "[green]✓ OK[/green]"
        delta_str = f"{data['delta']:+.1%}"
        table.add_row(
            metric,
            f"{data['before']:.1%}",
            f"{data['after']:.1%}",
            delta_str,
            status,
        )

    console.print(table)
