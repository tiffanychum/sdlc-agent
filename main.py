"""
SDLC Agent — CLI entrypoint.

A general-purpose coding agent (similar to Manus / Claude Code) that delegates
tasks to specialized agents: Coder, Runner, and Researcher.

Usage:
    python main.py chat          Interactive chat with the multi-agent system
    python main.py eval          Run evaluation pipeline
    python main.py test-mcp      Quick MCP communication health check
"""

import asyncio
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.tree import Tree

console = Console()

AGENT_LABELS = {
    "coder": "Coder Agent (files + git)",
    "runner": "Runner Agent (shell + tests)",
    "researcher": "Researcher Agent (web)",
}


def _extract_response(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b if isinstance(b, str) else b.get("text", "") for b in content
        )
    return str(content)


def _render_trace(trace: list[dict], elapsed: float) -> Tree:
    """Render the agent execution trace as a Rich tree with step statuses."""
    tree = Tree("[bold]Agent Workflow[/bold]")
    step = 1

    for entry in trace:
        if entry.get("step") == "routing":
            agent_id = entry.get("selected_agent", "?")
            label = AGENT_LABELS.get(agent_id, agent_id)
            tree.add(f"[green]✓[/green] Step {step}: [bold]Route[/bold] → {label}")
            step += 1

        elif entry.get("step") == "execution":
            agent_id = entry.get("agent", "?")
            tool_calls = entry.get("tool_calls", [])

            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get("tool", "?")
                    args = tc.get("args", {})

                    args_preview = ", ".join(
                        f'{k}="{str(v)[:50]}"' for k, v in args.items()
                    )
                    node = tree.add(
                        f"[green]✓[/green] Step {step}: [bold]{tool_name}[/bold]({args_preview})"
                    )
                    step += 1
            else:
                tree.add(
                    f"[yellow]![/yellow] Step {step}: [bold]{agent_id}[/bold] responded (no tools called)"
                )
                step += 1

    tree.add(f"[blue]●[/blue] Done — {step - 1} step(s) in {elapsed:.1f}s")
    return tree


async def chat_mode():
    """Interactive chat with the multi-agent coding assistant."""
    from src.orchestrator import build_orchestrator

    console.print(Panel(
        "[bold]SDLC Agent — General-Purpose Coding Assistant[/bold]\n\n"
        "Agents: Coder (files + git), Runner (shell + tests), Researcher (web)\n"
        "Tools: Filesystem, Shell, Git, Web (all via MCP)\n\n"
        "Try:\n"
        '  "Read main.py and explain the architecture"\n'
        '  "List all Python files in src/mcp_servers"\n'
        '  "Run the tests"\n'
        '  "Search how to use LangGraph create_react_agent"\n\n'
        "Type 'quit' to exit.",
        title="Welcome",
    ))

    orchestrator = await build_orchestrator()

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        with console.status("[bold yellow]Working...[/bold yellow]", spinner="dots"):
            start_time = time.time()

            try:
                result = await orchestrator.ainvoke({
                    "messages": [{"role": "user", "content": user_input}],
                    "selected_agent": "",
                    "agent_trace": [],
                })
                elapsed = time.time() - start_time
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
                continue

        trace = result.get("agent_trace", [])
        tree = _render_trace(trace, elapsed)
        console.print()
        console.print(tree)

        last_msg = result["messages"][-1]
        response = _extract_response(
            last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        )

        agent_used = result.get("selected_agent", "unknown")
        console.print()
        console.print(f"[bold green]{agent_used}:[/bold green]")
        console.print(Markdown(response))


async def eval_mode():
    """Run the automated evaluation pipeline."""
    from src.evaluation.evaluator import AgentEvaluator
    from src.evaluation.reporter import print_eval_report

    console.print("[bold]Running evaluation pipeline...[/bold]\n")

    evaluator = AgentEvaluator()
    run = await evaluator.run_evaluation()
    print_eval_report(run)

    console.print(f"\n[dim]Results saved to {evaluator.results_dir}/[/dim]")


async def mcp_health_check():
    """Health check: verify all MCP servers can list their tools."""
    from src.mcp_servers.filesystem_server import list_tools as fs_tools
    from src.mcp_servers.shell_server import list_tools as shell_tools
    from src.mcp_servers.git_server import list_tools as git_tools
    from src.mcp_servers.web_server import list_tools as web_tools

    console.print("[bold]MCP Server Health Check[/bold]\n")

    checks = [
        ("Filesystem MCP Server", fs_tools),
        ("Shell MCP Server", shell_tools),
        ("Git MCP Server", git_tools),
        ("Web MCP Server", web_tools),
    ]

    all_healthy = True
    for name, list_fn in checks:
        try:
            tools = await list_fn()
            console.print(f"  [green]✓[/green] {name}: {len(tools)} tools registered")
            for tool in tools:
                console.print(f"    → {tool.name}: {tool.description[:60]}...")
        except Exception as e:
            console.print(f"  [red]✗[/red] {name}: {e}")
            all_healthy = False

    status = "[green]All servers healthy[/green]" if all_healthy else "[red]Some servers unhealthy[/red]"
    console.print(f"\n{status}")


def main():
    if len(sys.argv) < 2:
        console.print("Usage: python main.py [chat|eval|test-mcp]")
        sys.exit(1)

    command = sys.argv[1]

    match command:
        case "chat":
            asyncio.run(chat_mode())
        case "eval":
            asyncio.run(eval_mode())
        case "test-mcp":
            asyncio.run(mcp_health_check())
        case _:
            console.print(f"Unknown command: {command}")
            console.print("Usage: python main.py [chat|eval|test-mcp]")
            sys.exit(1)


if __name__ == "__main__":
    main()
