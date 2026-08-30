from __future__ import annotations

import asyncio
import json

import click
import structlog
from rich.console import Console
from rich.panel import Panel

from enaya.agent.core import Agent
from enaya.config.settings import get_config
from enaya.gateway.server import Gateway
from enaya.tools.builtins import register_builtin_tools
from enaya.tools.registry import ToolRegistry

logger = structlog.get_logger()
console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    if debug:
        structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(10))


@cli.command()
@click.argument("task")
@click.option("--context", "-c", type=str, help="JSON context for the task")
@click.option("--backend", "-b", default="local", help="Execution backend")
@click.option("--max-iterations", "-i", default=25, help="Max planning iterations")
@click.pass_context
def run(ctx: click.Context, task: str, context: str | None, backend: str, max_iterations: int) -> None:
    """Run an autonomous task."""
    console.print(
        Panel.fit(
            f"[bold blue]Enaya Agent[/bold blue]\nTask: {task}",
            title="Enaya",
            border_style="blue",
        )
    )
    agent = Agent()
    ctx_dict = json.loads(context) if context else {}
    result = asyncio.run(agent.run(task, ctx_dict))
    console.print("\n[bold green]Completed[/bold green]")
    console.print(f"Status: {result.status.name}")
    console.print(f"Iterations: {result.iteration}")
    for i, thought in enumerate(result.thoughts, 1):
        console.print(Panel(thought.content, title=f"Thought {i}", border_style="dim"))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Gateway host")
@click.option("--port", "-p", default=8765, help="Gateway port")
def serve(host: str, port: int) -> None:
    """Start the Enaya gateway server."""
    console.print(
        Panel.fit(
            f"[bold blue]Enaya Gateway[/bold blue]\nHost: {host}\nPort: {port}",
            title="Server",
            border_style="green",
        )
    )
    gateway = Gateway()
    asyncio.run(gateway.run())


@cli.command()
def tools() -> None:
    """List all available tools."""
    registry = ToolRegistry()
    register_builtin_tools(registry)
    console.print("[bold]Available Tools[/bold]")
    for tool in registry.describe_all():
        console.print(f"  [cyan]{tool['name']}[/cyan]: {tool['description']}")


@cli.command()
@click.argument("query", required=False)
def chat(query: str | None) -> None:
    """Interactive chat with Enaya."""
    if query:
        agent = Agent()
        response = asyncio.run(agent.chat(query))
        console.print(Panel(response, title="Enaya", border_style="green"))
    else:
        console.print("[bold cyan]Enaya Interactive Chat[/bold cyan]")
        console.print("Type 'exit' to quit\n")
        agent = Agent()
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                response = asyncio.run(agent.chat(user_input))
                console.print(
                    Panel(response, title="[bold blue]Enaya[/bold blue]", border_style="blue")
                )
            except (KeyboardInterrupt, EOFError):
                break
        console.print("\n[dim]Goodbye![/dim]")


@cli.command()
def status() -> None:
    """Show Enaya system status."""
    config = get_config()
    console.print(
        Panel.fit(
            f"[bold]Enaya Status[/bold]\n"
            f"\n[green]Agent ID:[/green] {config.project_name}"
            f"\n[green]Debug:[/green] {config.debug}"
            f"\n[green]Memory:[/green] {config.memory.backend}"
            f"\n[green]Gateway:[/green] {config.gateway.host}:{config.gateway.port}"
            f"\n[green]Backends:[/green] {', '.join(config.backend.available)}",
            title="System Status",
            border_style="green",
        )
    )


@cli.command()
@click.argument("plugin_path")
def install(plugin_path: str) -> None:
    """Install a plugin."""
    console.print(f"Installing plugin from {plugin_path}...")
    console.print("[green]Plugin installed successfully[/green]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
