"""Command Line Interface for llm-ledger.

Provides commands:
- start:   Launch transparent HTTP proxy server
- stats:   Display usage, token, latency, and cost statistics
- tail:    Follow requests in real time
- export:  Export usage logs to JSON or CSV
- models:  Browse model pricing and supported providers
- config:  Inspect active database and configuration paths
- reset:   Clear tracked records
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console

from llm_ledger import __version__
from llm_ledger.db import (
    clear_records,
    export_records,
    get_default_db_path,
    get_recent_requests,
    get_requests_since_id,
    get_stats,
    resolve_db_path,
)
from llm_ledger.pricing import (
    DEFAULT_PRICING_PATH,
    get_all_pricing,
    load_custom_pricing,
)
from llm_ledger.proxy import create_proxy_app
from llm_ledger.ui import (
    build_tail_table,
    render_models_table,
    render_start_banner,
    render_stats_dashboard,
    render_tail_single,
)

app = typer.Typer(
    name="llm-ledger",
    help="Lightweight, local-first proxy and CLI tracking real-time LLM token usage, latency, and exact USD cost.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold green]llm-ledger[/bold green] version [cyan]{__version__}[/cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show llm-ledger version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """llm-ledger: Track real-time token usage, latency, and cost across LLM providers."""
    pass


@app.command("start")
def start_proxy(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host interface to bind the proxy server to."),
    port: int = typer.Option(8787, "--port", "-P", help="Port to listen on."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite database file."),
    default_project: str = typer.Option("default", "--default-project", "-p", help="Default project name for requests."),
    pricing_file: Optional[Path] = typer.Option(None, "--pricing-file", help="Path to custom JSON pricing file."),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level (critical, error, warning, info, debug)."),
) -> None:
    """Start the transparent LLM proxy server."""
    resolved_db = resolve_db_path(db)
    custom_pricing = load_custom_pricing(pricing_file)

    render_start_banner(
        host=host,
        port=port,
        db_path=str(resolved_db),
        default_project=default_project,
    )

    proxy_app = create_proxy_app(
        db_path=resolved_db,
        default_project=default_project,
        custom_pricing=custom_pricing,
    )

    uvicorn.run(
        proxy_app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=False,
    )


@app.command("stats")
def show_stats(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter statistics by project name."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Filter by provider (openai, anthropic, gemini, openrouter)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model substring."),
    days: Optional[int] = typer.Option(None, "--days", "-d", help="Number of past days to include (e.g. 7 or 30)."),
    since: Optional[str] = typer.Option(None, "--since", help="ISO start timestamp (e.g. 2026-01-01)."),
    until: Optional[str] = typer.Option(None, "--until", help="ISO end timestamp."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite database file."),
) -> None:
    """Display usage, token, latency, and cost statistics."""
    resolved_db = resolve_db_path(db)
    stats = get_stats(
        project=project,
        provider=provider,
        model=model,
        days=days,
        since=since,
        until=until,
        db_path=resolved_db,
    )

    filters = []
    if project:
        filters.append(f"project: {project}")
    if provider:
        filters.append(f"provider: {provider}")
    if model:
        filters.append(f"model: {model}")
    if days:
        filters.append(f"last {days} days")
    filter_label = ", ".join(filters)

    render_stats_dashboard(stats, filter_label=filter_label)


@app.command("tail")
def tail_requests(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter live stream by project."),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent requests to show initially."),
    follow: bool = typer.Option(True, "--follow/--no-follow", "-f", help="Continuously poll and follow new requests."),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Polling interval in seconds."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite database file."),
) -> None:
    """Follow LLM requests in real time."""
    resolved_db = resolve_db_path(db)
    recent = get_recent_requests(limit=limit, project=project, db_path=resolved_db)
    # Reverse so oldest in initial batch is printed first
    recent_chronological = list(reversed(recent))

    if recent_chronological:
        console.print(build_tail_table(recent_chronological, title="Recent Requests"))
    else:
        console.print("[dim]No previous requests recorded.[/dim]")

    if not follow:
        return

    last_id = max((r["id"] for r in recent), default=0)
    console.print(f"[dim]Tailing new requests (polling every {interval}s). Press Ctrl+C to stop...[/dim]\n")

    try:
        while True:
            time.sleep(interval)
            new_records = get_requests_since_id(
                last_id=last_id,
                project=project,
                limit=50,
                db_path=resolved_db,
            )
            for r in new_records:
                render_tail_single(r)
                if r["id"] > last_id:
                    last_id = r["id"]
    except KeyboardInterrupt:
        console.print("\n[dim]Tailing stopped.[/dim]")


@app.command("export")
def export_data(
    format: str = typer.Option("json", "--format", "-f", help="Export format: json or csv."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File path to save the exported data (defaults to stdout)."),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project."),
    since: Optional[str] = typer.Option(None, "--since", help="Filter start timestamp."),
    until: Optional[str] = typer.Option(None, "--until", help="Filter end timestamp."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite database file."),
) -> None:
    """Export tracked request metadata to JSON or CSV."""
    resolved_db = resolve_db_path(db)
    result = export_records(
        format=format,
        project=project,
        since=since,
        until=until,
        db_path=resolved_db,
    )

    if output:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result, encoding="utf-8")
        console.print(f"[green]Successfully exported records to[/green] [bold]{output_path}[/bold]")
    else:
        sys.stdout.write(result + "\n")


@app.command("models")
def list_models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter models by provider (openai, anthropic, gemini, openrouter)."),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search model by name or ID."),
    pricing_file: Optional[Path] = typer.Option(None, "--pricing-file", help="Path to custom JSON pricing file."),
) -> None:
    """List supported models and their input/output token pricing per 1M tokens."""
    custom = load_custom_pricing(pricing_file)
    pricing_catalog = get_all_pricing(custom)
    render_models_table(
        pricing_catalog,
        provider_filter=provider,
        search_query=search,
    )


@app.command("config")
def show_config(
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite database file."),
) -> None:
    """Show active configuration, database path, and pricing file location."""
    resolved_db = resolve_db_path(db)
    console.print(f"[bold white]Database Location:[/bold white]   {resolved_db}")
    console.print(f"[bold white]Default Pricing File:[/bold white] {DEFAULT_PRICING_PATH}")
    console.print(f"[bold white]Version:[/bold white]              {__version__}")


@app.command("reset")
def reset_database(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Clear only records for this project."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    db: Optional[Path] = typer.Option(None, "--db", help="Path to SQLite database file."),
) -> None:
    """Clear recorded requests from the database."""
    resolved_db = resolve_db_path(db)
    target = f"records for project '{project}'" if project else "ALL records in the database"

    if not yes:
        confirm = typer.confirm(f"Are you sure you want to permanently delete {target}?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Abort()

    deleted = clear_records(project=project, db_path=resolved_db)
    console.print(f"[bold green]Deleted {deleted} {target}.[/bold green]")


if __name__ == "__main__":
    app()
