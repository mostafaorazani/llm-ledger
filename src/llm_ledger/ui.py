"""Rich UI terminal components for llm-ledger.

Provides panels, formatted tables, statistics dashboards, and live tail feeds.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_ledger import __version__

console = Console()


def format_currency(amount: float) -> str:
    """Format USD currency cleanly, showing precision for fractional cents."""
    if amount == 0.0:
        return "$0.00"
    if amount < 0.001:
        return f"${amount:.6f}"
    elif amount < 0.01:
        return f"${amount:.4f}"
    elif amount < 1.0:
        return f"${amount:.4f}"
    else:
        return f"${amount:,.2f}"


def format_tokens(count: int) -> str:
    """Format integer token count with commas."""
    return f"{count:,}"


def format_latency(latency_ms: float) -> str:
    """Format latency in ms or seconds."""
    if latency_ms < 1000:
        return f"{latency_ms:.0f} ms"
    return f"{latency_ms / 1000:.2f} s"


def format_status_code(code: int) -> Text:
    """Return colored status code text."""
    if 200 <= code < 300:
        return Text(str(code), style="bold green")
    elif 400 <= code < 500:
        return Text(str(code), style="bold yellow")
    else:
        return Text(str(code), style="bold red")


def format_provider(provider: str) -> Text:
    """Return colored provider label."""
    prov = provider.lower()
    styles = {
        "openai": "bold green",
        "anthropic": "bold magenta",
        "gemini": "bold cyan",
        "openrouter": "bold yellow",
    }
    return Text(provider, style=styles.get(prov, "bold white"))


def render_stats_dashboard(stats: dict[str, Any], filter_label: str = "") -> None:
    """Display complete statistics dashboard using Rich panels and tables."""
    total_reqs = stats.get("total_requests", 0)
    total_cost = stats.get("total_cost_usd", 0.0)
    total_tokens = stats.get("total_tokens", 0)
    in_tokens = stats.get("total_input_tokens", 0)
    out_tokens = stats.get("total_output_tokens", 0)
    avg_latency = stats.get("avg_latency_ms", 0.0)
    p50_latency = stats.get("p50_latency_ms", 0.0)
    p90_latency = stats.get("p90_latency_ms", 0.0)

    if total_reqs == 0:
        empty_msg = Text("No tracked requests found matching the criteria.", style="italic yellow")
        console.print(Panel(empty_msg, title="llm-ledger Stats", border_style="dim"))
        return

    # Overview Cards Table
    overview_table = Table.grid(expand=True, padding=(0, 2))
    overview_table.add_column(justify="center", ratio=1)
    overview_table.add_column(justify="center", ratio=1)
    overview_table.add_column(justify="center", ratio=1)
    overview_table.add_column(justify="center", ratio=1)

    card_cost = Panel(
        Text(format_currency(total_cost), style="bold green", justify="center"),
        title="[dim]Total Cost[/dim]",
        border_style="green",
    )
    card_reqs = Panel(
        Text(f"{total_reqs:,}", style="bold cyan", justify="center"),
        title="[dim]Total Requests[/dim]",
        border_style="cyan",
    )
    card_tokens = Panel(
        Text(
            f"{format_tokens(total_tokens)}\n[dim]({format_tokens(in_tokens)} in / {format_tokens(out_tokens)} out)[/dim]",
            style="bold yellow",
            justify="center",
        ),
        title="[dim]Tokens[/dim]",
        border_style="yellow",
    )
    card_latency = Panel(
        Text(
            f"{format_latency(avg_latency)}\n[dim]p50: {format_latency(p50_latency)} | p90: {format_latency(p90_latency)}[/dim]",
            style="bold magenta",
            justify="center",
        ),
        title="[dim]Avg Latency[/dim]",
        border_style="magenta",
    )

    overview_table.add_row(card_cost, card_reqs, card_tokens, card_latency)

    title = "LLM Ledger - Usage & Cost Statistics"
    if filter_label:
        title += f" ({filter_label})"

    console.print(Panel(overview_table, title=f"[bold]{title}[/bold]", border_style="blue"))

    # Project Breakdown
    by_project = stats.get("by_project", [])
    if by_project:
        proj_table = Table(title="Usage by Project", show_header=True, header_style="bold cyan")
        proj_table.add_column("Project", style="bold")
        proj_table.add_column("Requests", justify="right")
        proj_table.add_column("Input Tokens", justify="right")
        proj_table.add_column("Output Tokens", justify="right")
        proj_table.add_column("Total Tokens", justify="right")
        proj_table.add_column("Cost (USD)", justify="right", style="green")
        proj_table.add_column("Share", justify="right")
        proj_table.add_column("Avg Latency", justify="right")

        for p in by_project:
            p_cost = p.get("cost_usd", 0.0)
            share = (p_cost / total_cost * 100.0) if total_cost > 0 else 0.0
            proj_table.add_row(
                p.get("project", "default"),
                format_tokens(p.get("requests", 0)),
                format_tokens(p.get("input_tokens", 0)),
                format_tokens(p.get("output_tokens", 0)),
                format_tokens(p.get("total_tokens", 0)),
                format_currency(p_cost),
                f"{share:.1f}%",
                format_latency(p.get("avg_latency_ms", 0.0)),
            )
        console.print(proj_table)

    # Provider Breakdown
    by_provider = stats.get("by_provider", [])
    if by_provider:
        prov_table = Table(title="Usage by Provider", show_header=True, header_style="bold magenta")
        prov_table.add_column("Provider")
        prov_table.add_column("Requests", justify="right")
        prov_table.add_column("Tokens", justify="right")
        prov_table.add_column("Cost (USD)", justify="right", style="green")
        prov_table.add_column("Share", justify="right")
        prov_table.add_column("Avg Latency", justify="right")

        for prov in by_provider:
            prov_cost = prov.get("cost_usd", 0.0)
            share = (prov_cost / total_cost * 100.0) if total_cost > 0 else 0.0
            prov_table.add_row(
                format_provider(prov.get("provider", "unknown")),
                format_tokens(prov.get("requests", 0)),
                format_tokens(prov.get("total_tokens", 0)),
                format_currency(prov_cost),
                f"{share:.1f}%",
                format_latency(prov.get("avg_latency_ms", 0.0)),
            )
        console.print(prov_table)

    # Model Breakdown
    by_model = stats.get("by_model", [])
    if by_model:
        model_table = Table(title="Usage by Model", show_header=True, header_style="bold green")
        model_table.add_column("Model", style="bold")
        model_table.add_column("Provider")
        model_table.add_column("Requests", justify="right")
        model_table.add_column("In / Out Tokens", justify="right")
        model_table.add_column("Total Tokens", justify="right")
        model_table.add_column("Cost (USD)", justify="right", style="bold green")
        model_table.add_column("Avg Latency", justify="right")

        for m in by_model:
            model_table.add_row(
                m.get("model", "unknown"),
                format_provider(m.get("provider", "unknown")),
                format_tokens(m.get("requests", 0)),
                f"{format_tokens(m.get('input_tokens', 0))} / {format_tokens(m.get('output_tokens', 0))}",
                format_tokens(m.get("total_tokens", 0)),
                format_currency(m.get("cost_usd", 0.0)),
                format_latency(m.get("avg_latency_ms", 0.0)),
            )
        console.print(model_table)


def build_tail_table(records: list[dict[str, Any]], title: str = "Live Activity") -> Table:
    """Build a Rich Table representing recent/live requests."""
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Time")
    table.add_column("Project", style="bold cyan")
    table.add_column("Provider")
    table.add_column("Model", style="white")
    table.add_column("Tokens (In/Out)", justify="right")
    table.add_column("Cost", justify="right", style="bold green")
    table.add_column("Latency", justify="right")
    table.add_column("Status", justify="center")

    for r in records:
        ts = r.get("timestamp", "")
        # Extract HH:MM:SS from ISO timestamp
        time_part = ts[11:19] if len(ts) >= 19 else ts
        in_tok = r.get("input_tokens", 0)
        out_tok = r.get("output_tokens", 0)
        table.add_row(
            str(r.get("id", "")),
            time_part,
            r.get("project", "default"),
            format_provider(r.get("provider", "")),
            r.get("model", "unknown"),
            f"{format_tokens(in_tok)} / {format_tokens(out_tok)}",
            format_currency(r.get("cost_usd", 0.0)),
            format_latency(r.get("latency_ms", 0.0)),
            format_status_code(r.get("status_code", 200)),
        )

    return table


def render_tail_single(r: dict[str, Any]) -> None:
    """Print a single request record line for tail output."""
    ts = r.get("timestamp", "")
    time_part = ts[11:19] if len(ts) >= 19 else ts
    in_tok = r.get("input_tokens", 0)
    out_tok = r.get("output_tokens", 0)
    status_text = format_status_code(r.get("status_code", 200))
    provider_text = format_provider(r.get("provider", ""))

    console.print(
        f"[dim]{time_part}[/dim] "
        f"[bold cyan]{r.get('project', 'default'):<12}[/bold cyan] "
        f"{provider_text} "
        f"[bold white]{r.get('model', 'unknown'):<24}[/bold white] "
        f"[yellow]{format_tokens(in_tok)} in / {format_tokens(out_tok)} out[/yellow] "
        f"[bold green]{format_currency(r.get('cost_usd', 0.0)):>10}[/bold green] "
        f"[magenta]{format_latency(r.get('latency_ms', 0.0)):>8}[/magenta] "
        f"{status_text}"
    )


def render_models_table(
    pricing_data: dict[str, dict[str, Any]],
    provider_filter: str | None = None,
    search_query: str | None = None,
) -> None:
    """Display table of supported models and pricing."""
    table = Table(
        title="Model Pricing Catalog (USD per 1M tokens)",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Model Identifier", style="bold")
    table.add_column("Display Name")
    table.add_column("Provider")
    table.add_column("Input ($/1M)", justify="right", style="green")
    table.add_column("Output ($/1M)", justify="right", style="bold green")

    prov_clean = provider_filter.lower().strip() if provider_filter else None
    search_clean = search_query.lower().strip() if search_query else None

    # Sort by provider, then model
    items = sorted(pricing_data.items(), key=lambda x: (x[1].get("provider", ""), x[0]))

    match_count = 0
    for model_id, entry in items:
        p = entry.get("provider", "custom")
        if prov_clean and p != prov_clean:
            continue
        if search_clean and (search_clean not in model_id.lower() and search_clean not in entry.get("display", "").lower()):
            continue

        match_count += 1
        table.add_row(
            model_id,
            entry.get("display", model_id),
            format_provider(p),
            f"${entry.get('input', 0.0):.4f}",
            f"${entry.get('output', 0.0):.4f}",
        )

    if match_count == 0:
        console.print("[yellow]No models matched your filter.[/yellow]")
    else:
        console.print(table)
        console.print(
            f"[dim]Total models listed: {match_count}. "
            "Custom pricing can be added via ~/.llm-ledger/pricing.json[/dim]\n"
        )


def render_start_banner(
    host: str,
    port: int,
    db_path: str,
    default_project: str,
) -> None:
    """Print the startup banner when proxy starts."""
    banner = f"""[bold green]llm-ledger[/bold green] [dim]v{__version__}[/dim]
[cyan]Transparent LLM Proxy & Real-Time Cost Tracker[/cyan]

[bold white]Server:[/bold white]          http://{host}:{port}
[bold white]Database:[/bold white]        {db_path}
[bold white]Default Project:[/bold white] {default_project}

[bold yellow]Integration Endpoints:[/bold yellow]
  • OpenAI:      http://{host}:{port}/openai/v1
  • Anthropic:   http://{host}:{port}/anthropic/v1
  • Gemini:      http://{host}:{port}/gemini/v1beta
  • OpenRouter:  http://{host}:{port}/openrouter/api/v1

[bold yellow]Project-Scoped Endpoints:[/bold yellow]
  • http://{host}:{port}/p/[cyan]<project-name>[/cyan]/openai/v1
  • http://{host}:{port}/p/[cyan]<project-name>[/cyan]/anthropic/v1

[dim]Press Ctrl+C to stop the proxy.[/dim]
"""
    console.print(Panel(banner, border_style="blue", title="[bold]Proxy Ready[/bold]"))
