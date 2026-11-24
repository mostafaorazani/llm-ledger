"""Database layer for llm-ledger using built-in sqlite3.

Privacy guarantee:
Only metadata (tokens, model, cost, latency, status code, project, provider, endpoint)
is stored. NEVER stores API keys, headers, prompts, or completions.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


DEFAULT_DB_DIR = Path.home() / ".llm-ledger"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "ledger.db"


def get_default_db_path() -> Path:
    """Return configured database path via env var or default location."""
    env_path = os.environ.get("LLM_LEDGER_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_DB_PATH


def resolve_db_path(db_path: Path | str | None = None) -> Path:
    """Resolve given db path or fall back to default."""
    if db_path is not None:
        p = Path(db_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    default_p = get_default_db_path()
    default_p.parent.mkdir(parents=True, exist_ok=True)
    return default_p


def init_db(db_path: Path | str | None = None) -> Path:
    """Initialize SQLite schema and indexes if not already created."""
    path = resolve_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                project TEXT NOT NULL DEFAULT 'default',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL DEFAULT 0.0,
                status_code INTEGER NOT NULL DEFAULT 200,
                streaming INTEGER NOT NULL DEFAULT 0,
                endpoint TEXT NOT NULL DEFAULT ''
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_project ON requests(project);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);"
        )
        conn.commit()
    return path


@contextmanager
def get_db(db_path: Path | str | None = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding SQLite connection with Row row_factory."""
    path = init_db(db_path)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_request(
    record: dict[str, Any],
    db_path: Path | str | None = None,
) -> int:
    """Insert a tracked request metadata record and return its row ID.
    
    Expected keys:
    - timestamp (optional, defaults to now UTC)
    - project (optional, defaults to 'default')
    - provider (required)
    - model (required)
    - input_tokens (int)
    - output_tokens (int)
    - total_tokens (int, calculated if missing)
    - cost_usd (float)
    - latency_ms (float)
    - status_code (int, defaults to 200)
    - streaming (bool/int, defaults to 0)
    - endpoint (str, defaults to '')
    """
    timestamp = record.get("timestamp") or datetime.now(timezone.utc).isoformat()
    project = (record.get("project") or "default").strip()
    provider = (record.get("provider") or "unknown").strip().lower()
    model = (record.get("model") or "unknown").strip()
    input_tokens = int(record.get("input_tokens") or 0)
    output_tokens = int(record.get("output_tokens") or 0)
    total_tokens = int(record.get("total_tokens") or (input_tokens + output_tokens))
    cost_usd = float(record.get("cost_usd") or 0.0)
    latency_ms = float(record.get("latency_ms") or 0.0)
    status_code = int(record.get("status_code") or 200)
    streaming = 1 if record.get("streaming") else 0
    endpoint = str(record.get("endpoint") or "")

    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO requests (
                timestamp, project, provider, model,
                input_tokens, output_tokens, total_tokens,
                cost_usd, latency_ms, status_code, streaming, endpoint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                project,
                provider,
                model,
                input_tokens,
                output_tokens,
                total_tokens,
                cost_usd,
                latency_ms,
                status_code,
                streaming,
                endpoint,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0


def _build_filter_clauses(
    project: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    days: int | None = None,
    since: str | None = None,
    until: str | None = None,
) -> tuple[str, list[Any]]:
    """Build WHERE clauses and parameters for queries."""
    clauses: list[str] = []
    params: list[Any] = []

    if project:
        clauses.append("project = ?")
        params.append(project)
    if provider:
        clauses.append("provider = ?")
        params.append(provider.lower())
    if model:
        clauses.append("model LIKE ?")
        params.append(f"%{model}%")
    if days is not None and days > 0:
        clauses.append("timestamp >= datetime('now', ? || ' days')")
        params.append(f"-{days}")
    if since:
        clauses.append("timestamp >= ?")
        params.append(since)
    if until:
        clauses.append("timestamp <= ?")
        params.append(until)

    where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_str, params


def get_stats(
    project: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    days: int | None = None,
    since: str | None = None,
    until: str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Aggregate statistics for requests matching the given filters."""
    where_sql, params = _build_filter_clauses(
        project=project,
        provider=provider,
        model=model,
        days=days,
        since=since,
        until=until,
    )

    with get_db(db_path) as conn:
        # Overall totals
        overall_query = f"""
            SELECT
                COUNT(*) as total_requests,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms,
                COALESCE(MIN(latency_ms), 0.0) as min_latency_ms,
                COALESCE(MAX(latency_ms), 0.0) as max_latency_ms
            FROM requests
            {where_sql}
        """
        row = conn.execute(overall_query, params).fetchone()
        totals = dict(row) if row else {}

        # Latency percentiles (p50, p90, p99)
        latencies_query = f"""
            SELECT latency_ms FROM requests
            {where_sql}
            ORDER BY latency_ms ASC
        """
        latency_rows = [r["latency_ms"] for r in conn.execute(latencies_query, params).fetchall()]
        n = len(latency_rows)
        p50 = latency_rows[int(n * 0.50)] if n > 0 else 0.0
        p90 = latency_rows[int(n * 0.90)] if n > 0 else 0.0
        p99 = latency_rows[int(n * 0.99)] if n > 0 else 0.0

        # By Project
        project_query = f"""
            SELECT
                project,
                COUNT(*) as requests,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
            FROM requests
            {where_sql}
            GROUP BY project
            ORDER BY cost_usd DESC, requests DESC
        """
        by_project = [dict(r) for r in conn.execute(project_query, params).fetchall()]

        # By Provider
        provider_query = f"""
            SELECT
                provider,
                COUNT(*) as requests,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
            FROM requests
            {where_sql}
            GROUP BY provider
            ORDER BY cost_usd DESC, requests DESC
        """
        by_provider = [dict(r) for r in conn.execute(provider_query, params).fetchall()]

        # By Model
        model_query = f"""
            SELECT
                model,
                provider,
                COUNT(*) as requests,
                COALESCE(SUM(input_tokens), 0) as input_tokens,
                COALESCE(SUM(output_tokens), 0) as output_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
            FROM requests
            {where_sql}
            GROUP BY model, provider
            ORDER BY cost_usd DESC, requests DESC
        """
        by_model = [dict(r) for r in conn.execute(model_query, params).fetchall()]

        # Daily timeline
        daily_query = f"""
            SELECT
                SUBSTR(timestamp, 1, 10) as date,
                COUNT(*) as requests,
                COALESCE(SUM(total_tokens), 0) as tokens,
                COALESCE(SUM(cost_usd), 0.0) as cost_usd
            FROM requests
            {where_sql}
            GROUP BY SUBSTR(timestamp, 1, 10)
            ORDER BY date ASC
        """
        daily_timeline = [dict(r) for r in conn.execute(daily_query, params).fetchall()]

        return {
            "total_requests": totals.get("total_requests", 0),
            "total_input_tokens": totals.get("total_input_tokens", 0),
            "total_output_tokens": totals.get("total_output_tokens", 0),
            "total_tokens": totals.get("total_tokens", 0),
            "total_cost_usd": totals.get("total_cost_usd", 0.0),
            "avg_latency_ms": totals.get("avg_latency_ms", 0.0),
            "min_latency_ms": totals.get("min_latency_ms", 0.0),
            "max_latency_ms": totals.get("max_latency_ms", 0.0),
            "p50_latency_ms": p50,
            "p90_latency_ms": p90,
            "p99_latency_ms": p99,
            "by_project": by_project,
            "by_provider": by_provider,
            "by_model": by_model,
            "daily_timeline": daily_timeline,
        }


def get_recent_requests(
    limit: int = 50,
    offset: int = 0,
    project: str | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent tracked requests."""
    clauses = []
    params: list[Any] = []
    if project:
        clauses.append("project = ?")
        params.append(project)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT * FROM requests
        {where_sql}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_requests_since_id(
    last_id: int,
    project: str | None = None,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return new requests with id > last_id (for live tail streaming)."""
    clauses = ["id > ?"]
    params: list[Any] = [last_id]
    if project:
        clauses.append("project = ?")
        params.append(project)

    query = f"""
        SELECT * FROM requests
        WHERE {' AND '.join(clauses)}
        ORDER BY id ASC
        LIMIT ?
    """
    params.append(limit)

    with get_db(db_path) as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def get_projects(db_path: Path | str | None = None) -> list[str]:
    """Return list of distinct project names."""
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT project FROM requests ORDER BY project").fetchall()
        return [r["project"] for r in rows]


def get_models(db_path: Path | str | None = None) -> list[str]:
    """Return list of distinct model names."""
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT model FROM requests ORDER BY model").fetchall()
        return [r["model"] for r in rows]


def export_records(
    format: str = "json",
    project: str | None = None,
    since: str | None = None,
    until: str | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Export tracked requests in JSON or CSV format."""
    where_sql, params = _build_filter_clauses(project=project, since=since, until=until)
    query = f"SELECT * FROM requests {where_sql} ORDER BY id ASC"

    with get_db(db_path) as conn:
        records = [dict(r) for r in conn.execute(query, params).fetchall()]

    fmt = format.strip().lower()
    if fmt == "csv":
        if not records:
            return "id,timestamp,project,provider,model,input_tokens,output_tokens,total_tokens,cost_usd,latency_ms,status_code,streaming,endpoint\n"
        output = io.StringIO()
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()
    else:
        return json.dumps(records, indent=2)


def clear_records(
    project: str | None = None,
    db_path: Path | str | None = None,
) -> int:
    """Delete records matching criteria and return count of deleted rows."""
    with get_db(db_path) as conn:
        if project:
            cursor = conn.execute("DELETE FROM requests WHERE project = ?", (project,))
        else:
            cursor = conn.execute("DELETE FROM requests")
        conn.commit()
        return cursor.rowcount
