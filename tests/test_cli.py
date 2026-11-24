"""Tests for Typer CLI commands."""

import json
from pathlib import Path
from typer.testing import CliRunner

from llm_ledger.cli import app
from llm_ledger.db import init_db, insert_request

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llm-ledger version" in result.stdout


def test_cli_config(tmp_path: Path):
    db_file = tmp_path / "custom.db"
    result = runner.invoke(app, ["config", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "custom.db" in result.stdout


def test_cli_models():
    result = runner.invoke(app, ["models", "--provider", "openai"])
    assert result.exit_code == 0
    assert "gpt-4o" in result.stdout
    assert "claude" not in result.stdout


def test_cli_stats_empty(tmp_path: Path):
    db_file = tmp_path / "empty.db"
    init_db(db_file)
    result = runner.invoke(app, ["stats", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "No tracked requests found" in result.stdout


def test_cli_stats_populated(tmp_path: Path):
    db_file = tmp_path / "stats.db"
    insert_request(
        {
            "project": "cli-test",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 500,
            "output_tokens": 250,
            "cost_usd": 0.00375,
            "latency_ms": 120.0,
        },
        db_path=db_file,
    )
    result = runner.invoke(app, ["stats", "--db", str(db_file)], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "Total Cost" in result.stdout
    assert "cli-test" in result.stdout


def test_cli_tail_no_follow(tmp_path: Path):
    db_file = tmp_path / "tail.db"
    insert_request(
        {
            "project": "tail-test",
            "provider": "openai",
            "model": "gpt-4o",
            "cost_usd": 0.005,
        },
        db_path=db_file,
    )
    result = runner.invoke(app, ["tail", "--no-follow", "--db", str(db_file)], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "tail-test" in result.stdout


def test_cli_export(tmp_path: Path):
    db_file = tmp_path / "export.db"
    insert_request(
        {
            "project": "proj-exp",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "cost_usd": 0.015,
        },
        db_path=db_file,
    )

    # Export to stdout
    result = runner.invoke(app, ["export", "--format", "json", "--db", str(db_file)])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data) == 1
    assert data[0]["project"] == "proj-exp"

    # Export to file
    out_file = tmp_path / "exported.csv"
    result_csv = runner.invoke(app, ["export", "--format", "csv", "--output", str(out_file), "--db", str(db_file)])
    assert result_csv.exit_code == 0
    assert out_file.exists()
    assert "proj-exp" in out_file.read_text(encoding="utf-8")


def test_cli_reset(tmp_path: Path):
    db_file = tmp_path / "reset.db"
    insert_request({"project": "p1", "model": "m1", "provider": "openai"}, db_path=db_file)
    result = runner.invoke(app, ["reset", "--yes", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "Deleted 1" in result.stdout
