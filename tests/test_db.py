"""Tests for SQLite database operations and aggregations."""

import json
from pathlib import Path

import pytest
from llm_ledger.db import (
    clear_records,
    export_records,
    get_models,
    get_projects,
    get_recent_requests,
    get_requests_since_id,
    get_stats,
    init_db,
    insert_request,
)


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_ledger.db"
    init_db(db_file)
    return db_file


def test_insert_and_retrieve_stats(test_db: Path):
    id1 = insert_request(
        {
            "project": "app-a",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "cost_usd": 0.00225,
            "latency_ms": 350.0,
            "status_code": 200,
            "streaming": 0,
            "endpoint": "/v1/chat/completions",
        },
        db_path=test_db,
    )
    assert id1 == 1

    id2 = insert_request(
        {
            "project": "app-a",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "input_tokens": 50,
            "output_tokens": 150,
            "total_tokens": 200,
            "cost_usd": 0.00240,
            "latency_ms": 450.0,
            "status_code": 200,
            "streaming": 1,
            "endpoint": "/v1/messages",
        },
        db_path=test_db,
    )
    assert id2 == 2

    id3 = insert_request(
        {
            "project": "app-b",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_tokens": 500,
            "output_tokens": 500,
            "total_tokens": 1000,
            "cost_usd": 0.000375,
            "latency_ms": 200.0,
            "status_code": 200,
            "streaming": 0,
            "endpoint": "/v1/chat/completions",
        },
        db_path=test_db,
    )
    assert id3 == 3

    # Overall stats
    stats = get_stats(db_path=test_db)
    assert stats["total_requests"] == 3
    assert stats["total_input_tokens"] == 650
    assert stats["total_output_tokens"] == 850
    assert stats["total_tokens"] == 1500
    assert abs(stats["total_cost_usd"] - 0.005025) < 1e-6
    assert len(stats["by_project"]) == 2
    assert len(stats["by_provider"]) == 2

    # Filtered by project app-a
    stats_a = get_stats(project="app-a", db_path=test_db)
    assert stats_a["total_requests"] == 2
    assert stats_a["total_tokens"] == 500


def test_recent_requests_and_tail(test_db: Path):
    for i in range(5):
        insert_request(
            {
                "project": f"p-{i}",
                "provider": "openai",
                "model": "gpt-4o",
                "cost_usd": 0.01 * (i + 1),
            },
            db_path=test_db,
        )

    recent = get_recent_requests(limit=3, db_path=test_db)
    assert len(recent) == 3
    # Ordered DESC by id
    assert recent[0]["id"] == 5
    assert recent[1]["id"] == 4
    assert recent[2]["id"] == 3

    # get_requests_since_id for tail polling
    since_3 = get_requests_since_id(last_id=3, db_path=test_db)
    assert len(since_3) == 2
    assert since_3[0]["id"] == 4
    assert since_3[1]["id"] == 5


def test_export_formats(test_db: Path):
    insert_request(
        {
            "project": "export-proj",
            "provider": "openai",
            "model": "gpt-4o",
            "input_tokens": 10,
            "output_tokens": 20,
            "cost_usd": 0.0002,
        },
        db_path=test_db,
    )

    # JSON export
    json_out = export_records(format="json", db_path=test_db)
    parsed = json.loads(json_out)
    assert len(parsed) == 1
    assert parsed[0]["project"] == "export-proj"

    # CSV export
    csv_out = export_records(format="csv", db_path=test_db)
    assert "export-proj" in csv_out
    assert "gpt-4o" in csv_out
    assert "timestamp,project,provider" in csv_out


def test_distinct_projects_and_models(test_db: Path):
    insert_request({"project": "alpha", "model": "gpt-4o", "provider": "openai"}, db_path=test_db)
    insert_request({"project": "beta", "model": "claude-3-opus", "provider": "anthropic"}, db_path=test_db)
    insert_request({"project": "alpha", "model": "gpt-4o-mini", "provider": "openai"}, db_path=test_db)

    projects = get_projects(db_path=test_db)
    assert sorted(projects) == ["alpha", "beta"]

    models = get_models(db_path=test_db)
    assert "gpt-4o" in models
    assert "claude-3-opus" in models


def test_clear_records(test_db: Path):
    insert_request({"project": "p1", "model": "m1", "provider": "openai"}, db_path=test_db)
    insert_request({"project": "p2", "model": "m2", "provider": "anthropic"}, db_path=test_db)

    # Clear only p1
    deleted_p1 = clear_records(project="p1", db_path=test_db)
    assert deleted_p1 == 1

    remaining = get_recent_requests(db_path=test_db)
    assert len(remaining) == 1
    assert remaining[0]["project"] == "p2"

    # Clear all
    deleted_all = clear_records(db_path=test_db)
    assert deleted_all == 1
    assert len(get_recent_requests(db_path=test_db)) == 0
