"""Tests for the Starlette transparent HTTP proxy server."""

import json
from pathlib import Path
from typing import AsyncGenerator

import httpx
import pytest
from llm_ledger.db import get_recent_requests, init_db
from llm_ledger.proxy import create_proxy_app, parse_proxy_route


def test_parse_proxy_route():
    # Provider direct
    proj, prov, path = parse_proxy_route("/openai/v1/chat/completions")
    assert proj is None
    assert prov == "openai"
    assert path == "/v1/chat/completions"

    # Project scoped
    proj, prov, path = parse_proxy_route("/p/my-app/openai/v1/chat/completions")
    assert proj == "my-app"
    assert prov == "openai"
    assert path == "/v1/chat/completions"

    # Anthropic
    proj, prov, path = parse_proxy_route("/p/finance/anthropic/v1/messages")
    assert proj == "finance"
    assert prov == "anthropic"
    assert path == "/v1/messages"

    # Gemini
    proj, prov, path = parse_proxy_route("/gemini/v1beta/models/gemini-1.5-pro:generateContent")
    assert proj is None
    assert prov == "gemini"
    assert path == "/v1beta/models/gemini-1.5-pro:generateContent"

    # OpenRouter shorthand /openrouter/v1 -> /api/v1
    proj, prov, path = parse_proxy_route("/openrouter/v1/chat/completions")
    assert proj is None
    assert prov == "openrouter"
    assert path == "/api/v1/chat/completions"


@pytest.mark.asyncio
async def test_health_check_endpoint(tmp_path: Path):
    db_file = tmp_path / "ledger.db"
    app = create_proxy_app(db_path=db_file)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "llm-ledger"


@pytest.mark.asyncio
async def test_unknown_route_returns_404(tmp_path: Path):
    db_file = tmp_path / "ledger.db"
    app = create_proxy_app(db_path=db_file)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/invalid/endpoint")
        assert resp.status_code == 404
        assert "Unknown proxy route" in resp.json()["error"]


@pytest.mark.asyncio
async def test_openai_non_streaming_proxy(tmp_path: Path):
    db_file = tmp_path / "ledger.db"

    # Upstream mock handler
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.openai.com/v1/chat/completions"
        assert request.headers.get("authorization") == "Bearer sk-test-key-12345"
        
        # Upstream returns OpenAI completion
        content = json.dumps({
            "id": "chatcmpl-mock",
            "model": "gpt-4o-2024-08-06",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }).encode("utf-8")
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_proxy_app(db_path=db_file, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/p/web-bot/openai/v1/chat/completions",
            headers={"Authorization": "Bearer sk-test-key-12345", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Confidential Prompt"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chatcmpl-mock"

    # Verify record was stored in SQLite
    records = get_recent_requests(limit=10, db_path=db_file)
    assert len(records) == 1
    r = records[0]
    assert r["project"] == "web-bot"
    assert r["provider"] == "openai"
    assert r["model"] == "gpt-4o-2024-08-06"
    assert r["input_tokens"] == 100
    assert r["output_tokens"] == 50
    assert r["total_tokens"] == 150
    assert r["cost_usd"] > 0.0
    assert r["streaming"] == 0
    assert r["status_code"] == 200

    # Privacy Guarantee Verification:
    # Verify that the SQLite DB does NOT contain any secret prompt or auth token
    import sqlite3
    with sqlite3.connect(db_file) as conn:
        cursor = conn.execute("SELECT * FROM requests")
        row_str = " ".join(str(val) for val in cursor.fetchone())
        assert "sk-test-key" not in row_str
        assert "Confidential Prompt" not in row_str
        assert "Hello world" not in row_str


@pytest.mark.asyncio
async def test_openai_streaming_sse_proxy(tmp_path: Path):
    db_file = tmp_path / "ledger.db"

    # Upstream mock streaming handler
    def mock_stream_handler(request: httpx.Request) -> httpx.Response:
        sse_lines = [
            b'data: {"id":"c-1","choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"id":"c-1","choices":[{"delta":{"content":" world"}}]}\n\n',
            b'data: {"id":"c-1","model":"gpt-4o","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":10,"total_tokens":35}}\n\n',
            b'data: [DONE]\n\n',
        ]
        return httpx.Response(
            200,
            content=b"".join(sse_lines),
            headers={"content-type": "text/event-stream"},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_stream_handler))
    app = create_proxy_app(db_path=db_file, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/openai/v1/chat/completions",
            headers={"X-Project-Id": "streaming-app"},
            json={"model": "gpt-4o", "stream": True},
        )
        assert resp.status_code == 200
        content = resp.text
        assert "[DONE]" in content
        assert "world" in content

    records = get_recent_requests(db_path=db_file)
    assert len(records) == 1
    r = records[0]
    assert r["project"] == "streaming-app"
    assert r["provider"] == "openai"
    assert r["input_tokens"] == 25
    assert r["output_tokens"] == 10
    assert r["total_tokens"] == 35
    assert r["streaming"] == 1


@pytest.mark.asyncio
async def test_anthropic_proxy(tmp_path: Path):
    db_file = tmp_path / "ledger.db"

    def mock_anthropic(request: httpx.Request) -> httpx.Response:
        assert "api.anthropic.com" in str(request.url)
        content = json.dumps({
            "id": "msg_anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 200,
            },
        }).encode("utf-8")
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_anthropic))
    app = create_proxy_app(db_path=db_file, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/p/claude-agent/anthropic/v1/messages",
            headers={"x-api-key": "secret-anthropic-key"},
            json={"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    records = get_recent_requests(db_path=db_file)
    assert len(records) == 1
    r = records[0]
    assert r["project"] == "claude-agent"
    assert r["provider"] == "anthropic"
    assert r["input_tokens"] == 100
    assert r["output_tokens"] == 200
    assert r["total_tokens"] == 300


@pytest.mark.asyncio
async def test_gemini_proxy(tmp_path: Path):
    db_file = tmp_path / "ledger.db"

    def mock_gemini(request: httpx.Request) -> httpx.Response:
        assert "generativelanguage.googleapis.com" in str(request.url)
        content = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}}],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 75,
                "totalTokenCount": 125,
            },
            "modelVersion": "gemini-1.5-pro",
        }).encode("utf-8")
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_gemini))
    app = create_proxy_app(db_path=db_file, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/gemini/v1beta/models/gemini-1.5-pro:generateContent?key=AIzaSySecret",
            json={"contents": [{"parts": [{"text": "hi"}]}]},
        )
        assert resp.status_code == 200

    records = get_recent_requests(db_path=db_file)
    assert len(records) == 1
    r = records[0]
    assert r["provider"] == "gemini"
    assert r["model"] == "gemini-1.5-pro"
    assert r["input_tokens"] == 50
    assert r["output_tokens"] == 75
    # Verify endpoint is stripped of query parameter containing API key
    assert "AIzaSy" not in r["endpoint"]


@pytest.mark.asyncio
async def test_openrouter_proxy(tmp_path: Path):
    db_file = tmp_path / "ledger.db"

    def mock_openrouter(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        content = json.dumps({
            "id": "gen-123",
            "model": "deepseek/deepseek-chat",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 80,
                "total_tokens": 200,
            },
        }).encode("utf-8")
        return httpx.Response(200, content=content, headers={"content-type": "application/json"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_openrouter))
    app = create_proxy_app(db_path=db_file, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Using shorthand /openrouter/v1/chat/completions
        resp = await client.post(
            "/openrouter/v1/chat/completions",
            json={"model": "deepseek/deepseek-chat"},
        )
        assert resp.status_code == 200

    records = get_recent_requests(db_path=db_file)
    assert len(records) == 1
    r = records[0]
    assert r["provider"] == "openrouter"
    assert r["model"] == "deepseek/deepseek-chat"
    assert r["input_tokens"] == 120
    assert r["output_tokens"] == 80
    assert r["cost_usd"] > 0.0


@pytest.mark.asyncio
async def test_upstream_error_recording(tmp_path: Path):
    db_file = tmp_path / "ledger.db"

    def mock_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Rate limit exceeded"})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_error))
    app = create_proxy_app(db_path=db_file, http_client=mock_client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/openai/v1/chat/completions",
            json={"model": "gpt-4o"},
        )
        assert resp.status_code == 429

    records = get_recent_requests(db_path=db_file)
    assert len(records) == 1
    r = records[0]
    assert r["status_code"] == 429
    assert r["cost_usd"] == 0.0

