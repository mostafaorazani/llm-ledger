"""Async transparent HTTP proxy with SSE streaming support.

Routes client requests to LLM providers (OpenAI, Anthropic, Gemini, OpenRouter),
tracks token usage, latency, and cost in real-time, and stores metadata in SQLite.

Zero-friction URL mapping:
- http://localhost:8787/openai/v1/...             -> https://api.openai.com/v1/...
- http://localhost:8787/p/my-app/openai/v1/...    -> https://api.openai.com/v1/... (project='my-app')
- http://localhost:8787/anthropic/v1/...          -> https://api.anthropic.com/v1/...
- http://localhost:8787/gemini/v1beta/...         -> https://generativelanguage.googleapis.com/v1beta/...
- http://localhost:8787/openrouter/api/v1/...     -> https://openrouter.ai/api/v1/...
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from llm_ledger import __version__
from llm_ledger.db import init_db, insert_request
from llm_ledger.parsers import (
    StreamUsageAccumulator,
    parse_request_metadata,
    parse_response_json,
)
from llm_ledger.pricing import calculate_cost, load_custom_pricing

logger = logging.getLogger("llm_ledger.proxy")

PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai",
}


def get_provider_base_url(provider: str) -> str:
    """Return upstream base URL for provider, with env var override support.
    
    Supports:
    - LLM_LEDGER_OPENAI_URL
    - LLM_LEDGER_ANTHROPIC_URL
    - LLM_LEDGER_GEMINI_URL
    - LLM_LEDGER_OPENROUTER_URL
    """
    prov = provider.lower().strip()
    env_var = f"LLM_LEDGER_{prov.upper()}_URL"
    custom_url = os.environ.get(env_var)
    if custom_url:
        return custom_url.rstrip("/")
    return PROVIDER_BASE_URLS.get(prov, "")

HOP_BY_HOP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}

FILTER_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
}


def parse_proxy_route(path: str) -> tuple[str | None, str | None, str]:
    """Parse incoming request path into (project, provider, upstream_path).
    
    Examples:
    - /p/my-app/openai/v1/chat/completions -> ("my-app", "openai", "/v1/chat/completions")
    - /openai/v1/chat/completions          -> (None, "openai", "/v1/chat/completions")
    - /p/rag/gemini/v1beta/models/...      -> ("rag", "gemini", "/v1beta/models/...")
    - /openrouter/v1/chat/completions      -> (None, "openrouter", "/api/v1/chat/completions")
    """
    clean = "/" + path.strip("/")
    
    # Project prefix pattern: /p/{project}/{provider}/{rest}
    project_match = re.match(r"^/p/([^/]+)/([^/]+)(/.*)?$", clean)
    if project_match:
        project = project_match.group(1)
        provider = project_match.group(2).lower()
        upstream_path = project_match.group(3) or "/"
        return project, provider, _normalize_upstream_path(provider, upstream_path)

    # Provider prefix pattern: /{provider}/{rest}
    provider_match = re.match(r"^/([^/]+)(/.*)?$", clean)
    if provider_match:
        candidate_provider = provider_match.group(1).lower()
        if candidate_provider in PROVIDER_BASE_URLS:
            provider = candidate_provider
            upstream_path = provider_match.group(2) or "/"
            return None, provider, _normalize_upstream_path(provider, upstream_path)

    return None, None, clean


def _normalize_upstream_path(provider: str, upstream_path: str) -> str:
    """Normalize vendor-specific shorthand paths and deduplicate repeated segments."""
    if not upstream_path.startswith("/"):
        upstream_path = "/" + upstream_path

    # Deduplicate repeated prefix segments if client library concatenated them
    while upstream_path.startswith("/v1/v1/"):
        upstream_path = upstream_path[3:]
    while upstream_path.startswith("/v1beta/v1beta/"):
        upstream_path = upstream_path[7:]
    while upstream_path.startswith("/api/v1/api/v1/"):
        upstream_path = upstream_path[7:]
    while upstream_path.startswith("/api/api/"):
        upstream_path = upstream_path[4:]

    # For OpenRouter: if client targets /v1/..., redirect to /api/v1/...
    if provider == "openrouter":
        if upstream_path.startswith("/v1/"):
            upstream_path = "/api" + upstream_path
        elif upstream_path == "/v1":
            upstream_path = "/api/v1"

    return upstream_path


def _filter_request_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter out hop-by-hop headers from client request."""
    return {
        k: v for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Filter response headers before returning to client."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() not in FILTER_RESPONSE_HEADERS:
            out[k] = v
    return out


def create_proxy_app(
    db_path: Path | str | None = None,
    default_project: str = "default",
    custom_pricing: dict[str, dict[str, Any]] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Starlette:
    """Create and configure Starlette proxy application."""
    resolved_db = init_db(db_path)
    pricing_data = custom_pricing if custom_pricing is not None else load_custom_pricing()

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        if http_client is not None:
            app.state.client = http_client
            yield
        else:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=60.0),
                follow_redirects=True,
            )
            app.state.client = client
            yield
            await client.aclose()

    async def health_check(_: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "service": "llm-ledger",
            "version": __version__,
            "db_path": str(resolved_db),
        })

    async def proxy_handler(request: Request) -> Response:
        path = request.url.path
        if path in ("", "/"):
            return JSONResponse({
                "status": "ok",
                "service": "llm-ledger",
                "version": __version__,
                "routes": {
                    "openai": "/openai/v1 or /p/<project>/openai/v1",
                    "anthropic": "/anthropic/v1 or /p/<project>/anthropic/v1",
                    "gemini": "/gemini/... or /p/<project>/gemini/...",
                    "openrouter": "/openrouter/api/v1 or /p/<project>/openrouter/api/v1",
                },
            })

        route_project, provider, upstream_path = parse_proxy_route(path)

        if not provider or provider not in PROVIDER_BASE_URLS:
            return JSONResponse(
                {
                    "error": "Unknown proxy route",
                    "message": f"Path '{path}' did not match any supported provider.",
                    "supported_providers": list(PROVIDER_BASE_URLS.keys()),
                    "examples": [
                        "/openai/v1/chat/completions",
                        "/p/my-app/openai/v1/chat/completions",
                        "/anthropic/v1/messages",
                        "/p/my-app/anthropic/v1/messages",
                    ],
                },
                status_code=404,
            )

        # Determine project name
        project = (
            request.headers.get("x-project-id")
            or request.headers.get("x-ledger-project")
            or route_project
            or app.state.default_project
            or "default"
        ).strip()

        # Build upstream URL
        base_url = get_provider_base_url(provider)
        upstream_url = f"{base_url}{upstream_path}"
        if request.url.query:
            upstream_url = f"{upstream_url}?{request.url.query}"

        # Read client body and extract request metadata
        body_bytes = await request.body()
        req_meta = parse_request_metadata(body_bytes, path=upstream_path)
        is_streaming_req = req_meta.get("stream", False)

        # Inject stream_options.include_usage: true for OpenAI/OpenRouter streaming
        # to ensure the provider emits usage chunks
        if is_streaming_req and provider in ("openai", "openrouter") and body_bytes:
            try:
                body_json = json.loads(body_bytes.decode("utf-8"))
                if isinstance(body_json, dict) and "stream_options" not in body_json:
                    body_json["stream_options"] = {"include_usage": True}
                    body_bytes = json.dumps(body_json).encode("utf-8")
            except Exception:
                pass

        # Prepare headers for upstream
        req_headers = _filter_request_headers(dict(request.headers))

        client: httpx.AsyncClient = getattr(app.state, "client", None)
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=60.0),
                follow_redirects=True,
            )
            app.state.client = client
        start_time = time.perf_counter()

        # -------------------------------------------------------------
        # Non-streaming request handling
        # -------------------------------------------------------------
        if not is_streaming_req:
            try:
                upstream_resp = await client.request(
                    method=request.method,
                    url=upstream_url,
                    headers=req_headers,
                    content=body_bytes,
                )
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.error("Error connecting to upstream %s: %s", upstream_url, exc)
                return JSONResponse(
                    {"error": "Proxy error", "details": str(exc)},
                    status_code=502,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse tokens and calculate cost if JSON response
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            model_name = req_meta.get("model", "")
            cost_usd = 0.0

            if upstream_resp.status_code == 200:
                try:
                    data = upstream_resp.json()
                    if isinstance(data, dict):
                        usage = parse_response_json(provider, data, fallback_model=model_name)
                        input_tokens = usage.input_tokens
                        output_tokens = usage.output_tokens
                        total_tokens = usage.total_tokens
                        if usage.model:
                            model_name = usage.model
                        cost_usd = calculate_cost(
                            model_name,
                            input_tokens,
                            output_tokens,
                            app.state.custom_pricing,
                        )
                except Exception as parse_err:
                    logger.debug("Failed parsing non-streaming JSON: %s", parse_err)

            # Record in SQLite (privacy-first: strictly metadata)
            insert_request(
                {
                    "project": project,
                    "provider": provider,
                    "model": model_name or "unknown",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                    "status_code": upstream_resp.status_code,
                    "streaming": 0,
                    "endpoint": upstream_path,
                },
                db_path=app.state.db_path,
            )

            resp_headers = _filter_response_headers(upstream_resp.headers)
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )

        # -------------------------------------------------------------
        # Streaming request handling (SSE)
        # -------------------------------------------------------------
        try:
            req = client.build_request(
                method=request.method,
                url=upstream_url,
                headers=req_headers,
                content=body_bytes,
            )
            upstream_resp = await client.send(req, stream=True)
        except Exception as exc:
            logger.error("Error opening stream to %s: %s", upstream_url, exc)
            return JSONResponse(
                {"error": "Proxy streaming error", "details": str(exc)},
                status_code=502,
            )

        # If upstream responded with an error (e.g. 401, 429), stream error body and log
        if upstream_resp.status_code >= 400:
            error_content = await upstream_resp.aread()
            latency_ms = (time.perf_counter() - start_time) * 1000
            insert_request(
                {
                    "project": project,
                    "provider": provider,
                    "model": req_meta.get("model") or "unknown",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": latency_ms,
                    "status_code": upstream_resp.status_code,
                    "streaming": 1,
                    "endpoint": upstream_path,
                },
                db_path=app.state.db_path,
            )
            resp_headers = _filter_response_headers(upstream_resp.headers)
            return Response(
                content=error_content,
                status_code=upstream_resp.status_code,
                headers=resp_headers,
            )

        accumulator = StreamUsageAccumulator(
            provider=provider,
            fallback_model=req_meta.get("model", ""),
        )

        async def stream_generator() -> AsyncGenerator[bytes, None]:
            line_buffer = ""
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
                    # Feed decoded chunk into accumulator for SSE parsing
                    try:
                        text = chunk.decode("utf-8", errors="replace")
                        line_buffer += text
                        while "\n" in line_buffer:
                            line, line_buffer = line_buffer.split("\n", 1)
                            accumulator.feed_line(line)
                    except Exception:
                        pass
                if line_buffer:
                    accumulator.feed_line(line_buffer)
            finally:
                await upstream_resp.aclose()
                latency_ms = (time.perf_counter() - start_time) * 1000
                usage = accumulator.get_usage()
                final_model = usage.model or req_meta.get("model") or "unknown"
                cost_usd = calculate_cost(
                    final_model,
                    usage.input_tokens,
                    usage.output_tokens,
                    app.state.custom_pricing,
                )

                insert_request(
                    {
                        "project": project,
                        "provider": provider,
                        "model": final_model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost_usd": cost_usd,
                        "latency_ms": latency_ms,
                        "status_code": upstream_resp.status_code,
                        "streaming": 1,
                        "endpoint": upstream_path,
                    },
                    db_path=app.state.db_path,
                )

        resp_headers = _filter_response_headers(upstream_resp.headers)
        media_type = upstream_resp.headers.get("content-type") or "text/event-stream"
        return StreamingResponse(
            stream_generator(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=media_type,
        )

    routes = [
        Route("/health", endpoint=health_check, methods=["GET"]),
        Route("/", endpoint=proxy_handler, methods=["GET"]),
        Route(
            "/{full_path:path}",
            endpoint=proxy_handler,
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        ),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.db_path = resolved_db
    app.state.default_project = default_project
    app.state.custom_pricing = pricing_data
    if http_client is not None:
        app.state.client = http_client

    return app
