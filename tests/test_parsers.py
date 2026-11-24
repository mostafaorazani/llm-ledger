"""Tests for response and SSE stream parsers."""

import pytest
from llm_ledger.parsers import (
    StreamUsageAccumulator,
    TokenUsage,
    parse_anthropic_response,
    parse_gemini_response,
    parse_openai_response,
    parse_request_metadata,
    parse_response_json,
    parse_sse_chunks,
)


def test_parse_request_metadata():
    body = b'{"model": "gpt-4o", "stream": true, "messages": [{"role": "user", "content": "secret"}]}'
    meta = parse_request_metadata(body)
    assert meta["model"] == "gpt-4o"
    assert meta["stream"] is True
    # Verify prompt content was not captured
    assert "messages" not in meta
    assert "content" not in meta


def test_parse_request_metadata_gemini_path():
    path = "/v1beta/models/gemini-1.5-pro:generateContent"
    meta = parse_request_metadata(b"{}", path=path)
    assert meta["model"] == "gemini-1.5-pro"


def test_parse_openai_response():
    data = {
        "id": "chatcmpl-test",
        "model": "gpt-4o-2024-08-06",
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 28,
            "total_tokens": 40,
        },
    }
    usage = parse_openai_response(data)
    assert usage.input_tokens == 12
    assert usage.output_tokens == 28
    assert usage.total_tokens == 40
    assert usage.model == "gpt-4o-2024-08-06"


def test_parse_anthropic_response():
    data = {
        "id": "msg_01",
        "model": "claude-3-5-sonnet-20241022",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 120,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 5,
        },
    }
    usage = parse_anthropic_response(data)
    assert usage.input_tokens == 65  # 50 + 10 + 5
    assert usage.output_tokens == 120
    assert usage.total_tokens == 185
    assert usage.model == "claude-3-5-sonnet-20241022"


def test_parse_gemini_response():
    data = {
        "modelVersion": "gemini-1.5-flash-001",
        "usageMetadata": {
            "promptTokenCount": 20,
            "candidatesTokenCount": 80,
            "totalTokenCount": 100,
        },
    }
    usage = parse_gemini_response(data)
    assert usage.input_tokens == 20
    assert usage.output_tokens == 80
    assert usage.total_tokens == 100
    assert usage.model == "gemini-1.5-flash-001"


def test_parse_response_json_router():
    # Test router dispatching correctly
    openai_data = {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 5, "completion_tokens": 10}}
    usage = parse_response_json("openai", openai_data)
    assert usage.input_tokens == 5
    assert usage.output_tokens == 10

    anthropic_data = {"model": "claude-3-haiku", "usage": {"input_tokens": 8, "output_tokens": 16}}
    usage2 = parse_response_json("anthropic", anthropic_data)
    assert usage2.input_tokens == 8
    assert usage2.output_tokens == 16


def test_openai_sse_streaming_with_usage():
    sse_text = """
data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":" there"}}]}

data: {"id":"chatcmpl-1","choices":[],"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}

data: [DONE]
"""
    usage = parse_sse_chunks("openai", sse_text, fallback_model="gpt-4o")
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15
    assert usage.model == "gpt-4o"


def test_openai_sse_streaming_without_usage_heuristic():
    sse_text = """
data: {"id":"chatcmpl-1","model":"gpt-4o","choices":[{"delta":{"content":"Hello world, this is a test response."}}]}

data: [DONE]
"""
    usage = parse_sse_chunks("openai", sse_text, fallback_model="gpt-4o")
    # No usage emitted, so fallback heuristic estimates from 38 chars: 38 // 4 = 9 tokens
    assert usage.output_tokens > 0
    assert usage.model == "gpt-4o"


def test_anthropic_sse_streaming():
    sse_text = """
event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":35,"output_tokens":1}}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello!"}}

event: message_delta
data: {"type":"message_delta","usage":{"output_tokens":60}}

event: message_stop
data: {"type":"message_stop"}
"""
    usage = parse_sse_chunks("anthropic", sse_text)
    assert usage.input_tokens == 35
    assert usage.output_tokens == 60
    assert usage.total_tokens == 95
    assert usage.model == "claude-3-5-sonnet-20241022"


def test_gemini_sse_streaming():
    sse_text = """
data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":5,"totalTokenCount":20},"modelVersion":"gemini-1.5-pro"}

data: {"candidates":[{"content":{"parts":[{"text":" there"}]}}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":25,"totalTokenCount":40}}
"""
    usage = parse_sse_chunks("gemini", sse_text)
    assert usage.input_tokens == 15
    assert usage.output_tokens == 25
    assert usage.total_tokens == 40
    assert usage.model == "gemini-1.5-pro"
