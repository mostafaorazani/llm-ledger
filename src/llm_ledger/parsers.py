"""Response and stream parsers for LLM providers.

Extracts token counts (input, output, total) and model metadata from
both non-streaming JSON responses and streaming SSE feeds across:
- OpenAI
- Anthropic
- Gemini
- OpenRouter
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    """Standardized token usage and model metadata."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""

    def finalize(self) -> TokenUsage:
        """Ensure total_tokens is set."""
        if self.total_tokens == 0 and (self.input_tokens > 0 or self.output_tokens > 0):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


def parse_request_metadata(body_bytes: bytes, path: str = "") -> dict[str, Any]:
    """Extract non-sensitive request parameters (model, stream).
    
    NEVER extracts or returns prompt text or credentials.
    """
    metadata: dict[str, Any] = {
        "model": "",
        "stream": False,
    }

    if path:
        # Check for Gemini model in URL path: e.g. /v1beta/models/gemini-1.5-pro:generateContent
        gemini_match = re.search(r"models/([^/:]+)", path)
        if gemini_match:
            metadata["model"] = gemini_match.group(1)

    if not body_bytes:
        return metadata

    try:
        data = json.loads(body_bytes.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            if "model" in data and isinstance(data["model"], str):
                metadata["model"] = data["model"]
            if "stream" in data:
                metadata["stream"] = bool(data["stream"])
    except Exception:
        pass

    return metadata


# ---------------------------------------------------------------------------
# Non-streaming JSON parsers
# ---------------------------------------------------------------------------

def parse_openai_response(data: dict[str, Any], fallback_model: str = "") -> TokenUsage:
    """Parse OpenAI / OpenRouter non-streaming response."""
    model = data.get("model") or fallback_model
    usage = data.get("usage") or {}

    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model=str(model),
    ).finalize()


def parse_anthropic_response(data: dict[str, Any], fallback_model: str = "") -> TokenUsage:
    """Parse Anthropic non-streaming response."""
    model = data.get("model") or fallback_model
    usage = data.get("usage") or {}

    in_tokens = int(usage.get("input_tokens") or 0)
    # Include prompt cache tokens if present
    cache_create = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    input_tokens = in_tokens + cache_create + cache_read

    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = input_tokens + output_tokens

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model=str(model),
    ).finalize()


def parse_gemini_response(data: dict[str, Any], fallback_model: str = "") -> TokenUsage:
    """Parse Google Gemini non-streaming response."""
    model = data.get("modelVersion") or fallback_model
    metadata = data.get("usageMetadata") or {}

    input_tokens = int(metadata.get("promptTokenCount") or 0)
    output_tokens = int(metadata.get("candidatesTokenCount") or 0)
    total_tokens = int(metadata.get("totalTokenCount") or (input_tokens + output_tokens))

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model=str(model),
    ).finalize()


def parse_response_json(
    provider: str,
    data: dict[str, Any],
    fallback_model: str = "",
) -> TokenUsage:
    """Route non-streaming JSON response parsing to provider-specific logic."""
    prov = provider.lower().strip()
    if prov == "anthropic":
        return parse_anthropic_response(data, fallback_model)
    elif prov == "gemini":
        return parse_gemini_response(data, fallback_model)
    elif prov in ("openai", "openrouter"):
        return parse_openai_response(data, fallback_model)
    else:
        # Fallback: check OpenAI schema first, then Anthropic, then Gemini
        if "usage" in data:
            if "prompt_tokens" in data["usage"]:
                return parse_openai_response(data, fallback_model)
            elif "input_tokens" in data["usage"]:
                return parse_anthropic_response(data, fallback_model)
        elif "usageMetadata" in data:
            return parse_gemini_response(data, fallback_model)

        return TokenUsage(model=fallback_model)


# ---------------------------------------------------------------------------
# Streaming SSE parsers
# ---------------------------------------------------------------------------

class StreamUsageAccumulator:
    """Accumulates token usage and model metadata across streaming SSE events."""

    def __init__(self, provider: str, fallback_model: str = "") -> None:
        self.provider = provider.lower().strip()
        self.fallback_model = fallback_model
        self.model = fallback_model
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self._accumulated_chars = 0
        self._current_event_type = ""

    def feed_line(self, line: str) -> None:
        """Feed a single SSE line (e.g. 'data: {...}', 'event: message_delta')."""
        line = line.strip()
        if not line:
            return

        if line.startswith("event:"):
            self._current_event_type = line[6:].strip()
            return

        if not line.startswith("data:"):
            return

        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return

        try:
            payload = json.loads(data_str)
        except Exception:
            return

        if not isinstance(payload, dict):
            return

        self._process_payload(payload)

    def _process_payload(self, payload: dict[str, Any]) -> None:
        # Track model if present
        if "model" in payload and payload["model"]:
            self.model = str(payload["model"])
        elif "modelVersion" in payload and payload["modelVersion"]:
            self.model = str(payload["modelVersion"])

        prov = self.provider

        # Anthropic streaming
        if prov == "anthropic":
            # Event message_start contains message.usage.input_tokens and model
            if payload.get("type") == "message_start":
                msg = payload.get("message") or {}
                if "model" in msg and msg["model"]:
                    self.model = str(msg["model"])
                usage = msg.get("usage") or {}
                self.input_tokens = int(usage.get("input_tokens") or 0)
                cache_create = int(usage.get("cache_creation_input_tokens") or 0)
                cache_read = int(usage.get("cache_read_input_tokens") or 0)
                self.input_tokens += cache_create + cache_read
                if "output_tokens" in usage and usage["output_tokens"]:
                    self.output_tokens = max(self.output_tokens, int(usage["output_tokens"]))

            # Event message_delta contains delta.usage.output_tokens
            elif payload.get("type") == "message_delta":
                usage = payload.get("usage") or {}
                if "output_tokens" in usage:
                    self.output_tokens = max(self.output_tokens, int(usage["output_tokens"]))

            # Also check content_block_delta for text heuristic fallback
            elif payload.get("type") == "content_block_delta":
                delta = payload.get("delta") or {}
                text = delta.get("text") or ""
                self._accumulated_chars += len(text)

        # Gemini streaming
        elif prov == "gemini":
            metadata = payload.get("usageMetadata")
            if metadata and isinstance(metadata, dict):
                if "promptTokenCount" in metadata:
                    self.input_tokens = int(metadata["promptTokenCount"])
                if "candidatesTokenCount" in metadata:
                    self.output_tokens = int(metadata["candidatesTokenCount"])
                if "totalTokenCount" in metadata:
                    self.total_tokens = int(metadata["totalTokenCount"])

        # OpenAI and OpenRouter streaming
        else:
            # Check for usage in chunk (enabled by stream_options.include_usage: true)
            usage = payload.get("usage")
            if usage and isinstance(usage, dict):
                in_tok = usage.get("prompt_tokens")
                out_tok = usage.get("completion_tokens")
                tot_tok = usage.get("total_tokens")
                if in_tok is not None:
                    self.input_tokens = int(in_tok)
                if out_tok is not None:
                    self.output_tokens = int(out_tok)
                if tot_tok is not None:
                    self.total_tokens = int(tot_tok)

            # Accumulate text delta if no usage is found
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                for choice in choices:
                    if isinstance(choice, dict):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if isinstance(content, str):
                            self._accumulated_chars += len(content)

    def get_usage(self) -> TokenUsage:
        """Return final aggregated TokenUsage."""
        # Heuristic fallback if output tokens were not returned by upstream
        out_tokens = self.output_tokens
        if out_tokens == 0 and self._accumulated_chars > 0:
            # Standard heuristic: roughly 4 characters per token
            out_tokens = max(1, self._accumulated_chars // 4)

        in_tokens = self.input_tokens
        tot_tokens = self.total_tokens or (in_tokens + out_tokens)

        return TokenUsage(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=tot_tokens,
            model=self.model or self.fallback_model or "unknown",
        ).finalize()


def parse_sse_chunks(
    provider: str,
    raw_text: str,
    fallback_model: str = "",
) -> TokenUsage:
    """Helper to parse a full SSE string."""
    acc = StreamUsageAccumulator(provider, fallback_model)
    for line in raw_text.splitlines():
        acc.feed_line(line)
    return acc.get_usage()
