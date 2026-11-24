"""Pricing table and cost calculation engine for LLM providers.

Rates are specified in USD per 1,000,000 (1M) input and output tokens.
Supports standard models from OpenAI, Anthropic, Gemini, OpenRouter,
as well as custom override files (~/.llm-ledger/pricing.json).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_PRICING_PATH = Path.home() / ".llm-ledger" / "pricing.json"

# Built-in rates: USD per 1M tokens
# Format: { model_id: {"provider": str, "input": float, "output": float, "display": str} }
BUILTIN_PRICING: dict[str, dict[str, Any]] = {
    # ------------------ OpenAI ------------------
    "gpt-4o": {
        "provider": "openai",
        "input": 2.50,
        "output": 10.00,
        "display": "GPT-4o",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "input": 0.15,
        "output": 0.60,
        "display": "GPT-4o Mini",
    },
    "o1": {
        "provider": "openai",
        "input": 15.00,
        "output": 60.00,
        "display": "o1",
    },
    "o1-mini": {
        "provider": "openai",
        "input": 3.00,
        "output": 12.00,
        "display": "o1-mini",
    },
    "o1-preview": {
        "provider": "openai",
        "input": 15.00,
        "output": 60.00,
        "display": "o1-preview",
    },
    "o3-mini": {
        "provider": "openai",
        "input": 1.10,
        "output": 4.40,
        "display": "o3-mini",
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "input": 10.00,
        "output": 30.00,
        "display": "GPT-4 Turbo",
    },
    "gpt-4": {
        "provider": "openai",
        "input": 30.00,
        "output": 60.00,
        "display": "GPT-4",
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "input": 0.50,
        "output": 1.50,
        "display": "GPT-3.5 Turbo",
    },
    "text-embedding-3-small": {
        "provider": "openai",
        "input": 0.02,
        "output": 0.00,
        "display": "Embedding 3 Small",
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "input": 0.13,
        "output": 0.00,
        "display": "Embedding 3 Large",
    },
    "text-embedding-ada-002": {
        "provider": "openai",
        "input": 0.10,
        "output": 0.00,
        "display": "Embedding Ada 002",
    },

    # ------------------ Anthropic ------------------
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "input": 3.00,
        "output": 15.00,
        "display": "Claude 3.5 Sonnet",
    },
    "claude-3-5-haiku": {
        "provider": "anthropic",
        "input": 0.80,
        "output": 4.00,
        "display": "Claude 3.5 Haiku",
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "input": 15.00,
        "output": 75.00,
        "display": "Claude 3 Opus",
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "input": 3.00,
        "output": 15.00,
        "display": "Claude 3 Sonnet",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "input": 0.25,
        "output": 1.25,
        "display": "Claude 3 Haiku",
    },

    # ------------------ Gemini ------------------
    "gemini-2.0-flash": {
        "provider": "gemini",
        "input": 0.10,
        "output": 0.40,
        "display": "Gemini 2.0 Flash",
    },
    "gemini-2.0-flash-exp": {
        "provider": "gemini",
        "input": 0.00,
        "output": 0.00,
        "display": "Gemini 2.0 Flash Exp (Free)",
    },
    "gemini-2.0-flash-lite": {
        "provider": "gemini",
        "input": 0.075,
        "output": 0.30,
        "display": "Gemini 2.0 Flash Lite",
    },
    "gemini-1.5-pro": {
        "provider": "gemini",
        "input": 1.25,
        "output": 5.00,
        "display": "Gemini 1.5 Pro",
    },
    "gemini-1.5-flash": {
        "provider": "gemini",
        "input": 0.075,
        "output": 0.30,
        "display": "Gemini 1.5 Flash",
    },
    "gemini-1.5-flash-8b": {
        "provider": "gemini",
        "input": 0.0375,
        "output": 0.15,
        "display": "Gemini 1.5 Flash-8B",
    },
    "gemini-1.0-pro": {
        "provider": "gemini",
        "input": 0.50,
        "output": 1.50,
        "display": "Gemini 1.0 Pro",
    },
    "text-embedding-004": {
        "provider": "gemini",
        "input": 0.00,
        "output": 0.00,
        "display": "Gemini Embedding 004",
    },

    # ------------------ OpenRouter & Popular Open Weights ------------------
    "deepseek-chat": {
        "provider": "openrouter",
        "input": 0.14,
        "output": 0.28,
        "display": "DeepSeek V3 (Chat)",
    },
    "deepseek-reasoner": {
        "provider": "openrouter",
        "input": 0.55,
        "output": 2.19,
        "display": "DeepSeek R1",
    },
    "deepseek/deepseek-chat": {
        "provider": "openrouter",
        "input": 0.14,
        "output": 0.28,
        "display": "DeepSeek V3 (Chat)",
    },
    "deepseek/deepseek-r1": {
        "provider": "openrouter",
        "input": 0.55,
        "output": 2.19,
        "display": "DeepSeek R1",
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "provider": "openrouter",
        "input": 0.13,
        "output": 0.40,
        "display": "Llama 3.3 70B Instruct",
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "provider": "openrouter",
        "input": 2.00,
        "output": 2.00,
        "display": "Llama 3.1 405B Instruct",
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "provider": "openrouter",
        "input": 0.35,
        "output": 0.40,
        "display": "Llama 3.1 70B Instruct",
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "provider": "openrouter",
        "input": 0.05,
        "output": 0.05,
        "display": "Llama 3.1 8B Instruct",
    },
    "mistralai/mistral-large-2407": {
        "provider": "openrouter",
        "input": 2.00,
        "output": 6.00,
        "display": "Mistral Large",
    },
    "mistralai/mistral-small": {
        "provider": "openrouter",
        "input": 0.20,
        "output": 0.60,
        "display": "Mistral Small",
    },
    "qwen/qwen-2.5-72b-instruct": {
        "provider": "openrouter",
        "input": 0.35,
        "output": 0.40,
        "display": "Qwen 2.5 72B Instruct",
    },
}


def load_custom_pricing(file_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load custom pricing rules from a JSON file.
    
    Accepts:
    {
      "custom-model": {
        "input": 1.5,
        "output": 4.5,
        "provider": "custom",
        "display": "Custom Model"
      }
    }
    Or with keys input_cost_per_million / output_cost_per_million.
    """
    path: Path | None = None
    if file_path:
        path = Path(file_path).expanduser().resolve()
    else:
        env_val = os.environ.get("LLM_LEDGER_PRICING")
        if env_val:
            path = Path(env_val).expanduser().resolve()
        elif DEFAULT_PRICING_PATH.is_file():
            path = DEFAULT_PRICING_PATH

    if not path or not path.is_file():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for key, val in data.items():
            if isinstance(val, dict):
                in_cost = float(val.get("input") or val.get("input_cost_per_million") or 0.0)
                out_cost = float(val.get("output") or val.get("output_cost_per_million") or 0.0)
                provider = str(val.get("provider") or "custom").lower()
                display = str(val.get("display") or key)
                normalized[key.lower()] = {
                    "provider": provider,
                    "input": in_cost,
                    "output": out_cost,
                    "display": display,
                }
        return normalized
    except Exception:
        return {}


def _clean_model_name(name: str) -> str:
    """Normalize model string: lowercase, strip, remove leading/trailing slashes."""
    return name.strip().lower()


def find_model_pricing(
    model: str,
    custom_pricing: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Find pricing entry matching model name using exact and fuzzy matching.
    
    Matching order:
    1. Exact match in custom pricing
    2. Exact match in built-in pricing
    3. Strip provider prefix (e.g. 'openai/gpt-4o' -> 'gpt-4o')
    4. Strip dated snapshot suffix (e.g. 'gpt-4o-2024-08-06' -> 'gpt-4o',
       'claude-3-5-sonnet-20241022' -> 'claude-3-5-sonnet',
       'gemini-1.5-pro-002' -> 'gemini-1.5-pro')
    5. Prefix match against known base models
    """
    if not model:
        return None

    cleaned = _clean_model_name(model)
    custom = custom_pricing if custom_pricing is not None else load_custom_pricing()

    # 1. Exact match in custom
    if cleaned in custom:
        return cleaned, custom[cleaned]

    # 2. Exact match in builtin
    if cleaned in BUILTIN_PRICING:
        return cleaned, BUILTIN_PRICING[cleaned]

    # 3. Strip leading provider prefix: e.g. "openai/gpt-4o" -> "gpt-4o"
    if "/" in cleaned:
        unprefixed = cleaned.split("/", 1)[1]
        if unprefixed in custom:
            return unprefixed, custom[unprefixed]
        if unprefixed in BUILTIN_PRICING:
            return unprefixed, BUILTIN_PRICING[unprefixed]
    else:
        unprefixed = cleaned

    # 4. Strip dated snapshot: -YYYYMMDD or -YYYY-MM-DD or -latest or -preview or -001, -002
    simplified = re.sub(r"-\d{4}-?\d{2}-?\d{2}$", "", unprefixed)
    simplified = re.sub(r"-(latest|preview|exp)$", "", simplified)
    simplified = re.sub(r"-\d{3}$", "", simplified)

    if simplified in custom:
        return simplified, custom[simplified]
    if simplified in BUILTIN_PRICING:
        return simplified, BUILTIN_PRICING[simplified]

    # 5. Prefix match: find the longest matching key in custom or builtin
    all_keys = sorted(
        list(custom.keys()) + list(BUILTIN_PRICING.keys()),
        key=len,
        reverse=True,
    )
    for key in all_keys:
        if unprefixed.startswith(key) or simplified.startswith(key):
            entry = custom.get(key) or BUILTIN_PRICING[key]
            return key, entry

    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    custom_pricing: dict[str, dict[str, Any]] | None = None,
) -> float:
    """Calculate USD cost given model and token counts.
    
    Formula: (input_tokens * input_rate + output_tokens * output_rate) / 1,000,000
    If model is not found, returns 0.0.
    """
    match = find_model_pricing(model, custom_pricing)
    if not match:
        return 0.0

    _, entry = match
    input_rate = float(entry.get("input", 0.0))
    output_rate = float(entry.get("output", 0.0))

    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0
    return round(cost, 8)


def get_all_pricing(
    custom_pricing: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return combined dictionary of built-in and custom pricing."""
    custom = custom_pricing if custom_pricing is not None else load_custom_pricing()
    combined = dict(BUILTIN_PRICING)
    combined.update(custom)
    return combined
