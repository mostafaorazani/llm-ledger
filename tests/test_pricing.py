"""Tests for pricing engine and cost calculations."""

import json
from pathlib import Path

import pytest
from llm_ledger.pricing import (
    calculate_cost,
    find_model_pricing,
    get_all_pricing,
    load_custom_pricing,
)


def test_builtin_model_exact_lookup():
    match = find_model_pricing("gpt-4o")
    assert match is not None
    key, entry = match
    assert key == "gpt-4o"
    assert entry["input"] == 2.50
    assert entry["output"] == 10.00


def test_fuzzy_prefix_and_dated_snapshots():
    # Dated snapshot: gpt-4o-2024-08-06 -> gpt-4o
    match = find_model_pricing("gpt-4o-2024-08-06")
    assert match is not None
    assert match[0] == "gpt-4o"

    # Anthropic dated snapshot
    match_claude = find_model_pricing("claude-3-5-sonnet-20241022")
    assert match_claude is not None
    assert match_claude[0] == "claude-3-5-sonnet"

    # Gemini versioned snapshot
    match_gemini = find_model_pricing("gemini-1.5-flash-002")
    assert match_gemini is not None
    assert match_gemini[0] == "gemini-1.5-flash"


def test_provider_prefix_stripping():
    # OpenAI prefix from OpenRouter or LiteLLM
    match = find_model_pricing("openai/gpt-4o")
    assert match is not None
    assert match[0] == "gpt-4o"


def test_calculate_cost_exact():
    # gpt-4o: 2.50 in / 10.00 out per 1M tokens
    # 1,000 in, 2,000 out -> (1000 * 2.5 + 2000 * 10.0) / 1,000,000
    # = (2,500 + 20,000) / 1,000,000 = 22,500 / 1,000,000 = 0.0225 USD
    cost = calculate_cost("gpt-4o", 1000, 2000)
    assert abs(cost - 0.0225) < 1e-6


def test_free_model_cost():
    # gemini-2.0-flash-exp is 0.0
    cost = calculate_cost("gemini-2.0-flash-exp", 50000, 50000)
    assert cost == 0.0


def test_unknown_model_cost():
    cost = calculate_cost("non-existent-model-xyz", 1000, 1000)
    assert cost == 0.0


def test_custom_pricing_override(tmp_path: Path):
    custom_file = tmp_path / "custom_pricing.json"
    custom_data = {
        "custom-llama-model": {
            "input": 0.50,
            "output": 1.50,
            "provider": "custom",
            "display": "Custom Llama",
        },
        "gpt-4o": {
            "input": 1.00,  # Discounted enterprise rate
            "output": 4.00,
            "provider": "openai",
        },
    }
    custom_file.write_text(json.dumps(custom_data), encoding="utf-8")

    custom_pricing = load_custom_pricing(custom_file)
    assert "custom-llama-model" in custom_pricing
    assert custom_pricing["gpt-4o"]["input"] == 1.00

    # Calculate using custom pricing table
    cost_custom = calculate_cost("custom-llama-model", 10000, 10000, custom_pricing)
    # (10000*0.5 + 10000*1.5)/1M = 20000 / 1M = 0.02
    assert abs(cost_custom - 0.02) < 1e-6

    # Override gpt-4o with discount
    cost_override = calculate_cost("gpt-4o", 10000, 10000, custom_pricing)
    # (10000*1.0 + 10000*4.0)/1M = 50000 / 1M = 0.05
    assert abs(cost_override - 0.05) < 1e-6
