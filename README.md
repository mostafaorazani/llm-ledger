# llm-ledger

[![PyPI version](https://img.shields.io/pypi/v/llm-ledger.svg)](https://pypi.org/project/llm-ledger/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/YOUR_GITHUB_USERNAME/llm-ledger/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/llm-ledger/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, local-first HTTP proxy and CLI that tracks real-time token counts, request latency, and exact USD costs across OpenAI, Anthropic, Gemini, and OpenRouter at the project level.

---

## Quick Start

```bash
# 1. Start the proxy (zero-install with uvx)
uvx llm-ledger start

# 2. Update your SDK base URL (append /p/<project-name> to track per project)
# OpenAI:     http://localhost:8787/openai/v1      or  http://localhost:8787/p/my-app/openai/v1
# Anthropic:  http://localhost:8787/anthropic/v1   or  http://localhost:8787/p/my-app/anthropic/v1
# Gemini:     http://localhost:8787/gemini/v1beta  or  http://localhost:8787/p/my-app/gemini/v1beta
# OpenRouter: http://localhost:8787/openrouter/api/v1  or  http://localhost:8787/p/my-app/openrouter/api/v1

# 3. View live stats in another terminal
llm-ledger stats
```

---

## Privacy-First Architecture

`llm-ledger` is designed from the ground up to keep your data private:

- **No Prompt or Response Storage:** Request payloads and generated completions pass through the proxy in-flight without ever being logged or saved to disk.
- **No Credential Logging:** `Authorization`, `x-api-key`, and other authentication headers are stripped before saving metadata and are never persisted in SQLite.
- **Local-Only Storage:** All tracked metadata is stored strictly on your local machine in an SQLite database located at `~/.llm-ledger/ledger.db`.
- **Strictly Quantitative Metadata:** Only operational numbers are saved:
  - Timestamp & project tag
  - Provider & model identifier
  - Input, output, and total token counts
  - Calculated USD cost
  - Roundtrip latency in milliseconds
  - HTTP status code & clean endpoint path

---

## CLI Commands

### 1. View Usage & Cost Dashboards
```bash
llm-ledger stats                      # Overall summary & breakdowns
llm-ledger stats --project my-app     # Filter by project
llm-ledger stats --days 7             # Past 7 days
```

### 2. Follow Requests in Real Time
```bash
llm-ledger tail                       # Live scrolling request feed
llm-ledger tail -p my-app             # Filter tail by project
```

### 3. Check Models & Pricing Catalog
```bash
llm-ledger models                     # Complete USD rate table per 1M tokens
llm-ledger models --provider openai   # Filter by provider
```

### 4. Export Metadata
```bash
llm-ledger export --format json -o ledger.json
llm-ledger export --format csv -o ledger.csv
```

### 5. Custom Pricing Overrides
Create `~/.llm-ledger/pricing.json` to override or add custom model rates (USD per 1,000,000 tokens):
```json
{
  "my-custom-model": {
    "input": 1.50,
    "output": 4.50,
    "provider": "custom",
    "display": "Fine-Tuned Model"
  }
}
```

---

## License

MIT License.
