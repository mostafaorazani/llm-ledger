# llm-ledger

`llm-ledger` is a local-first transparent proxy and command-line monitoring tool that computes exact token counts, latency metrics, and USD cost breakdowns for calls across LLM providers.

## Key Features

- **Local Storage:** SQLite database kept on your local filesystem.
- **Privacy First:** Headers, authentication tokens, prompt text, and completion text are never stored. Only quantitative metadata (tokens, model, cost, latency, status) is tracked.
- **Project Support:** Route traffic using path prefixes like `/p/<project-name>/<provider>/...` or headers like `X-Project-Id`.
- **Live Monitoring:** Real-time tailing of incoming requests, rich statistical terminal dashboards, and CSV/JSON export.

## CLI Usage

- `llm-ledger start`: Runs the proxy server on port 8787.
- `llm-ledger stats`: Displays summary cards and project/provider breakdowns.
- `llm-ledger tail`: Displays streaming requests in real-time.
- `llm-ledger models`: Shows the pricing table per million tokens.
- `llm-ledger export`: Exports tracked metadata to JSON or CSV.
- `llm-ledger reset`: Clears stored records.

## Proxy Endpoints

Redirect API requests by updating the host base URL:
- OpenAI: `http://localhost:8787/openai/v1` or `http://localhost:8787/p/<project>/openai/v1`
- Anthropic: `http://localhost:8787/anthropic/v1` or `http://localhost:8787/p/<project>/anthropic/v1`
- Gemini: `http://localhost:8787/gemini/v1beta` or `http://localhost:8787/p/<project>/gemini/v1beta`
- OpenRouter: `http://localhost:8787/openrouter/api/v1` or `http://localhost:8787/p/<project>/openrouter/api/v1`
