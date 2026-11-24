"""End-to-End smoke test runner for llm-ledger.

Tests:
1. Spawns mock upstream LLM server on port 8790.
2. Starts `llm-ledger start --port 8787 --db smoke_test.db` in background process.
3. Tests all routes:
   - /openai/v1/chat/completions
   - /p/demo/openai/v1/chat/completions
   - /anthropic/v1/messages
   - /p/demo/anthropic/v1/messages
   - /gemini/v1beta/models/gemini-1.5-pro:generateContent
   - /p/demo/gemini/v1beta/models/gemini-1.5-pro:generateContent
   - /openrouter/api/v1/chat/completions
   - /p/demo/openrouter/api/v1/chat/completions
   - /openrouter/v1/chat/completions
4. Verifies exact upstream URLs received (checks for no duplicate /v1 or /api segments).
5. Verifies `llm-ledger stats` output.
6. Verifies `llm-ledger export --format json` output.
7. Shuts down background proxy server cleanly.
"""

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MOCK_PORT = 8790
PROXY_PORT = 8787
DB_FILE = Path("smoke_test.db")

received_upstream_requests = []


class MockLLMHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""
        
        received_upstream_requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": body.decode("utf-8", errors="replace"),
        })

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Provider specific mock responses
        if "messages" in self.path:  # Anthropic
            resp = {
                "id": "msg_mock_1",
                "model": "claude-3-5-sonnet",
                "usage": {"input_tokens": 200, "output_tokens": 100},
            }
        elif "gemini" in self.path or "models" in self.path:  # Gemini
            resp = {
                "modelVersion": "gemini-1.5-pro",
                "candidates": [{"content": {"parts": [{"text": "Mock Gemini response"}]}}],
                "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 150, "totalTokenCount": 450},
            }
        elif "openrouter" in self.path or "deepseek" in str(body):  # OpenRouter
            resp = {
                "id": "gen_or_1",
                "model": "deepseek/deepseek-chat",
                "choices": [{"message": {"role": "assistant", "content": "Mock OpenRouter response"}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 200, "total_tokens": 600},
            }
        else:  # OpenAI
            resp = {
                "id": "chatcmpl_mock_1",
                "model": "gpt-4o",
                "choices": [{"message": {"role": "assistant", "content": "Mock OpenAI response"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }

        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress HTTP logging


def run_mock_server():
    server = http.server.HTTPServer(("127.0.0.1", MOCK_PORT), MockLLMHandler)
    server.serve_forever()


def main():
    if DB_FILE.exists():
        DB_FILE.unlink()

    print(f"[*] Starting Mock LLM Server on port {MOCK_PORT}...")
    mock_thread = threading.Thread(target=run_mock_server, daemon=True)
    mock_thread.start()
    time.sleep(0.5)

    # Configure proxy environment to route to mock server
    proxy_env = os.environ.copy()
    proxy_env["LLM_LEDGER_OPENAI_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
    proxy_env["LLM_LEDGER_ANTHROPIC_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
    proxy_env["LLM_LEDGER_GEMINI_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
    proxy_env["LLM_LEDGER_OPENROUTER_URL"] = f"http://127.0.0.1:{MOCK_PORT}"
    proxy_env["PYTHONPATH"] = "src"

    python_exe = sys.executable
    print(f"[*] Launching `llm-ledger start --port {PROXY_PORT}` in background...")
    proxy_proc = subprocess.Popen(
        [
            python_exe,
            "-m",
            "llm_ledger.cli",
            "start",
            "--port",
            str(PROXY_PORT),
            "--db",
            str(DB_FILE),
        ],
        env=proxy_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for proxy server to become healthy
        health_url = f"http://127.0.0.1:{PROXY_PORT}/health"
        healthy = False
        for attempt in range(20):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as resp:
                    if resp.status == 200:
                        healthy = True
                        print("[+] Proxy server is healthy and responding.")
                        break
            except Exception:
                time.sleep(0.3)

        if not healthy:
            stdout, stderr = proxy_proc.communicate(timeout=2)
            print("[!] Proxy failed to start:")
            print("STDOUT:", stdout.decode("utf-8", errors="replace"))
            print("STDERR:", stderr.decode("utf-8", errors="replace"))
            sys.exit(1)

        # -------------------------------------------------------------------
        # Step 2: Test all provider routes and verify prefix stripping
        # -------------------------------------------------------------------
        test_routes = [
            # (name, request_path, payload, expected_upstream_path)
            (
                "OpenAI default route",
                f"http://127.0.0.1:{PROXY_PORT}/openai/v1/chat/completions",
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                "/v1/chat/completions",
            ),
            (
                "OpenAI project-tagged route",
                f"http://127.0.0.1:{PROXY_PORT}/p/demo/openai/v1/chat/completions",
                {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                "/v1/chat/completions",
            ),
            (
                "Anthropic default route",
                f"http://127.0.0.1:{PROXY_PORT}/anthropic/v1/messages",
                {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
                "/v1/messages",
            ),
            (
                "Anthropic project-tagged route",
                f"http://127.0.0.1:{PROXY_PORT}/p/demo/anthropic/v1/messages",
                {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
                "/v1/messages",
            ),
            (
                "Gemini default route",
                f"http://127.0.0.1:{PROXY_PORT}/gemini/v1beta/models/gemini-1.5-pro:generateContent",
                {"contents": [{"parts": [{"text": "hi"}]}]},
                "/v1beta/models/gemini-1.5-pro:generateContent",
            ),
            (
                "Gemini project-tagged route",
                f"http://127.0.0.1:{PROXY_PORT}/p/demo/gemini/v1beta/models/gemini-1.5-pro:generateContent",
                {"contents": [{"parts": [{"text": "hi"}]}]},
                "/v1beta/models/gemini-1.5-pro:generateContent",
            ),
            (
                "OpenRouter API v1 route",
                f"http://127.0.0.1:{PROXY_PORT}/openrouter/api/v1/chat/completions",
                {"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
                "/api/v1/chat/completions",
            ),
            (
                "OpenRouter project-tagged route",
                f"http://127.0.0.1:{PROXY_PORT}/p/demo/openrouter/api/v1/chat/completions",
                {"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
                "/api/v1/chat/completions",
            ),
            (
                "OpenRouter shorthand v1 route",
                f"http://127.0.0.1:{PROXY_PORT}/openrouter/v1/chat/completions",
                {"model": "deepseek/deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
                "/api/v1/chat/completions",
            ),
        ]

        print("\n[*] Sending requests through all provider routes...")
        for name, url, payload, expected_upstream in test_routes:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200, f"{name} failed with status {resp.status}"

            latest_received = received_upstream_requests[-1]
            actual_upstream = latest_received["path"]
            
            # Check for no duplicate segments
            assert "/v1/v1" not in actual_upstream, f"Duplicate /v1 found in {actual_upstream}"
            assert "/api/api" not in actual_upstream, f"Duplicate /api found in {actual_upstream}"
            assert "/api/v1/api/v1" not in actual_upstream, f"Duplicate /api/v1 found in {actual_upstream}"
            assert actual_upstream == expected_upstream, f"Expected {expected_upstream}, got {actual_upstream}"
            print(f"  [OK] {name:<32} -> Upstream: {actual_upstream}")

        # -------------------------------------------------------------------
        # Step 3: Verify stats and export output
        # -------------------------------------------------------------------
        print("\n[*] Verifying `llm-ledger stats`...")
        cmd_env = os.environ.copy()
        cmd_env["PYTHONPATH"] = "src"
        cmd_env["COLUMNS"] = "160"

        stats_cmd = subprocess.run(
            [python_exe, "-m", "llm_ledger.cli", "stats", "--db", str(DB_FILE)],
            env=cmd_env,
            capture_output=True,
            text=True,
        )
        if stats_cmd.returncode != 0:
            print("STATS STDERR:", stats_cmd.stderr)
        assert stats_cmd.returncode == 0
        assert "Total Cost" in stats_cmd.stdout
        assert "Total Requests" in stats_cmd.stdout
        assert "demo" in stats_cmd.stdout
        print(stats_cmd.stdout)

        print("[*] Verifying `llm-ledger export --format json`...")
        export_cmd = subprocess.run(
            [python_exe, "-m", "llm_ledger.cli", "export", "--format", "json", "--db", str(DB_FILE)],
            env=cmd_env,
            capture_output=True,
            text=True,
        )
        assert export_cmd.returncode == 0
        exported_data = json.loads(export_cmd.stdout)
        assert len(exported_data) == len(test_routes), f"Expected {len(test_routes)} records, got {len(exported_data)}"

        print(f"[+] Total exported records: {len(exported_data)}")
        for rec in exported_data:
            assert rec["status_code"] == 200
            assert rec["latency_ms"] >= 0.0
            assert rec["total_tokens"] > 0
            assert rec["cost_usd"] >= 0.0
            print(f"  Record #{rec['id']}: project={rec['project']:<8} provider={rec['provider']:<10} model={rec['model']:<24} tokens={rec['total_tokens']:<5} cost=${rec['cost_usd']:.6f} latency={rec['latency_ms']:.1f}ms")

        print("\n[+] ALL END-TO-END SMOKE TESTS PASSED SUCCESSFULLY!")

    finally:
        # Step 4: Stop the background server
        print("\n[*] Stopping proxy server...")
        proxy_proc.terminate()
        try:
            proxy_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
        print("[+] Proxy server stopped cleanly.")
        time.sleep(0.5)
        try:
            if DB_FILE.exists():
                DB_FILE.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
