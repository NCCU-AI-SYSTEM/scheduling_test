"""OpenAI-compatible client for the planner, against a local stub server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm import openai_http, router


class Stub:
    """Serves queued (status, body) replies and records the requests."""

    def __enter__(self):
        stub = self
        self.requests, self.replies = [], []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append((self.path, dict(self.headers), body))
                status, reply = stub.replies.pop(0)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *_a):
        self.server.shutdown()


def _ok(text):
    return 200, {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}


def test_request_shape_and_result(monkeypatch):
    with Stub() as s:
        s.replies.append(_ok('{"need": "x"}'))
        monkeypatch.setenv("V4_OPENAI_BASE_URL", s.url + "/")
        monkeypatch.setenv("V4_OPENAI_API_KEY", "k1")
        schema = {"type": "object"}
        res = openai_http.chat("m1", [{"role": "user", "content": "q"}], fmt=schema,
                               options={"num_gpu": 0, "num_thread": 4})
    path, headers, body = s.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k1"
    assert body["model"] == "m1" and body["temperature"] == 0.0 and body["stream"] is False
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
    assert "options" not in body and "num_gpu" not in body       # Ollama runtime options have no meaning here
    assert res.content == '{"need": "x"}' and res.prompt_tokens == 7 and res.output_tokens == 3 and not res.error


def test_extra_body_from_environment(monkeypatch):
    with Stub() as s:
        s.replies.append(_ok("{}"))
        monkeypatch.setenv("V4_OPENAI_BASE_URL", s.url)
        monkeypatch.setenv("V4_OPENAI_EXTRA", '{"reasoning_effort": "none"}')
        openai_http.chat("m", [{"role": "user", "content": "q"}])
    assert s.requests[0][2]["reasoning_effort"] == "none"


def test_rate_limit_is_retried_then_other_errors_are_reported(monkeypatch):
    monkeypatch.setattr(openai_http.time, "sleep", lambda _s: None)
    with Stub() as s:
        s.replies += [(429, {}), _ok("fine"), (500, {})]
        monkeypatch.setenv("V4_OPENAI_BASE_URL", s.url)
        assert openai_http.chat("m", []).content == "fine"
        bad = openai_http.chat("m", [])
    assert bad.content == "" and bad.error == "HTTP 500"


def test_no_base_url_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("V4_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="V4_OPENAI_BASE_URL"):
        openai_http.chat("m", [])


def test_router_picks_by_environment(monkeypatch):
    called = []
    monkeypatch.setattr(router.openai_http, "chat", lambda *a, **k: called.append("openai"))
    monkeypatch.setattr(router.ollama_http, "chat", lambda *a, **k: called.append("ollama"))
    monkeypatch.delenv("V4_OPENAI_BASE_URL", raising=False)
    router.chat("m", [])
    monkeypatch.setenv("V4_OPENAI_BASE_URL", "http://x/v1")
    router.chat("m", [])
    assert called == ["ollama", "openai"]
