"""Tests for the asynchronous harness turn flow in the SoT chat server.

These cover the behavior added so a long harness run no longer holds an HTTP
request open: background turns, polled progress, per-send timeouts, and the
"interrupted" signal a restart leaves behind.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sketch_of_thought import server as server_mod
from sketch_of_thought.chat import InvocationResult, ProviderCancelled, ProviderError


class FakeAdapter:
    """Stands in for a real harness adapter."""

    def __init__(self, *, behavior: str = "ok", delay: float = 0.0, error: str = "") -> None:
        self.behavior = behavior
        self.delay = delay
        self.error = error
        self.calls = 0

    def invoke(self, prompt, *, model, native_session_id, initial, system_prompt, cwd, effort="", plan_mode=True, cancel=None, timeout=None, progress=None):
        self.calls += 1
        if progress is not None:
            progress({"kind": "reasoning", "label": "reasoning started", "detail": "thinking about it"})
        if self.behavior == "wait":
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if cancel is not None and cancel.is_set():
                    raise ProviderCancelled("stopped")
                time.sleep(0.02)
        if self.delay:
            time.sleep(self.delay)
        if progress is not None:
            progress({"kind": "tool", "label": "command_execution done", "detail": "ls"})
        if self.behavior == "error":
            raise ProviderError(self.error or "harness exploded")
        return InvocationResult("final answer", "session-1", "raw", "reasoned")


def make_app(monkeypatch, tmp_path, adapter):
    monkeypatch.setenv("SOT_CHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server_mod, "adapter_for", lambda key: adapter)
    return server_mod.ChatApplication()


def new_conversation(app):
    return app.create({"title": "t", "provider": "codex-fugu", "model": "fugu-max", "paradigm": "conceptual_chaining"})


def wait_for_status(app, conversation_id, wanted=("completed", "failed", "cancelled"), timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = app.turn_status(conversation_id)
        if status["status"] in wanted:
            return status
        time.sleep(0.02)
    raise AssertionError(f"turn did not reach {wanted}; last={app.turn_status(conversation_id)}")


def test_resolve_timeout():
    assert server_mod._resolve_timeout(None) >= 1
    assert server_mod._resolve_timeout(120) == 120
    with pytest.raises(ValueError):
        server_mod._resolve_timeout(5)
    with pytest.raises(ValueError):
        server_mod._resolve_timeout("soon")


def test_start_turn_records_progress_and_reply(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter())
    conversation = new_conversation(app)

    turn = app.start_turn(conversation.id, {"message": "hello", "paradigm": "conceptual_chaining", "timeout": 120})
    assert turn["status"] == "running"
    assert turn["timeout"] == 120

    final = wait_for_status(app, conversation.id)
    assert final["status"] == "completed"
    assert final["step_count"] >= 2
    assert any(event["kind"] == "reasoning" for event in final["events"])

    messages = final["conversation"]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "final answer"


def test_failed_turn_keeps_user_message_and_reports_error(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter(behavior="error", error="boom"))
    conversation = new_conversation(app)

    app.start_turn(conversation.id, {"message": "hi", "paradigm": "conceptual_chaining"})
    final = wait_for_status(app, conversation.id)

    assert final["status"] == "failed"
    assert "boom" in (final["error"] or "")
    assert [m["role"] for m in final["conversation"]["messages"]] == ["user"]


def test_cancel_marks_turn_cancelled(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter(behavior="wait"))
    conversation = new_conversation(app)

    app.start_turn(conversation.id, {"message": "long", "paradigm": "conceptual_chaining"})
    time.sleep(0.1)
    assert app.cancel(conversation.id) is True

    final = wait_for_status(app, conversation.id)
    assert final["status"] == "cancelled"
    assert [m["role"] for m in final["conversation"]["messages"]] == ["user"]


def test_busy_conversation_rejects_second_turn(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter(behavior="wait"))
    conversation = new_conversation(app)

    app.start_turn(conversation.id, {"message": "one", "paradigm": "conceptual_chaining"})
    with pytest.raises(server_mod.ConversationBusy):
        app.start_turn(conversation.id, {"message": "two", "paradigm": "conceptual_chaining"})
    app.cancel(conversation.id)


def test_idle_status_marks_interrupted_transcript(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter(behavior="error", error="cut short"))
    conversation = new_conversation(app)
    app.start_turn(conversation.id, {"message": "hi", "paradigm": "conceptual_chaining"})
    wait_for_status(app, conversation.id)

    # Simulate a restart: a fresh application has no in-memory turn state.
    fresh = server_mod.ChatApplication()
    status = fresh.turn_status(conversation.id)
    assert status["status"] == "idle"
    summary = next(c for c in fresh.bootstrap()["conversations"] if c["id"] == conversation.id)
    assert summary["interrupted"] is True


def test_http_messages_returns_202_and_turn_polls(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter(delay=0.05))
    conversation = new_conversation(app)
    handler = type("SoTRequestHandler", (server_mod.RequestHandler,), {"app": app})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        request = urllib.request.Request(
            f"{base}/api/conversations/{conversation.id}/messages",
            data=json.dumps({"message": "hi", "paradigm": "conceptual_chaining"}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 202
            started = json.loads(response.read())
        assert started["status"] == "running"

        deadline = time.monotonic() + 5
        status = started
        while time.monotonic() < deadline:
            with urllib.request.urlopen(f"{base}/api/conversations/{conversation.id}/turn", timeout=5) as response:
                status = json.loads(response.read())
            if status["status"] != "running":
                break
            time.sleep(0.03)
        assert status["status"] == "completed"
        assert status["conversation"]["messages"][-1]["content"] == "final answer"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_rejects_out_of_range_timeout(monkeypatch, tmp_path):
    app = make_app(monkeypatch, tmp_path, FakeAdapter())
    conversation = new_conversation(app)
    handler = type("SoTRequestHandler", (server_mod.RequestHandler,), {"app": app})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        request = urllib.request.Request(
            f"{base}/api/conversations/{conversation.id}/messages",
            data=json.dumps({"message": "hi", "paradigm": "conceptual_chaining", "timeout": 1}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=5)
        assert excinfo.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
