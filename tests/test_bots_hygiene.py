"""ChatGPT bots hygiene — PLAN.md step 8 (mock, redaction, cost guard). No network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.model.ai import chatgpt

pytestmark = pytest.mark.unit


class _CapturingClient:
    """Minimal new-SDK-style fake that records the messages it received."""

    def __init__(self):
        self.captured = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, temperature):
        self.captured = messages
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]
        )


def test_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(chatgpt, "get_secret", lambda *_a, **_k: None)
    out = chatgpt.chatgpt_response("hello")
    assert out.startswith("[ChatGPT unavailable")


def test_secrets_are_redacted_before_send(monkeypatch):
    client = _CapturingClient()
    monkeypatch.setattr(chatgpt, "_get_openai_client", lambda: (client, ""))

    # token assembled at runtime so no literal secret appears in source
    token = "sk-" + "ABCDEFGHIJKLMNOPQRSTUVWX"
    leaked = f"my key is {token} and a token"
    chatgpt.chatgpt_response(leaked)

    sent = client.captured[0]["content"]
    assert token not in sent
    assert "REDACTED" in sent


def test_cost_guard_rejects_oversized_prompt(monkeypatch):
    client = _CapturingClient()
    monkeypatch.setattr(chatgpt, "_get_openai_client", lambda: (client, ""))

    out = chatgpt.chatgpt_response("x" * (chatgpt.MAX_PROMPT_CHARS + 1))
    assert "too long" in out
    assert client.captured is None  # API never called -> no cost incurred


def test_happy_path_returns_completion(monkeypatch):
    client = _CapturingClient()
    monkeypatch.setattr(chatgpt, "_get_openai_client", lambda: (client, ""))
    assert chatgpt.chatgpt_response("hi") == "OK"
