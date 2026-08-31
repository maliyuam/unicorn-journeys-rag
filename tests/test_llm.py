"""LLM layer robustness.

A response with no text block used to raise StopIteration out of a generator
expression, which killed an entire evaluation run instead of falling back to
the offline path. These pin the graceful failures.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.llm import ClaudeLLM, LLMUnavailable


class _Block:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


def _stub(monkeypatch, response):
    llm = ClaudeLLM.__new__(ClaudeLLM)  # no network, no credentials
    llm.model = "claude-test"
    monkeypatch.setattr(llm, "_create", lambda **kw: response, raising=False)
    return llm


def test_complete_json_raises_on_empty_response(monkeypatch):
    llm = _stub(monkeypatch, _Response([], stop_reason="max_tokens"))
    with pytest.raises(LLMUnavailable, match="empty response"):
        llm.complete_json("sys", "user", {"type": "object"})


def test_complete_json_raises_on_truncated_json(monkeypatch):
    llm = _stub(
        monkeypatch,
        _Response([_Block("text", '{"claims": [{"claim": "a"')], stop_reason="max_tokens"),
    )
    with pytest.raises(LLMUnavailable, match="truncated"):
        llm.complete_json("sys", "user", {"type": "object"})


def test_complete_json_raises_on_malformed_json(monkeypatch):
    llm = _stub(monkeypatch, _Response([_Block("text", "not json at all")]))
    with pytest.raises(LLMUnavailable, match="malformed"):
        llm.complete_json("sys", "user", {"type": "object"})


def test_complete_json_parses_valid_payload(monkeypatch):
    llm = _stub(monkeypatch, _Response([_Block("text", '{"ok": true}')]))
    assert llm.complete_json("sys", "user", {"type": "object"}) == {"ok": True}


def test_complete_ignores_non_text_blocks_but_fails_when_empty(monkeypatch):
    llm = _stub(monkeypatch, _Response([_Block("thinking"), _Block("text", "hello")]))
    assert llm.complete("sys", "user") == "hello"

    llm2 = _stub(monkeypatch, _Response([_Block("thinking")]))
    with pytest.raises(LLMUnavailable, match="empty response"):
        llm2.complete("sys", "user")


def test_refusal_surfaces_as_unavailable(monkeypatch):
    llm = _stub(monkeypatch, _Response([_Block("text", "no")], stop_reason="refusal"))
    with pytest.raises(LLMUnavailable, match="declined"):
        llm.complete("sys", "user")


def test_offline_fallback_is_temporary_not_permanent(monkeypatch):
    """A transient failure — or a key added after start-up — must not pin the
    process to the offline engine forever."""
    import backend.llm as mod

    monkeypatch.setattr(mod, "_llm_instance", None)
    monkeypatch.setattr(mod, "_offline_since", 0.0)
    monkeypatch.setattr(mod, "_credentials_might_exist", lambda: False)

    first = mod.get_llm()
    assert first.mode == "offline"

    # within the cool-off, no re-probe: the same offline instance is reused
    monkeypatch.setattr(mod, "_credentials_might_exist", lambda: True)
    assert mod.get_llm() is first

    # once the cool-off lapses, it probes again and can recover
    monkeypatch.setattr(mod, "_offline_since", -mod._OFFLINE_RETRY_SECONDS * 2)

    class _FakeClaude(mod.ClaudeLLM):
        def __init__(self):
            self.model = "claude-test"

        def complete(self, *a, **kw):
            return "ok"

    monkeypatch.setattr(mod, "ClaudeLLM", _FakeClaude)
    recovered = mod.get_llm()
    assert isinstance(recovered, _FakeClaude)
    # and once live, it is cached without further probing
    assert mod.get_llm() is recovered
