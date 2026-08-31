"""LLM layer.

`ClaudeLLM` calls the Anthropic API (claude-opus-5 by default, replacing the
paper's GPT-4o mini) using structured outputs for machine-readable stages.
`OfflineLLM` is a deterministic extractive fallback so the entire pipeline —
expansion, generation, evaluation — still runs with no API credentials.
"""
from __future__ import annotations

import json
import re
import time

from . import config


class LLMUnavailable(Exception):
    pass


class ClaudeLLM:
    mode = "claude"

    def __init__(self):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = config.ANTHROPIC_MODEL

    @property
    def label(self) -> str:
        return f"Claude ({self.model})"

    def _create(self, **kwargs):
        try:
            return self.client.messages.create(model=self.model, **kwargs)
        except self._anthropic.AuthenticationError as e:
            raise LLMUnavailable(str(e)) from e
        except self._anthropic.APIConnectionError as e:
            raise LLMUnavailable(str(e)) from e

    def complete(self, system: str, user: str, max_tokens: int = 8000,
                 effort: str | None = None) -> str:
        kwargs: dict = {
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if effort:
            kwargs["output_config"] = {"effort": effort}
        response = self._create(**kwargs)
        if response.stop_reason == "refusal":
            raise LLMUnavailable("Claude declined this request (safety refusal).")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            raise LLMUnavailable(
                f"empty response from {self.model} (stop_reason={response.stop_reason})"
            )
        return text

    def complete_json(self, system: str, user: str, schema: dict,
                      max_tokens: int = 8000, effort: str | None = None) -> dict:
        kwargs: dict = {
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if effort:
            kwargs["output_config"]["effort"] = effort
        response = self._create(**kwargs)
        if response.stop_reason == "refusal":
            raise LLMUnavailable("Claude declined this request (safety refusal).")
        text = "".join(b.text for b in response.content if b.type == "text")
        if not text.strip():
            # No text block at all — most often the reply was cut off before
            # any content was emitted. Fail as LLMUnavailable so callers fall
            # back to the offline path instead of the whole run dying on a
            # StopIteration from next().
            raise LLMUnavailable(
                f"empty response from {self.model} (stop_reason={response.stop_reason}); "
                "try raising max_tokens"
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            if response.stop_reason == "max_tokens":
                raise LLMUnavailable(
                    f"structured output truncated at max_tokens={max_tokens}"
                ) from e
            raise LLMUnavailable(f"malformed structured output: {text[:160]!r}") from e


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class OfflineLLM:
    """Deterministic extractive fallback — no network, no keys.

    Query expansion is template-based; generation assembles retrieved
    sentences into themed sections; evaluation scores claims by lexical
    overlap with the retrieved context. Clearly surfaced as "offline" in the
    UI so results are never mistaken for model output.
    """

    mode = "offline"
    label = "Offline extractive engine (no API credentials found)"

    def complete(self, system: str, user: str, max_tokens: int = 8000,
                 effort: str | None = None) -> str:
        raise LLMUnavailable("offline")

    def complete_json(self, system: str, user: str, schema: dict,
                      max_tokens: int = 8000, effort: str | None = None) -> dict:
        raise LLMUnavailable("offline")

    # The pipeline calls these purpose-built methods instead when mode=="offline".
    @staticmethod
    def support_score(claim: str, context: str) -> float:
        """Fraction of a claim's content words present in the context."""
        claim_tokens = _tokens(claim)
        if not claim_tokens:
            return 1.0
        ctx_tokens = _tokens(context)
        return len(claim_tokens & ctx_tokens) / len(claim_tokens)


_llm_instance = None
_offline_since = 0.0
# How long to stay offline before probing for Claude again. Without this the
# first transient failure (or a key added after start-up) pins the whole
# process to the offline engine until it is restarted.
_OFFLINE_RETRY_SECONDS = 120.0


def get_llm():
    """Return a live Claude client if credentials resolve, else the offline
    engine. The offline fallback is temporary: it is re-probed periodically so
    a restored key or a passing outage recovers without a restart."""
    global _llm_instance, _offline_since
    if isinstance(_llm_instance, ClaudeLLM):
        return _llm_instance
    if _llm_instance is not None:
        if time.monotonic() - _offline_since < _OFFLINE_RETRY_SECONDS:
            return _llm_instance
        # cool-off elapsed — fall through and probe again
    try:
        # Only worth a network call if some credential source exists (env var
        # or an `ant auth login` profile on disk).
        if _credentials_might_exist():
            candidate = ClaudeLLM()
            candidate.complete(
                system="Reply with the single word: ok",
                user="ok?",
                max_tokens=1200,
                effort="low",
            )
            _llm_instance = candidate
            return _llm_instance
    except Exception:
        pass
    _llm_instance = OfflineLLM()
    _offline_since = time.monotonic()
    return _llm_instance


def _credentials_might_exist() -> bool:
    if config.ANTHROPIC_KEY_PRESENT:
        return True
    import os
    from pathlib import Path

    candidates = []
    if os.getenv("APPDATA"):
        candidates.append(Path(os.environ["APPDATA"]) / "Anthropic")
    home = Path.home()
    candidates.append(home / ".config" / "anthropic")
    return any(p.exists() and any(p.iterdir()) for p in candidates if p.exists())
