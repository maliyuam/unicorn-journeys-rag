"""LLM attribution audit.

The audit exists because two real mis-attributions reached this corpus: an
interview with Dr Omobola Johnson filed under Jeremy Johnson of Andela, and a
Techstars panel hosted by Jenny Fielding filed under Brice Nkengsa. Both are
replayed here against a stub judge, so the plumbing that would surface them
stays correct.

The excerpt sampler is the part that decides whether the judge can see the
truth at all: self-introductions live in the opening chunks, which is where
"my name is Jenny Fielding" appears.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.attribution import audit_corpus, audit_source, sample_excerpts
from backend.llm import LLMUnavailable, OfflineLLM


def _chunk(i, text, source_id="s1"):
    return {"id": f"c{i}", "chunk_index": i, "text": text, "source_id": source_id,
            "founder": "X", "company": "Y", "source_title": "T",
            "source_type": "youtube"}


class StubLLM:
    """Stands in for Claude. Records what it was shown."""

    mode = "claude"
    label = "stub"

    def __init__(self, verdict):
        self.verdict = verdict
        self.seen = []

    def complete_json(self, system, user, schema, max_tokens=4000, effort=None):
        self.seen.append(user)
        return dict(self.verdict)


def test_sampler_always_includes_the_opening():
    """Speakers introduce themselves at the start. A sampler that missed the
    opening would never see "my name is Jenny Fielding"."""
    chunks = [_chunk(0, "welcome everyone my name is Jenny Fielding of techstars")]
    chunks += [_chunk(i, f"filler passage number {i} about markets") for i in range(1, 30)]
    excerpt = sample_excerpts(chunks, "Brice Nkengsa", "Andela")
    assert "Jenny Fielding" in excerpt


def test_sampler_prefers_passages_naming_the_founder():
    chunks = [_chunk(0, "intro music and housekeeping")]
    chunks += [_chunk(i, "generic filler about the market") for i in range(1, 20)]
    chunks.append(_chunk(20, "Aboyeji founded Andela and later Flutterwave here"))
    excerpt = sample_excerpts(chunks, "Iyinoluwa Aboyeji", "Flutterwave Andela")
    assert "Aboyeji founded Andela" in excerpt


def test_sampler_caps_the_payload():
    chunks = [_chunk(i, "x" * 4000) for i in range(20)]
    excerpt = sample_excerpts(chunks, "A Founder", "Company")
    assert len(excerpt) <= 6500, "excerpt would blow the context budget"


def test_sampler_handles_a_single_chunk():
    excerpt = sample_excerpts([_chunk(0, "a short lonely transcript")], "A B", "C")
    assert "lonely" in excerpt


def test_audit_flags_a_different_person_with_the_same_surname():
    """The Omobola Johnson case."""
    llm = StubLLM({
        "verdict": "absent", "confidence": "high",
        "identified_as": "Dr Omobola Johnson, TLcom partner",
        "evidence": "so we actually met through your portfolio company",
        "reason": "The guest is a venture investor, not the Andela co-founder.",
    })
    source = {"source_id": "youtube-x", "founder": "Jeremy Johnson",
              "company": "Andela", "source_title": "Dr Omobola Johnson (TLcom)",
              "source_type": "youtube", "chunks": 29}
    out = audit_source(llm, source, [_chunk(0, "welcome Dr Johnson of TLcom")])
    assert out["verdict"] == "absent"
    assert out["source_id"] == "youtube-x"
    assert "Omobola" in out["identified_as"]
    # the judge must be told the founder AND the company it was filed under
    assert "Jeremy Johnson" in llm.seen[0] and "Andela" in llm.seen[0]


def test_audit_keeps_a_source_whose_name_the_asr_garbled():
    """"Tosi and La" is Tosin Eniolorunda. String matching deletes this; the
    LLM must be able to keep it."""
    llm = StubLLM({
        "verdict": "speaker", "confidence": "high",
        "identified_as": "Tosin Eniolorunda",
        "evidence": "my name is Tosi and La I am the co-founder and CEO",
        "reason": "Self-introduction, phonetically the founder, describes Moniepoint.",
    })
    source = {"source_id": "youtube-y", "founder": "Tosin Eniolorunda",
              "company": "Moniepoint", "source_title": "Moonshot interview",
              "source_type": "youtube", "chunks": 4}
    out = audit_source(llm, source, [
        _chunk(0, "my name is Tosi and La I am the co-founder and CEO of moneyo Inc")
    ])
    assert out["verdict"] == "speaker"


def test_audit_requires_an_llm():
    with pytest.raises(LLMUnavailable, match="needs an LLM"):
        audit_source(OfflineLLM(), {"source_id": "s", "founder": "A B",
                                    "source_title": "t"}, [_chunk(0, "text")])


def test_audit_corpus_summarises_and_flags_removable(monkeypatch, tmp_path):
    from backend import config

    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    from backend.engine import Engine

    engine = Engine()
    long_text = ("The founder launched the company and raised a round. " * 30)
    engine.ingest_documents([
        {"founder": "A Founder", "company": "Co", "source_title": "One",
         "source_type": "youtube", "text": long_text},
        {"founder": "A Founder", "company": "Co", "source_title": "Two",
         "source_type": "youtube", "text": long_text + " different content here."},
    ])

    verdicts = iter([
        {"verdict": "speaker", "confidence": "high", "identified_as": "A Founder",
         "evidence": "e", "reason": "r"},
        {"verdict": "absent", "confidence": "high", "identified_as": "Someone Else",
         "evidence": "e", "reason": "r"},
    ])

    class Seq(StubLLM):
        def complete_json(self, system, user, schema, max_tokens=4000, effort=None):
            return next(verdicts)

    messages = []
    result = audit_corpus(engine, Seq({}), report=lambda p, m: messages.append(m))
    assert result["audited"] == 2
    assert result["counts"]["speaker"] == 1 and result["counts"]["absent"] == 1
    assert len(result["removable"]) == 1
    assert result["removable"][0]["identified_as"] == "Someone Else"
    assert messages


def test_low_confidence_absent_is_not_auto_removable(monkeypatch, tmp_path):
    """Deleting on a guess would destroy real sources. Only confident
    verdicts are actionable."""
    from backend import config

    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    from backend.engine import Engine

    engine = Engine()
    engine.ingest_documents([
        {"founder": "A Founder", "company": "Co", "source_title": "One",
         "source_type": "youtube",
         "text": "The founder launched the company and raised a round. " * 30},
    ])
    llm = StubLLM({"verdict": "absent", "confidence": "low",
                   "identified_as": "unclear", "evidence": "", "reason": "thin"})
    result = audit_corpus(engine, llm)
    assert result["counts"]["absent"] == 1
    assert result["removable"] == [], "a low-confidence guess must not delete data"
