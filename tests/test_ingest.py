"""Ingest pipeline tests: parsing, normalization, validation, upsert
semantics, idempotency, deletion, error isolation, and background jobs.

Run:  python -m pytest tests/ -q
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.chunking import chunk_transcript
from backend.ingest import (
    IngestError,
    normalize_transcript,
    parse_captions,
    parse_upload,
    validate_doc,
)
from backend.jobs import JobManager

LONG_TEXT = (
    "The founder started the company in 2015 after leaving a bank job. "
    "The first product was a payments terminal for small merchants. "
    "In 2018 the company raised a Series A round of ten million dollars. "
    "Regulators initially resisted the agent banking model. "
    "The team overcame this by investing early in compliance. "
) * 6


# --- parsing ---------------------------------------------------------------

SRT = """1
00:00:01,000 --> 00:00:04,000
Hello and welcome to the show.

2
00:00:04,100 --> 00:00:08,000
Today we talk about payments in Africa.
"""

VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:04.000
<c>Hello and welcome</c> to the show.

00:00:04.100 --> 00:00:08.000
Hello and welcome to the show.
Today we talk about payments in Africa.
"""


def test_parse_srt_strips_cues_and_timestamps():
    text = parse_captions(SRT)
    assert "-->" not in text and "00:00" not in text
    assert "Hello and welcome to the show." in text
    assert "payments in Africa" in text


def test_parse_vtt_strips_headers_tags_and_duplicate_lines():
    text = parse_captions(VTT)
    assert "WEBVTT" not in text and "<c>" not in text
    assert text.count("Hello and welcome to the show.") == 1


def test_parse_upload_json_roundtrip():
    raw = b'{"founder": "A", "source_title": "T", "text": "hello world"}'
    doc = parse_upload("x.json", raw)
    assert doc == {"founder": "A", "source_title": "T", "text": "hello world"}


def test_parse_upload_rejects_bad_json_and_extension():
    with pytest.raises(IngestError):
        parse_upload("x.json", b"not json")
    with pytest.raises(IngestError):
        parse_upload("x.exe", b"data")


def test_parse_upload_rejects_oversize(monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 10)
    with pytest.raises(IngestError):
        parse_upload("x.txt", b"x" * 11)


# --- normalization & validation --------------------------------------------

def test_normalize_removes_bracket_cues_and_line_timestamps():
    dirty = "0:01 [Music] Welcome back.\n0:05 We raised (laughs) ten million."
    clean = normalize_transcript(dirty)
    assert "[Music]" not in clean and "(laughs)" not in clean
    assert "0:01" not in clean
    assert "Welcome back." in clean and "ten million." in clean


def test_validate_doc_requires_fields_and_length():
    with pytest.raises(IngestError):
        validate_doc({"source_title": "T", "text": LONG_TEXT})
    with pytest.raises(IngestError):
        validate_doc({"founder": "A", "source_title": "T", "text": "too short"})
    doc = validate_doc({"founder": "A", "source_title": "T", "text": LONG_TEXT})
    assert doc["source_id"] == "a-t"
    assert len(doc["content_hash"]) == 24


# --- chunking ---------------------------------------------------------------

def test_chunks_never_split_sentences():
    chunks = chunk_transcript(
        LONG_TEXT, founder="A", company="C", source_id="s", source_title="t"
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.rstrip().endswith((".", "!", "?"))


def test_unpunctuated_asr_text_is_still_chunked():
    """YouTube auto-captions arrive with no punctuation at all. Without a
    word cap the whole transcript collapses into one chunk and retrieval
    loses all granularity."""
    asr = " ".join(
        "so we started the company in lagos and raised money from investors"
        for _ in range(120)
    )  # ~1440 words, zero sentence boundaries
    chunks = chunk_transcript(
        asr, founder="A", company="C", source_id="s", source_title="t",
        target_words=220,
    )
    assert len(chunks) >= 5, "unpunctuated transcript collapsed into too few chunks"
    for c in chunks:
        assert len(c.text.split()) <= 320, "chunk far exceeds the target size"


# --- engine upsert / idempotency / delete -----------------------------------

@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    from backend.engine import Engine

    return Engine()


def _doc(text=LONG_TEXT, title="Interview one"):
    return {
        "founder": "Test Founder",
        "company": "TestCo",
        "source_title": title,
        "source_type": "podcast",
        "text": text,
    }


def test_ingest_is_idempotent(engine):
    first = engine.ingest_documents([_doc()])
    assert first["ingested"] == 1 and first["chunks"] > 0
    again = engine.ingest_documents([_doc()])
    assert again["ingested"] == 0 and again["skipped_unchanged"] == 1
    assert again["chunks"] == first["chunks"]


def test_reingest_changed_source_replaces_chunks(engine):
    engine.ingest_documents([_doc()])
    before = engine.store.stats()["chunks"]
    changed = _doc(text=LONG_TEXT + " A brand new closing statement was added here.")
    result = engine.ingest_documents([changed])
    assert result["ingested"] == 1
    sources = engine.list_sources()
    assert len(sources) == 1  # replaced, not duplicated
    assert engine.store.stats()["chunks"] >= before


def test_batch_isolates_bad_documents(engine):
    result = engine.ingest_documents(
        [_doc(), {"founder": "", "source_title": "bad", "text": LONG_TEXT}]
    )
    assert result["ingested"] == 1
    assert len(result["failed"]) == 1
    assert "founder" in result["failed"][0]["error"]


def test_delete_source(engine):
    engine.ingest_documents([_doc(), _doc(title="Interview two")])
    sources = engine.list_sources()
    assert len(sources) == 2
    engine.delete_source(sources[0]["source_id"])
    assert len(engine.list_sources()) == 1
    remaining_chunks = engine.store.all_chunks()
    assert all(c["source_id"] != sources[0]["source_id"] for c in remaining_chunks)
    # retrieval still works after deletion
    hits = engine.store.search(
        engine.embedder.embed_query("payments terminal merchants"), 3
    )
    assert hits


def test_index_persists_across_engine_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    from backend.engine import Engine

    e1 = Engine()
    e1.ingest_documents([_doc()])
    n = e1.store.stats()["chunks"]
    e2 = Engine()
    assert e2.store.stats()["chunks"] == n
    assert len(e2.list_sources()) == 1


# --- jobs -------------------------------------------------------------------

def test_job_manager_success_and_failure():
    manager = JobManager()

    def work(report):
        report(0.5, "halfway")
        return {"ok": True}

    job = manager.submit("test", work)
    for _ in range(100):
        if job.status == "done":
            break
        time.sleep(0.02)
    assert job.status == "done" and job.result == {"ok": True}

    def boom(report):
        raise RuntimeError("kaboom")

    bad = manager.submit("test", boom)
    for _ in range(100):
        if bad.status == "error":
            break
        time.sleep(0.02)
    assert bad.status == "error" and "kaboom" in bad.error
