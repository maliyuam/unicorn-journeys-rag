"""Sample corpus tests — the first thing a new clone runs.

The bundled corpus in `data/samples/` is the quickstart: it is what makes
`Load sample corpus` → Ask → Evaluate work before any real media exists. These
tests pin the two ways that path rots silently.

1. **Broken labels.** `data/eval/goldens.samples.json` matches evidence by
   substring against the shipped text. Reword a sample transcript without
   updating the golden and the harness does not fail — it quietly reports the
   golden as unverifiable and scores the run on fewer questions.
2. **Real people in a demo.** The samples are fictional on purpose, so no
   demo ever paraphrases or misattributes a real founder's words. A dropped-in
   real transcript would defeat that silently, so the marker is asserted.

Everything runs on the TF-IDF embedder and a temp index directory, so no test
touches a configured MongoDB or the developer's own corpus.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.chunking import chunk_transcript
from backend.eval_harness import chunk_is_relevant
from backend.ingest import load_sample_docs, validate_doc

SAMPLE_GOLDENS = config.DATA_DIR / "eval" / "goldens.samples.json"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """An isolated engine: local store in tmp_path, deterministic embedder, and
    the offline LLM — so these tests never reach the network, spend an API
    budget, or touch a configured MongoDB."""
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "tfidf")

    from backend import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_credentials_might_exist", lambda: False)
    monkeypatch.setattr(llm_mod, "_llm_instance", None)
    monkeypatch.setattr(llm_mod, "_offline_since", 0.0)

    from backend.engine import Engine

    return Engine()


def _sample_chunks() -> list[dict]:
    chunks: list[dict] = []
    for raw in load_sample_docs():
        doc = validate_doc(raw)
        chunks.extend(
            c.to_dict()
            for c in chunk_transcript(
                text=doc["text"],
                founder=doc["founder"],
                company=doc["company"],
                source_id=doc["source_id"],
                source_title=doc["source_title"],
                source_type=doc["source_type"],
            )
        )
    return chunks


# --- the corpus itself ------------------------------------------------------

def test_sample_corpus_is_present_and_valid():
    docs = load_sample_docs()
    assert len(docs) >= 4, "the shipped quickstart corpus went missing"
    for doc in docs:
        validated = validate_doc(doc)  # raises IngestError on anything malformed
        assert validated["founder"] and validated["company"]
        assert validated["source_type"] in ("youtube", "podcast", "report", "other")


def test_samples_are_marked_fictional():
    """Guards the promise in data/samples/README.md.

    If real interview transcripts are ever dropped in here, the demo would
    start putting real people's words into generated narratives. The marker is
    cheap to keep and the failure it prevents is not.
    """
    for path in sorted(config.SAMPLES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("fictional") is True, f"{path.name} is missing fictional: true"
        assert "FICTIONAL SAMPLE" in data.get("_note", ""), f"{path.name} lacks the note"


def test_samples_chunk_into_a_retrievable_corpus():
    """A single-chunk source cannot demonstrate ranking, so require several."""
    chunks = _sample_chunks()
    assert len(chunks) >= 10
    per_founder: dict[str, int] = {}
    for c in chunks:
        per_founder[c["founder"]] = per_founder.get(c["founder"], 0) + 1
    assert len(per_founder) >= 4
    assert all(n >= 2 for n in per_founder.values()), per_founder


# --- the labels -------------------------------------------------------------

def test_every_sample_golden_resolves_to_a_chunk():
    """No broken labels: each evidence string must appear in its founder's text."""
    goldens = json.loads(SAMPLE_GOLDENS.read_text(encoding="utf-8"))["goldens"]
    assert goldens, "sample golden set is empty"
    chunks = _sample_chunks()

    broken = []
    for g in goldens:
        pool = [c for c in chunks if c["founder"] == g["founder"]]
        if not pool or not any(chunk_is_relevant(c["text"], g["evidence"]) for c in pool):
            broken.append(f"{g['id']}: {g['evidence']}")
    assert not broken, "goldens whose evidence is not in the sample text:\n" + "\n".join(broken)


def test_sample_golden_ids_are_unique():
    goldens = json.loads(SAMPLE_GOLDENS.read_text(encoding="utf-8"))["goldens"]
    ids = [g["id"] for g in goldens]
    assert len(ids) == len(set(ids))


# --- end to end -------------------------------------------------------------

def test_ingesting_samples_indexes_and_is_idempotent(engine):
    docs = load_sample_docs()
    first = engine.ingest_documents(docs)
    assert first["ingested"] == len(docs)
    assert first["chunks"] >= 10
    assert first["failed"] == []

    # Re-running the quickstart must not duplicate the corpus.
    second = engine.ingest_documents(docs)
    assert second["ingested"] == 0
    assert second["skipped_unchanged"] == len(docs)
    assert second["chunks"] == first["chunks"]


def test_retrieval_finds_sample_evidence_offline(engine):
    """The quickstart claim: clone, load samples, ask, get grounded passages —
    with no credentials and no network."""
    engine.ingest_documents(load_sample_docs())
    goldens = json.loads(SAMPLE_GOLDENS.read_text(encoding="utf-8"))["goldens"]

    hits = 0
    for g in goldens:
        result = engine.ask(g["question"], founder=g["founder"], k=7)
        passages = result.get("chunks") or result.get("sources") or []
        texts = [p.get("text", "") for p in passages]
        if any(chunk_is_relevant(t, g["evidence"]) for t in texts):
            hits += 1

    # TF-IDF over 13 chunks with the founder filter applied: most questions
    # should land their evidence in the top 7. This is a smoke test for the
    # retrieval path, not a quality bar — scores live in the eval harness.
    assert hits >= len(goldens) * 0.7, f"only {hits}/{len(goldens)} goldens retrieved"
