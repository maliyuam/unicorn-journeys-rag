"""The grounding gate: never cite passages that do not support an answer.

This is the pipeline's most consequential behaviour, so it is pinned hard.
Before the gate existed, asking about a person absent from the corpus returned
the top-k passages with `[S1]` citations attached — a sourced-looking claim
about something nobody said. `test_unanswerable_questions_are_refused` is that
exact scenario.

The offline heuristic is what runs without credentials, so it gets the most
attention here: its thresholds are calibrated on a small set (see `config`),
and these tests fix the *behaviour* the calibration is supposed to produce so
a future retune cannot silently break it.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.grounding import assess_support, coverage
from backend.ingest import load_sample_docs

SAMPLE_GOLDENS = config.DATA_DIR / "eval" / "goldens.samples.json"


def _build_engine(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", backend)
    from backend.engine import Engine

    eng = Engine()
    eng.ingest_documents(load_sample_docs())
    return eng


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """TF-IDF: fast, deterministic, and the documented air-gapped fallback."""
    return _build_engine(tmp_path, monkeypatch, "tfidf")


@pytest.fixture
def neural_engine(tmp_path, monkeypatch):
    """The default backend (fastembed/BGE), where the gate actually works.

    Costs a one-off model download on a cold machine, like the other tests that
    build a real Engine.
    """
    return _build_engine(tmp_path, monkeypatch, "auto")


def _negatives():
    return json.loads(SAMPLE_GOLDENS.read_text(encoding="utf-8"))["negative_controls"]


# --- coverage ---------------------------------------------------------------

def test_coverage_ignores_the_founder_and_company_name():
    """Naming the founder must not make a question look supported.

    Every question about a founder mentions them, and their corpus mentions
    them throughout — so counting those tokens would score every question as
    covered, including the ones the corpus cannot answer.
    """
    chunks = [{"text": "Amara Okonkwo founded PaySwift in Lagos in 2017."}]
    # only the founder/company tokens overlap -> nothing distinctive is covered
    assert coverage("What is Amara Okonkwo's PaySwift salary?", chunks,
                    founder="Amara Okonkwo", company="PaySwift") == 0.0
    # a distinctive word that IS present
    assert coverage("When was PaySwift founded?", chunks,
                    founder="Amara Okonkwo", company="PaySwift") == 1.0


def test_coverage_of_a_question_with_no_distinctive_words_is_total():
    assert coverage("What is it about?", [{"text": "anything"}]) == 1.0


# --- the gate ---------------------------------------------------------------

def test_no_chunks_is_never_supported():
    verdict = assess_support(None, "anything?", [])
    assert verdict.supported is False
    assert verdict.method == "empty"


def test_gate_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "ABSTAIN_ENABLED", False)
    verdict = assess_support(None, "q", [{"text": "unrelated", "question_similarity": 0.0}])
    assert verdict.supported is True
    assert verdict.method == "disabled"


def test_missing_similarity_signal_does_not_block(monkeypatch):
    """A gate that fires because a diagnostic is absent is worse than no gate."""
    monkeypatch.setattr(config, "ABSTAIN_ENABLED", True)
    verdict = assess_support(None, "q", [{"text": "some passage"}])
    assert verdict.supported is True


def test_llm_verdict_is_used_when_credentials_resolve(monkeypatch):
    class FakeLLM:
        mode = "claude"

        def complete_json(self, **kwargs):
            return {"answerable": False, "reason": "the passages never mention a salary"}

    verdict = assess_support(
        FakeLLM(), "salary?", [{"text": "x", "question_similarity": 0.99}]
    )
    assert verdict.supported is False
    assert verdict.method == "llm"
    assert "salary" in verdict.reason


def test_falls_back_to_heuristic_when_the_llm_call_fails(monkeypatch):
    """A failing groundedness call must not silently disable the gate."""
    class BrokenLLM:
        mode = "claude"

        def complete_json(self, **kwargs):
            raise RuntimeError("api down")

    verdict = assess_support(
        BrokenLLM(), "What is the capital of Mongolia?",
        [{"text": "unrelated passage about solar panels", "question_similarity": 0.1}],
    )
    assert verdict.method == "heuristic"
    assert verdict.supported is False


# --- end to end, on the shipped corpus --------------------------------------

def test_answerable_questions_are_still_answered(engine):
    """The gate must not cost real answers — that is the whole risk of adding it."""
    goldens = json.loads(SAMPLE_GOLDENS.read_text(encoding="utf-8"))["goldens"]
    refused = [
        g["id"] for g in goldens
        if engine.ask(g["question"], founder=g["founder"], k=7).get("abstained")
    ]
    assert not refused, f"gate wrongly refused answerable questions: {refused}"


def test_unanswerable_questions_are_refused_on_the_default_embedder(neural_engine):
    """The failure this whole module exists to prevent.

    Asserted on the default backend, because that is what the gate is
    calibrated for and what almost everyone runs.
    """
    negatives = _negatives()
    assert negatives, "negative controls went missing from the sample golden set"

    answered = []
    for n in negatives:
        result = neural_engine.ask(n["question"], founder=n.get("founder"), k=7)
        if not result.get("abstained"):
            answered.append(n["id"])
        else:
            # An abstention that still hands back passages is not an abstention:
            # the UI renders them as sources and the distinction is lost.
            assert result["chunks"] == []
            assert result["citations"] == {}
    assert not answered, f"answered questions the corpus cannot support: {answered}"


def test_tfidf_abstention_is_weaker_but_not_absent(engine):
    """Pins the *measured* limit of the offline fallback rather than an aspiration.

    Under TF-IDF the similarity signal is dead — answerable and unanswerable
    questions were measured at 0.083-0.270 and 0.073-0.264 respectively, with
    the unanswerable median actually *higher* — so the gate runs on word
    coverage and the cross-encoder alone, using thresholds calibrated for BGE.

    Measured result: it refuses 7 of 14. Half the unanswerable questions still
    get an answer, "What is the capital of Mongolia?" among them, because
    "capital" happens to occur in a passage about merchant cash floats.

    **This is the documented limit of the offline fallback, not a target.** If
    abstention matters to you, run the default embedder or supply credentials.
    The number is pinned so the weakness stays visible and any regression is
    caught; tighten it if you improve the heuristic.
    """
    negatives = _negatives()
    refused = sum(
        1 for n in negatives
        if engine.ask(n["question"], founder=n.get("founder"), k=7).get("abstained")
    )
    assert refused >= 7, (
        f"offline gate refused only {refused}/{len(negatives)} — below the "
        "measured TF-IDF baseline of 7"
    )


def test_abstention_answer_text_is_explicit(neural_engine):
    result = neural_engine.ask("What is the capital of Mongolia?", k=7)
    assert result["abstained"] is True
    assert "cannot answer" in result["answer"].lower()
    assert "no sources are cited" in result["answer"].lower()
