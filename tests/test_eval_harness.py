"""Evaluation harness tests.

These guard the measurement itself. A silently wrong metric is worse than no
metric: it makes bad changes look good. Two real bugs are pinned here — a
normalised precision that could exceed 1.0, and a `best_k` that picked the
cheapest k rather than the one that actually gets evidence into context.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.eval_harness import (
    answer_contains_evidence,
    chunk_is_relevant,
    evaluate_retrieval,
    load_goldens,
    validate_goldens,
)

DOC_A = (
    "We started the company in Lagos in 2016 after leaving a bank. "
    "In 2019 we raised eighty one million dollars from investors. "
    "The regulator took two years to approve our licence. "
) * 6
DOC_B = (
    "Our agent network reaches merchants in every state of the country. "
    "We serve small businesses with credit and payment terminals. "
    "Reliability matters more than features in this market. "
) * 6


def test_chunk_is_relevant_normalises_case_and_whitespace():
    assert chunk_is_relevant("We raised  EIGHTY ONE\nmillion dollars", ["eighty one million"])
    assert chunk_is_relevant("Worked with UBER on driver financing", ["worked with uber"])
    assert not chunk_is_relevant("no such figure appears here", ["81 million"])
    # any-of semantics across evidence variants
    assert chunk_is_relevant("he built book netto at school", ["bookneto", "book netto"])


def test_answer_metric_tolerates_paraphrase_but_not_wrong_numbers():
    """Generated answers reword their sources, so exact substring matching
    under-counts correct answers. Figures are the exception: a different
    number is a different answer, never a paraphrase."""
    # paraphrase accepted
    assert answer_contains_evidence(
        "Nadayar Enegesi studied computer science before moving into tech.",
        ["degree in computer science"],
    )
    assert answer_contains_evidence(
        "Flutterwave has worked with Uber to reach drivers.", ["worked with uber"]
    )
    # percent sign normalised to the spoken word
    assert answer_contains_evidence(
        "Headline earnings rose 134% year on year.", ["134 percent"]
    )
    # a wrong figure must fail even though the wording matches
    assert not answer_contains_evidence(
        "They raised eighty two million dollars from investors.", ["81 million"]
    )
    # a missing figure must fail
    assert not answer_contains_evidence(
        "They raised a large round from investors.", ["81 million"]
    )
    # unrelated answer fails
    assert not answer_contains_evidence(
        "The context does not say.", ["degree in computer science"]
    )


def test_retrieval_labels_stay_strict():
    """Retrieval ground truth must NOT inherit the answer metric's tolerance —
    labels have to be mechanical to stay reproducible."""
    assert not chunk_is_relevant(
        "he studied computer science at university", ["degree in computer science"]
    )
    assert chunk_is_relevant(
        "he holds a degree in computer science", ["degree in computer science"]
    )


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    from backend.engine import Engine

    eng = Engine()
    eng.ingest_documents([
        {"founder": "A Founder", "company": "Co", "source_title": "Interview one",
         "source_type": "youtube", "text": DOC_A},
        {"founder": "A Founder", "company": "Co", "source_title": "Interview two",
         "source_type": "youtube", "text": DOC_B},
    ])
    return eng


def test_validate_goldens_flags_evidence_missing_from_corpus(engine):
    goldens = [
        {"id": "ok", "question": "How much was raised?", "founder": "A Founder",
         "evidence": ["eighty one million"]},
        {"id": "bad", "question": "What colour is the logo?", "founder": "A Founder",
         "evidence": ["chartreuse"]},
    ]
    problems = validate_goldens(engine, goldens)
    assert [p["id"] for p in problems] == ["bad"]


def test_normalised_precision_never_exceeds_one(engine):
    """Ceiling and precision must share a denominator. They did not, so k
    values larger than the founder's chunk pool reported prec_n > 1.0."""
    goldens = [{"id": "g", "question": "How much was raised?", "founder": "A Founder",
                "evidence": ["eighty one million"]}]
    result = evaluate_retrieval(engine, goldens, k_values=(3, 50))
    for k, m in result["per_k"].items():
        assert 0.0 <= m["precision_normalized"] <= 1.0, f"prec_n out of range at k={k}"
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["hit_rate"] <= 1.0


def test_best_k_is_cheapest_k_reaching_peak_hit_rate(engine):
    goldens = [{"id": "g", "question": "How much money did they raise from investors?",
                "founder": "A Founder", "evidence": ["eighty one million"]}]
    result = evaluate_retrieval(engine, goldens, k_values=(1, 3, 7, 10))
    per_k = result["per_k"]
    peak = max(m["hit_rate"] for m in per_k.values())
    ties = [k for k, m in per_k.items() if m["hit_rate"] >= peak - 0.01]
    assert result["best_k"] == min(ties)


def test_retrieval_details_support_error_analysis(engine):
    """Every scored question must yield an inspectable row — aggregate-only
    reporting is what let the chunking collapse hide."""
    goldens = [{"id": "g", "question": "How much was raised?", "founder": "A Founder",
                "evidence": ["eighty one million"]}]
    result = evaluate_retrieval(engine, goldens, k_values=(5,))
    row = result["details"][0]
    assert row["id"] == "g" and row["k"] == 5
    assert "top_sources" in row and row["relevant_in_corpus"] >= 1
    assert isinstance(row["hit"], bool)


def test_shipped_goldens_are_live_data_only():
    """No golden may depend on synthetic demo content."""
    goldens = load_goldens()
    assert len(goldens) >= 10
    for g in goldens:
        assert g["evidence"], f"{g['id']} has no evidence strings"
        assert g.get("origin") != "demo", f"{g['id']} still references demo data"
        hint = (g.get("source_hint") or "").lower()
        assert "demo transcript" not in hint, f"{g['id']} points at a demo transcript"
