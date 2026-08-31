"""Should this question be answered at all?

Retrieval always returns its best k passages. "Best" is a ranking, not a
finding — ask a corpus of four founder interviews for someone's salary and it
returns the four least-unrelated paragraphs, ranked confidently. Generation
then writes an answer from them and attaches `[S1]` citations, and a citation
reads as evidence. That is how a system produces a sourced-looking claim about
something nobody ever said.

This module is the gate in front of that. It answers one question — do these
passages actually contain what was asked? — and when the answer is no, the
pipeline abstains and returns **no citations at all**.

Two implementations, because the pipeline must work without credentials:

* **LLM gate (accurate, used when credentials resolve).** A schema-constrained
  judgement. It generalises, and it is the one to trust.
* **Heuristic gate (offline fallback).** Three signals combined. Its thresholds
  are calibrated, and the calibration is small enough that you should read
  `config.ABSTAIN_MIN_SIMILARITY` before relying on it.

Why three signals and not one: each was measured separately on 15 answerable
and 14 unanswerable questions against the sample corpus, and each one *on its
own* peaked at ~0.8 accuracy with the two distributions overlapping.

* `question_similarity` alone — BGE compresses unrelated text into a narrow
  band (unanswerable ran 0.48-0.71, answerable 0.65-0.77). Overlapping.
* `rerank_score` alone — better at keeping true answers but let 6 of 14
  unanswerable questions through.
* content-word coverage alone — defeated by questions built from common
  tokens ("Who won the 2019 Formula One championship?" scored 0.80 because
  "2019" and "won" appear in a passage about a default-rate crisis).

Combined they separated cleanly on that set. Read the honest caveat in
`config`: three thresholds fitted on 29 points is a tighter fit than the
evidence supports, so this is a floor for blatant cases, not a guarantee.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from . import config
from .llm import LLMUnavailable
from .prompts import GROUNDING_SYSTEM

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Words carrying no topical signal. A question is judged on what is distinctive
# about it, so the founder's own name and company are excluded separately —
# otherwise every question about a founder looks "covered" by their own corpus.
_STOPWORDS = set(
    """a an the in of to and with for on at by from was is were are had has
have be been being do does did what which when where who whom how why say says
said about their there this that these those it its his her he she they them we
you your i me my our us as or if not no nor so than then too very can could will
would shall should may might must much many more most other some such only own
same just company companies founder founders business businesses year years
thing things""".split()
)

_GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["answerable", "reason"],
    "additionalProperties": False,
}


@dataclass
class SupportVerdict:
    supported: bool
    reason: str
    method: str  # "llm" | "heuristic" | "disabled" | "empty"

    def as_dict(self) -> dict:
        return {"supported": self.supported, "reason": self.reason, "method": self.method}


def _content_tokens(text: str, exclude: tuple[str, ...] = ()) -> set[str]:
    excluded: set[str] = set()
    for item in exclude:
        excluded |= set(_TOKEN_RE.findall((item or "").lower()))
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and t not in excluded and len(t) > 2
    }


def coverage(question: str, chunks: list[dict], founder: str = "", company: str = "") -> float:
    """Fraction of the question's distinctive words that appear in the passages.

    A question the corpus cannot answer usually introduces vocabulary the
    corpus does not have — "salary", "dental", "Mongolia". The founder and
    company are excluded so that naming them cannot inflate the score.
    """
    wanted = _content_tokens(question, exclude=(founder, company))
    if not wanted:
        return 1.0
    body = set(_TOKEN_RE.findall(" ".join(c.get("text", "") for c in chunks).lower()))
    return len(wanted & body) / len(wanted)


def resolve_min_similarity(min_similarity: float | None) -> float:
    """Env override wins; otherwise the embedder's own calibrated floor."""
    if config.ABSTAIN_MIN_SIMILARITY is not None:
        return config.ABSTAIN_MIN_SIMILARITY
    if min_similarity is not None:
        return min_similarity
    return 0.64


def _heuristic_verdict(question, chunks, founder, company, min_similarity) -> SupportVerdict:
    floor = resolve_min_similarity(min_similarity)
    sims = [c["question_similarity"] for c in chunks if c.get("question_similarity") is not None]
    rrs = [c["rerank_score"] for c in chunks if c.get("rerank_score") is not None]
    top_sim = max(sims) if sims else None
    top_rr = max(rrs) if rrs else None
    cov = coverage(question, chunks, founder, company)

    # No similarity signal at all (e.g. embeddings unavailable): do not block.
    # A gate that fires because a diagnostic is missing is worse than no gate.
    if top_sim is None:
        return SupportVerdict(True, "no similarity signal available; not gated", "heuristic")

    # floor <= 0 means this embedder has no usable similarity scale (TF-IDF,
    # where the answerable and unanswerable distributions were measured to
    # overlap completely). Skip the test rather than apply it meaninglessly.
    if floor > 0 and top_sim < floor:
        return SupportVerdict(
            False,
            f"closest passage scores {top_sim:.3f} against the question, below the "
            f"{floor} floor — nothing retrieved is about this",
            "heuristic",
        )
    if cov >= config.ABSTAIN_MIN_COVERAGE:
        return SupportVerdict(
            True, f"{cov:.0%} of the question's distinctive words appear in the passages", "heuristic"
        )
    if top_rr is not None and top_rr >= config.ABSTAIN_MIN_RERANK:
        return SupportVerdict(
            True, f"cross-encoder scores the best passage {top_rr:+.2f} for this question", "heuristic"
        )
    return SupportVerdict(
        False,
        f"only {cov:.0%} of the question's distinctive words appear in the retrieved "
        f"passages"
        + (f" and the cross-encoder scores them {top_rr:+.2f}" if top_rr is not None else "")
        + " — the corpus does not appear to cover this",
        "heuristic",
    )


def _llm_verdict(llm, question, chunks, founder) -> SupportVerdict | None:
    """Ask the model whether the passages answer the question. None on failure."""
    passages = "\n\n".join(
        f"[S{i + 1}] {c.get('source_title', '')}\n{c.get('text', '')}"
        for i, c in enumerate(chunks)
    )
    who = f" The question is about {founder}." if founder else ""
    try:
        result = llm.complete_json(
            system=GROUNDING_SYSTEM,
            user=f"Question: {question}{who}\n\nPassages:\n{passages}",
            schema=_GROUNDING_SCHEMA,
            max_tokens=2000,
            effort="low",
        )
    except (LLMUnavailable, Exception) as e:  # noqa: BLE001 - fall back, never fail the query
        log.warning("grounding check unavailable (%s) — using the offline heuristic", e)
        return None
    return SupportVerdict(
        bool(result.get("answerable")),
        (result.get("reason") or "").strip() or "no reason given",
        "llm",
    )


def assess_support(llm, question, chunks, founder="", company="",
                   min_similarity: float | None = None) -> SupportVerdict:
    """Decide whether `chunks` support answering `question`.

    Falls back from the LLM gate to the heuristic whenever credentials are
    absent or the call fails, so the gate is never simply skipped.
    """
    if not chunks:
        return SupportVerdict(False, "nothing was retrieved for this question", "empty")
    if not config.ABSTAIN_ENABLED:
        return SupportVerdict(True, "grounding gate disabled (ABSTAIN_ENABLED=0)", "disabled")

    if getattr(llm, "mode", None) == "claude":
        verdict = _llm_verdict(llm, question, chunks, founder)
        if verdict is not None:
            return verdict
    return _heuristic_verdict(question, chunks, founder, company, min_similarity)

