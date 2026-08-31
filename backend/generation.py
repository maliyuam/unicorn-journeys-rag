"""Grounded generation: founder narratives (six themes, per the paper) and
free-form Q&A. Every context chunk gets a citation tag [S1]..[Sn] so the UI
can trace each statement back to its source — an upgrade over the paper,
which reported source grounding but did not surface inline citations.
"""
from __future__ import annotations

import re

from .llm import LLMUnavailable
from .retrieval import tokenize

_SOURCE_CAVEATS = """The passages are machine transcripts of interviews, \
podcasts, keynotes and panels. Read them with these properties in mind:

- **Speech recognition errors are common.** Proper nouns are often mangled \
("michelle eligbe" for Mitchell Elegbe, "olubenga GB apula" for Olugbenga \
Agboola, "flatterwave" for Flutterwave, "book netto" for Bookneto). When a \
garbled token is clearly the founder or company already named in the passage, \
use the correct spelling. Never let a misspelling invent a second person or a \
different company.
- **There is no punctuation or casing in auto-generated captions**, and \
fillers ("so", "um", "you know") are frequent. Judge meaning, not form.
- **Speakers are not labelled.** A passage may contain a host's question, \
another panellist's answer, or a third party being discussed. Attribute a \
statement to the founder only when the passage makes the speaker clear. If a \
claim could belong to another speaker, either omit it or mark it as stated in \
the source rather than as the founder's own words.
- **Numbers spoken aloud are error-prone.** Report a figure only as the \
passage states it; never convert, round, or reconcile conflicting figures."""

_NARRATIVE_SYSTEM = f"""You are a professor of Entrepreneurship and Innovation \
with expertise in African entrepreneurial ecosystems. You are reconstructing a \
founder's entrepreneurial journey from retrieved source material for an \
academic study.

{_SOURCE_CAVEATS}

Work in two steps. First, silently identify which passages actually concern \
this founder and what each one establishes. Then write the narrative using \
ONLY those passages, with exactly these Markdown sections in this order:

## Founder and Company
## Timeline of Critical Events
## Key Success Markers
## Ecosystem Influences and Interactions
## Challenges and Resilience
## Impact on the Ecosystem

Rules:
- Every factual statement must be supported by the passages and must cite them \
inline like [S1] or [S2][S5]. A sentence with no citation is not allowed.
- If the passages lack information for a section, write exactly: \
"No relevant information in the retrieved context." — do not pad the section \
with generalities about African entrepreneurship.
- Never use outside knowledge, even for facts you are confident about. If you \
know a figure that the passages do not state, leave it out.
- Prefer concrete dates, amounts, organisations and named people. When a \
passage gives a relative time ("two years later", "when we started"), keep the \
relative phrasing rather than computing a year.
- Under Timeline, order events chronologically where dates allow, and place \
undated events at the end marked "(date not stated)".
- Do not editorialise, praise, or draw lessons. Report what the sources show."""

_ANSWER_SYSTEM = f"""You answer questions about African startup founders using \
ONLY the numbered context passages provided.

{_SOURCE_CAVEATS}

Rules:
- Answer the question directly in the first sentence, then add only the detail \
the passages support.
- Cite every factual claim inline like [S1].
- If the passages do not answer the question, say exactly what is missing \
rather than guessing or substituting general knowledge. Partial answers are \
fine when you say which part is unsupported.
- Prefer the passages' own figures, dates and names. Quote a short phrase when \
the exact wording matters.
- Be concise. No preamble, no restating the question."""


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[S{i}] (source: {chunk['source_title']}, {chunk['source_type']}, "
            f"founder: {chunk['founder']})\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def citation_map(chunks: list[dict]) -> list[dict]:
    return [
        {
            "tag": f"S{i}",
            "chunk_id": chunk["id"],
            "source_title": chunk["source_title"],
            "source_type": chunk["source_type"],
            "founder": chunk["founder"],
        }
        for i, chunk in enumerate(chunks, 1)
    ]


def generate_narrative(llm, founder: str, company: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    if llm.mode == "claude":
        try:
            return llm.complete(
                system=_NARRATIVE_SYSTEM,
                user=(
                    f"Founder: {founder} ({company}).\n\nContext passages:\n\n{context}"
                ),
                max_tokens=10000,
            )
        except LLMUnavailable:
            pass
    return _offline_narrative(founder, company, chunks)


def generate_answer(llm, question: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    if llm.mode == "claude":
        try:
            return llm.complete(
                system=_ANSWER_SYSTEM,
                user=f"Question: {question}\n\nContext passages:\n\n{context}",
                max_tokens=4000,
            )
        except LLMUnavailable:
            pass
    return _offline_answer(question, chunks)


# --- offline extractive fallbacks ------------------------------------------

_THEME_KEYWORDS = {
    "Founder and Company": ["founder", "founded", "co-founded", "ceo", "company", "startup"],
    "Timeline of Critical Events": ["in 19", "in 20", "year", "launched", "started", "began", "became"],
    "Key Success Markers": ["raised", "million", "billion", "customers", "profitable", "valuation", "unicorn", "users"],
    "Ecosystem Influences and Interactions": ["mentor", "accelerator", "network", "partner", "investor", "diaspora"],
    "Challenges and Resilience": ["challenge", "difficult", "struggle", "regulat", "overcame", "failed", "hurdle"],
    "Impact on the Ecosystem": ["jobs", "impact", "trained", "inclusion", "policy", "community", "inspired"],
}


def _sentences_with_tags(chunks: list[dict]) -> list[tuple[str, str]]:
    out = []
    for i, chunk in enumerate(chunks, 1):
        for sent in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            sent = sent.strip()
            if len(sent.split()) >= 6:
                out.append((sent, f"S{i}"))
    return out


def _offline_narrative(founder: str, company: str, chunks: list[dict]) -> str:
    tagged = _sentences_with_tags(chunks)
    parts = [
        "> _Generated by the offline extractive engine (no LLM credentials). "
        "Sentences are selected verbatim from retrieved sources._",
        f"\n## Founder and Company\n{founder}, associated with {company}. [S1]"
        if chunks else "\n## Founder and Company\nNo relevant information in the retrieved context.",
    ]
    used: set[str] = set()
    for section, keywords in list(_THEME_KEYWORDS.items())[1:]:
        picks = []
        for sent, tag in tagged:
            lower = sent.lower()
            if sent in used:
                continue
            if any(k in lower for k in keywords):
                picks.append(f"- {sent} [{tag}]")
                used.add(sent)
            if len(picks) >= 5:
                break
        body = "\n".join(picks) if picks else "No relevant information in the retrieved context."
        parts.append(f"\n## {section}\n{body}")
    return "\n".join(parts)


def _offline_answer(question: str, chunks: list[dict]) -> str:
    q_tokens = set(tokenize(question))
    scored = []
    seen: set[str] = set()
    for sent, tag in _sentences_with_tags(chunks):
        if sent in seen:  # chunk overlap repeats boundary sentences
            continue
        seen.add(sent)
        overlap = len(q_tokens & set(tokenize(sent)))
        if overlap:
            scored.append((overlap, sent, tag))
    scored.sort(key=lambda t: -t[0])
    if not scored:
        return "The retrieved context does not contain an answer to this question."
    lines = ["_Offline extractive answer (no LLM credentials) — most relevant sentences:_", ""]
    for _, sent, tag in scored[:5]:
        lines.append(f"- {sent} [{tag}]")
    return "\n".join(lines)
