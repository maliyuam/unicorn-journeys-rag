"""LLM attribution audit: is this source actually about the founder we filed
it under?

The relevance filter judges a candidate from its title, channel and
description — everything available *before* fetching. Sometimes the transcript
then turns out to be someone else's: a panel the founder was billed for but
never spoke on, or a different person with the same surname. Two such cases
reached this corpus.

String matching cannot police this, because speech recognition mangles exactly
the words it would rely on. Observed here: "Tosin Eniolorunda" transcribed as
"Tosi and La", "Moniepoint" as "moneyo Inc money point", "Halan" as "Helen".
Companies also rename — Moniepoint was TeamApt — so a current name legitimately
appears nowhere in an older recording. An LLM reads through all of that.

Excerpts are sampled rather than sending whole transcripts: introductions are
where speakers name themselves, so the opening matters most, plus passages
densest in name-like tokens and one sample from the middle.
"""
from __future__ import annotations

import logging
import re

from .llm import LLMUnavailable

log = logging.getLogger("unicorn_rag.attribution")

MAX_EXCERPT_CHARS = 6000
OPENING_CHUNKS = 2
NAME_DENSE_CHUNKS = 3

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["speaker", "discussed", "absent"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "identified_as": {"type": "string"},
        "evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "identified_as", "evidence", "reason"],
    "additionalProperties": False,
}

_SYSTEM = """You audit whether a transcript really concerns the person it was \
filed under, for a research corpus of founder biographies. A wrong attribution \
puts a stranger's career into a founder's record, with citations that look \
entirely legitimate, so this judgement matters more than it may appear.

You will be given a founder's name, their company, and excerpts from a \
transcript that was attributed to them. Decide which is true:

- **speaker** — the founder is one of the speakers. Interviews, keynotes, and \
podcast appearances where they talk. Look for self-introductions, a host \
addressing them, or first-person accounts matching their known company.
- **discussed** — the founder does not speak, but the transcript is \
substantively about them or the company they built.
- **absent** — the founder neither speaks nor is a subject. This includes a \
different person who happens to share a surname, and panels or roundups where \
they were billed but never appear.

Critical: these are MACHINE transcripts. Proper nouns are frequently garbled, \
and this is the single most common reason to misjudge a source:
- "Tosin Eniolorunda" has appeared as "Tosi and La"
- "Moniepoint" as "moneyo Inc money point"
- "Halan" as "Helen"
- "Mitchell Elegbe" as "michelle eligbe", "Flutterwave" as "flatterwave"
Treat a plausible phonetic match as the real name. Judge by who the speaker \
evidently *is* — the company they describe running, the events they claim — \
not by exact spelling.

Companies also rename. Moniepoint was TeamApt; a current name can be absent \
from an older recording while the transcript is unmistakably about that \
company.

Set identified_as to who the transcript actually appears to be about or by — \
the founder's name if it matches, otherwise the other person or topic. Quote a \
short phrase in evidence. Only use high confidence when the excerpts settle \
it; if the excerpts are too thin to tell, say low and explain what is missing. \
Absent with high confidence is a deletion recommendation, so hold it to that \
standard."""


def _name_tokens(founder: str, company: str) -> list[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", f"{founder} {company}")]
    return [t for t in tokens if len(t) > 3]


def sample_excerpts(chunks: list[dict], founder: str, company: str) -> str:
    """Pick the passages most likely to reveal who is speaking."""
    ordered = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
    if not ordered:
        return ""
    picked: list[dict] = list(ordered[:OPENING_CHUNKS])
    picked_ids = {id(c) for c in picked}

    needles = _name_tokens(founder, company)
    if needles:
        scored = []
        for c in ordered[OPENING_CHUNKS:]:
            body = c["text"].lower()
            score = sum(body.count(n) for n in needles)
            if score:
                scored.append((score, c))
        scored.sort(key=lambda t: -t[0])
        for _, c in scored[:NAME_DENSE_CHUNKS]:
            if id(c) not in picked_ids:
                picked.append(c)
                picked_ids.add(id(c))

    middle = ordered[len(ordered) // 2]
    if id(middle) not in picked_ids:
        picked.append(middle)

    parts, total = [], 0
    for c in picked:
        text = c["text"]
        label = f"[chunk {c.get('chunk_index', '?')}]"
        room = MAX_EXCERPT_CHARS - total
        if room <= 200:
            break
        parts.append(f"{label} {text[:room]}")
        total += min(len(text), room)
    return "\n\n".join(parts)


def audit_source(llm, source: dict, chunks: list[dict]) -> dict:
    """Judge one source. Raises LLMUnavailable when there is no LLM."""
    if llm.mode != "claude":
        raise LLMUnavailable(
            "attribution auditing needs an LLM — string matching cannot see "
            "through speech-recognition errors"
        )
    excerpts = sample_excerpts(chunks, source["founder"], source.get("company", ""))
    if not excerpts:
        return {
            "source_id": source["source_id"],
            "founder": source["founder"],
            "source_title": source["source_title"],
            "verdict": "absent",
            "confidence": "low",
            "identified_as": "",
            "evidence": "",
            "reason": "no transcript text stored for this source",
        }
    user = (
        f"Founder filed under: {source['founder']}\n"
        f"Company: {source.get('company') or 'unknown'}\n"
        f"Source title: {source['source_title']}\n"
        f"Source type: {source.get('source_type', 'unknown')}\n\n"
        f"Transcript excerpts:\n\n{excerpts}"
    )
    result = llm.complete_json(
        system=_SYSTEM, user=user, schema=_VERDICT_SCHEMA, max_tokens=4000
    )
    return {
        "source_id": source["source_id"],
        "founder": source["founder"],
        "source_title": source["source_title"],
        "chunks": source.get("chunks", len(chunks)),
        **result,
    }


def audit_corpus(engine, llm, founders: list[str] | None = None, report=None) -> dict:
    """Audit every source (optionally only certain founders)."""
    by_source: dict[str, list[dict]] = {}
    for c in engine.store.all_chunks():
        by_source.setdefault(c["source_id"], []).append(c)

    sources = engine.list_sources()
    if founders:
        wanted = {f.lower() for f in founders}
        sources = [s for s in sources if s["founder"].lower() in wanted]

    results, failed = [], []
    for i, source in enumerate(sources):
        if report:
            report(
                i / max(len(sources), 1),
                f"[{i + 1}/{len(sources)}] {source['founder']}: "
                f"{source['source_title'][:40]}",
            )
        try:
            results.append(audit_source(llm, source, by_source.get(source["source_id"], [])))
        except LLMUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 — one bad source must not stop the audit
            failed.append({"source_id": source["source_id"], "error": str(e)[:200]})
            log.warning("audit failed for %s: %s", source["source_id"], str(e)[:120])

    counts = {"speaker": 0, "discussed": 0, "absent": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    if report:
        report(1.0, "audit complete")
    return {
        "audited": len(results),
        "counts": counts,
        "results": results,
        "failed": failed,
        "removable": [
            r for r in results
            if r["verdict"] == "absent" and r["confidence"] in ("high", "medium")
        ],
    }
