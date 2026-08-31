"""Evaluation: faithfulness / hallucination (claim-level LLM judge) and the
Context Completeness Score (CCS) across the paper's six themes — with a
lexical-overlap fallback in offline mode.

Refinement: when the hallucination rate exceeds a threshold, the narrative is
regenerated with the unsupported claims listed so they get removed or
rephrased (the paper's Appendix E stage), then re-evaluated once.
"""
from __future__ import annotations

import re

from .llm import LLMUnavailable, OfflineLLM

_FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_FAITHFULNESS_SYSTEM = """You are an expert factual auditor for a \
retrieval-augmented generation system. You are given a generated narrative and \
the retrieved context it was written from. Your job is to decide, claim by \
claim, whether the narrative stayed inside its evidence.

Extract every distinct factual claim from the narrative — one fact per claim, \
as granular as possible — and mark each **supported** or **unsupported**.

A claim is SUPPORTED when the context states it explicitly, or implies it so \
directly that no additional knowledge is needed. Judge meaning over wording:
- The context is a machine transcript, so it lacks punctuation and casing and \
frequently garbles proper nouns ("michelle eligbe" for Mitchell Elegbe, \
"flatterwave" for Flutterwave). A claim that corrects an obvious transcription \
error of an entity already present is still supported.
- Paraphrase, summarising several sentences into one, and reordering are all \
fine. Do not require matching words.

A claim is UNSUPPORTED when it adds a fact the context does not contain — \
most often a date, figure, job title, or organisation that appears nowhere in \
the passages, or a number that differs from the one stated. Outside knowledge \
that happens to be true is still unsupported.

Do NOT extract as claims: section headings, citation tags like [S1], linking \
or framing phrases ("In conclusion", "According to the context"), and explicit \
absence statements such as "No relevant information in the retrieved context." \
Those carry no factual content and must not be scored.

Judge only the claims present in the narrative. Omitting information from the \
context is not a fault here."""

_CCS_SCHEMA = {
    "type": "object",
    "properties": {
        "theme_scores": {
            "type": "object",
            "properties": {
                "founder_name_company": {"type": "number"},
                "timeline_critical_events": {"type": "number"},
                "key_success_markers": {"type": "number"},
                "ecosystem_influences": {"type": "number"},
                "challenges_resilience": {"type": "number"},
                "impact_ecosystem": {"type": "number"},
            },
            "required": [
                "founder_name_company",
                "timeline_critical_events",
                "key_success_markers",
                "ecosystem_influences",
                "challenges_resilience",
                "impact_ecosystem",
            ],
            "additionalProperties": False,
        },
        "missing_themes": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["theme_scores", "missing_themes", "summary"],
    "additionalProperties": False,
}

_CCS_SYSTEM = """You evaluate how completely an AI-generated narrative covers \
an entrepreneurial journey. This is a coverage measure, not an accuracy \
measure: you are asking "which parts of the journey does this narrative let a \
reader see?", never whether the narrative is true.

Score each of six themes from 0.0 to 1.0 against this rubric:

- **founder_name_company** — 1.0 both the founder and their company are stated \
explicitly; 0.6-0.8 one of the two is vague or missing; 0.0-0.5 neither is clear.
- **timeline_critical_events** — 1.0 three or more milestones WITH dates; \
0.6-0.8 events present but dates thin or missing; 0.0-0.5 no usable sequence.
- **key_success_markers** — 1.0 at least one concrete, specific achievement \
(a funding amount, a user or revenue figure, profitability, a named award); \
0.6-0.8 achievements named but vague; 0.0-0.5 none.
- **ecosystem_influences** — 1.0 clear interaction with the ecosystem across \
networks, finance, or infrastructure/policy; 0.6-0.8 mentioned without depth; \
0.0-0.5 absent.
- **challenges_resilience** — 1.0 at least one specific challenge AND how it \
was addressed; 0.6-0.8 challenges named but the response is unexplained; \
0.0-0.5 none.
- **impact_ecosystem** — 1.0 concrete contribution beyond the company itself \
(jobs created, people trained, financial inclusion, mentorship, policy \
advocacy); 0.6-0.8 impact asserted without specifics; 0.0-0.5 none.

Important: a section that honestly reports "No relevant information in the \
retrieved context" scores low for coverage — that is correct behaviour by the \
narrative but still means the theme is not covered. Do not reward or punish \
the narrative for the honesty; just score the coverage.

List every theme scoring below 0.6 in missing_themes, using the theme keys \
above. Give a one-sentence summary naming the strongest and weakest themes."""


def evaluate_faithfulness(llm, narrative: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(c["text"] for c in chunks)
    if llm.mode == "claude":
        try:
            result = llm.complete_json(
                system=_FAITHFULNESS_SYSTEM,
                user=f"NARRATIVE:\n{narrative}\n\nRETRIEVED CONTEXT:\n{context}",
                schema=_FAITHFULNESS_SCHEMA,
                # A long narrative yields 70+ claims, each echoed back with a
                # verdict; 12k truncated the JSON mid-audit.
                max_tokens=32000,
            )
            claims = result.get("claims", [])
            total = len(claims)
            supported = sum(1 for c in claims if c.get("supported"))
            return {
                "engine": "claude",
                "total_claims": total,
                "supported_claims": supported,
                "faithfulness": round(supported / total, 3) if total else 1.0,
                "hallucination_rate": round((total - supported) / total, 3) if total else 0.0,
                "unsupported": [c["claim"] for c in claims if not c.get("supported")],
            }
        except LLMUnavailable:
            pass
    return _offline_faithfulness(narrative, context)


def evaluate_ccs(llm, narrative: str) -> dict:
    if llm.mode == "claude":
        try:
            result = llm.complete_json(
                system=_CCS_SYSTEM,
                user=f"Narrative to evaluate:\n\n{narrative}",
                schema=_CCS_SCHEMA,
                max_tokens=4000,
            )
            scores = result["theme_scores"]
            ccs = round(sum(scores.values()) / 6, 3)
            return {"engine": "claude", "ccs": ccs, **result}
        except LLMUnavailable:
            pass
    return _offline_ccs(narrative)


def refine_narrative(llm, narrative: str, unsupported: list[str],
                     founder: str, company: str, chunks: list[dict]) -> str | None:
    """Regenerate with unsupported claims flagged for removal. Claude-only —
    offline mode is already extractive, so nothing to refine."""
    if llm.mode != "claude" or not unsupported:
        return None
    from .generation import _NARRATIVE_SYSTEM, _format_context

    flagged = "\n".join(f"- {c}" for c in unsupported)
    try:
        return llm.complete(
            system=_NARRATIVE_SYSTEM
            + "\n\n## Revision task\n\nAn audit of your previous draft found the "
            "claims below to be unsupported by the passages. Produce a corrected "
            "narrative that keeps the same structure and everything that was "
            "well-grounded, and for each flagged claim either (a) rewrite it so "
            "it says only what the passages support, or (b) remove it entirely "
            "if nothing in the passages supports any version of it. Do not "
            "replace a removed claim with a hedge like 'it is unclear whether' — "
            "just leave it out. Do not introduce any new claims while revising. "
            "If removing claims empties a section, use the standard "
            '"No relevant information in the retrieved context." line.\n\n'
            "Unsupported claims:\n" + flagged,
            user=(
                f"Founder: {founder} ({company}).\n\nContext passages:\n\n"
                + _format_context(chunks)
            ),
            max_tokens=10000,
        )
    except LLMUnavailable:
        return None


# --- offline fallbacks ------------------------------------------------------

_SKIP_RE = re.compile(
    r"^(#|>|\s*$)|no relevant information in the retrieved context", re.IGNORECASE
)


def _offline_faithfulness(narrative: str, context: str) -> dict:
    claims = []
    for raw_line in narrative.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()
        if not line or _SKIP_RE.match(raw_line.strip()):
            continue
        line = re.sub(r"\[S\d+\]", "", line).strip()
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if len(sent.split()) >= 5:
                claims.append(sent)
    supported, unsupported = 0, []
    for claim in claims:
        if OfflineLLM.support_score(claim, context) >= 0.7:
            supported += 1
        else:
            unsupported.append(claim)
    total = len(claims)
    return {
        "engine": "offline (lexical overlap)",
        "total_claims": total,
        "supported_claims": supported,
        "faithfulness": round(supported / total, 3) if total else 1.0,
        "hallucination_rate": round((total - supported) / total, 3) if total else 0.0,
        "unsupported": unsupported[:10],
    }


_CCS_OFFLINE_KEYS = {
    "founder_name_company": ["founder", "company", "co-founded"],
    "timeline_critical_events": ["19", "20", "launched", "started", "became"],
    "key_success_markers": ["raised", "million", "customers", "profitable", "valuation"],
    "ecosystem_influences": ["mentor", "accelerator", "investor", "partner", "network"],
    "challenges_resilience": ["challenge", "overcame", "regulat", "struggle", "difficult"],
    "impact_ecosystem": ["jobs", "trained", "impact", "inclusion", "policy"],
}


def _offline_ccs(narrative: str) -> dict:
    lower = narrative.lower()
    scores = {}
    for theme, keys in _CCS_OFFLINE_KEYS.items():
        hits = sum(1 for k in keys if k in lower)
        scores[theme] = round(min(1.0, hits / 3), 2)
    missing = [t for t, s in scores.items() if s < 0.6]
    ccs = round(sum(scores.values()) / 6, 3)
    return {
        "engine": "offline (keyword heuristic)",
        "ccs": ccs,
        "theme_scores": scores,
        "missing_themes": missing,
        "summary": "Heuristic completeness estimate — enable Claude for a rubric-based CCS.",
    }
