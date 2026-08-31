"""Hybrid retrieval: multi-query expansion -> (vector + BM25) -> RRF fusion.

Upgrades over the paper's pipeline (pure cosine over one embedding space):
  * BM25 keyword scoring runs alongside dense vectors and the two rankings
    are merged with reciprocal-rank fusion, which recovers exact names,
    figures, and dates that dense embeddings blur.
  * Multi-query expansion is schema-constrained (structured outputs), with a
    deterministic template fallback in offline mode.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from . import config
from .llm import LLMUnavailable
from .rerank import rerank_chunks

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / len(docs)) if docs else 0.0
        self.doc_freqs = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def top(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scores = self.scores(query_tokens)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


# --- multi-query expansion -------------------------------------------------

THEMES = [
    ("background", "early life, education, and entrepreneurial exposure"),
    ("inception", "how and when the venture(s) started, industries, co-founders"),
    ("funding", "funding rounds, investors, amounts, and valuations"),
    ("challenges", "regulatory, financial, or operational hurdles and how they were overcome"),
    ("milestones", "product launches, market expansions, acquisitions, profitability"),
    ("ecosystem", "mentorship, partnerships, policy influence, jobs created, broader impact"),
]

_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "subqueries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["subqueries"],
    "additionalProperties": False,
}

_EXPANSION_SYSTEM = """You are a query decomposition agent for a \
retrieval-augmented system whose corpus is machine transcripts of interviews, \
podcasts and keynotes with African startup founders.

Decompose the main query into 4 to 8 atomic sub-queries that will be run \
against a hybrid keyword + semantic index.

Each sub-query must:
- target ONE retrievable fact, never a compound request;
- stand alone, without pronouns or references to the other sub-queries;
- reuse the concrete anchors from the main query verbatim — personal names, \
company names, places, years — because exact terms drive the keyword half of \
the index;
- prefer the vocabulary a speaker would actually use out loud ("we raised", \
"we launched", "back in 2012") over formal or academic phrasing, since the \
corpus is transcribed speech;
- include, where the fact is quantitative, a phrasing that anticipates the \
figure ("how much did X raise", "what valuation", "how many customers").

When the question asks *which* company, person, investor or product — a \
category whose members you know — name the plausible candidates explicitly in \
separate sub-queries. A transcript says "we've worked with Uber", never "we \
worked with a ride-hailing company", so a sub-query containing the category \
alone cannot match it. Ask "did the company work with Uber?", "did the company \
work with Bolt?" and so on. Guessing costs nothing: a wrong candidate simply \
retrieves nothing, while a right one finds the passage that answers the \
question. Never state a guess as fact — these are search probes, not claims.

Cover distinct angles rather than restating one idea. If the main query is \
already atomic, still produce variants that differ in wording, because the \
transcripts may phrase the same fact very differently.

Do not answer the query. Do not add commentary."""


def expand_query(llm, question: str, founder: str | None = None) -> list[str]:
    context = f" The question concerns the founder {founder}." if founder else ""
    if llm.mode == "claude":
        try:
            result = llm.complete_json(
                system=_EXPANSION_SYSTEM,
                user=f"Main query: {question}.{context}",
                schema=_EXPANSION_SCHEMA,
                max_tokens=3000,
                effort="low",
            )
            subs = [s.strip() for s in result.get("subqueries", []) if s.strip()]
            if subs:
                # Keep the original question as the first retrieval query.
                # Measured on the golden set: without it, LLM-generated
                # subqueries drift from the user's exact wording and BM25
                # loses the rare-term anchors that find single-chunk facts
                # (hit@7 fell from 0.882 to 0.824).
                return [question] + subs[:7]
        except (LLMUnavailable, Exception):
            pass
    # deterministic fallback: the question itself + themed variants
    name = founder or ""
    subs = [question]
    lowered = question.lower()
    for _, desc in THEMES:
        head = desc.split(",")[0]
        if head not in lowered:
            subs.append(f"{name} {head} {question}".strip())
    return subs[:6]


# --- fusion ----------------------------------------------------------------

def _rrf(rank: int) -> float:
    return 1.0 / (config.RRF_K + rank + 1)


def build_bm25(corpus: list[dict]) -> BM25:
    return BM25([tokenize(c["text"]) for c in corpus])


def hybrid_search(
    store,
    embedder,
    question: str,
    subqueries: list[str],
    k: int = None,
    founder: str | None = None,
    corpus: list[dict] | None = None,
    bm25: BM25 | None = None,
    rerank: bool = True,
) -> list[dict]:
    """Run every subquery through dense + BM25 retrieval and fuse with RRF.

    `corpus`/`bm25` let a caller that issues many queries against the same
    slice (the evaluation sweep) load and index it once instead of per query —
    with a remote store that is the difference between seconds and minutes.
    """
    k = k or config.DEFAULT_TOP_K
    if corpus is None:
        corpus = store.all_chunks(founder=founder)
    if not corpus:
        return []
    if bm25 is None:
        bm25 = build_bm25(corpus)
    id_to_chunk = {c["id"]: c for c in corpus}

    fused: dict[str, float] = {}
    # Candidate depth per subquery before fusion. Deeper is NOT better here:
    # measured on the golden set, 4x/min-30 dropped hit@7 from 0.882 to 0.824,
    # because a wide pool lets chunks that rank mediocrely across many
    # subqueries outscore a chunk that is top-1 for one. Keep it tight.
    per_query_k = max(k * config.FUSION_POOL_MULTIPLIER, config.FUSION_POOL_MIN)
    for sq in subqueries:
        # dense
        qvec = embedder.embed_query(sq)
        for rank, (chunk, _score) in enumerate(
            store.search(qvec, per_query_k, founder=founder)
        ):
            fused[chunk["id"]] = fused.get(chunk["id"], 0.0) + _rrf(rank)
            id_to_chunk.setdefault(chunk["id"], chunk)
        # sparse
        for rank, (idx, _score) in enumerate(bm25.top(tokenize(sq), per_query_k)):
            cid = corpus[idx]["id"]
            fused[cid] = fused.get(cid, 0.0) + _rrf(rank)

    # Keep a deeper slice than k when a reranker can re-order it: fusion is
    # good at surfacing candidates and mediocre at ordering them, and the
    # answer-bearing chunk is often just below the cutoff.
    depth = max(k, config.RERANK_POOL) if rerank else k
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:depth]
    results = []
    for cid, score in ranked:
        chunk = dict(id_to_chunk[cid])
        chunk["fusion_score"] = round(score, 4)
        results.append(chunk)
    if rerank:
        return rerank_chunks(question, results, k)
    return results[:k]
