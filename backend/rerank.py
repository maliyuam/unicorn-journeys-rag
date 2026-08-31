"""Cross-encoder reranking.

Bi-encoders (the embedding model) encode the question and the passage
separately, so they can only match on overall topical similarity. That fails
precisely when the answer term is absent from the question — "which
ride-hailing company has Flutterwave worked with?" never surfaces "we've
worked with Uber", because nothing in the question is lexically or
semantically close to *Uber* specifically.

A cross-encoder reads the question and passage together and scores the pair
directly, so it can recognise that a passage answers a question rather than
merely sharing its topic. It is far too slow to run over a whole corpus, so
it reranks the top slice that fusion already found.

Loads lazily and degrades to a no-op if the model is unavailable, so
retrieval never hard-depends on it.
"""
from __future__ import annotations

import logging

from . import config

log = logging.getLogger("unicorn_rag.rerank")

_encoder = None
_probed = False


def get_reranker():
    """Return a rerank(query, passages) -> list[float] callable, or None."""
    global _encoder, _probed
    if _probed:
        return _encoder
    _probed = True
    if not config.RERANK_ENABLED:
        return None
    try:
        import os
        from pathlib import Path

        from fastembed.rerank.cross_encoder import TextCrossEncoder

        cache = Path(os.environ.get("LOCALAPPDATA", ".")) / "UnicornRAG" / "fastembed_cache"
        cache.mkdir(parents=True, exist_ok=True)
        model = TextCrossEncoder(config.RERANK_MODEL, cache_dir=str(cache))

        def rerank(query: str, passages: list[str]) -> list[float]:
            return list(model.rerank(query, passages))

        _encoder = rerank
        log.info("reranker loaded: %s", config.RERANK_MODEL)
    except Exception as e:  # noqa: BLE001 — reranking is an optimisation
        log.warning("reranker unavailable (%s); ranking on fusion alone", str(e)[:120])
        _encoder = None
    return _encoder


def rerank_chunks(question: str, chunks: list[dict], k: int) -> list[dict]:
    """Blend the cross-encoder's ranking with the fusion ranking it was given.

    Measured on the golden set, neither ranking dominates. The cross-encoder
    is far better at putting the answer first (hit@3 0.529 -> 0.824, MRR
    0.392 -> 0.686) but its own top-k drops chunks that fusion had ranked
    well, costing hit@7. Fusing the two rank lists keeps fusion's confident
    picks and still lets the cross-encoder promote a passage that actually
    answers the question.

    Returns the input order untouched when no reranker is available.
    """
    if len(chunks) <= 1:
        return chunks[:k]
    encoder = get_reranker()
    if encoder is None:
        return chunks[:k]
    try:
        scores = encoder(question, [c["text"] for c in chunks])
    except Exception as e:  # noqa: BLE001
        log.warning("rerank failed (%s); keeping fusion order", str(e)[:120])
        return chunks[:k]

    rerank_order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    rerank_rank = {i: r for r, i in enumerate(rerank_order)}
    kk = config.RERANK_BLEND_K

    def blended(i: int) -> float:
        # i is already the fusion rank, since `chunks` arrives fusion-ordered
        return 1.0 / (kk + i) + 1.0 / (kk + rerank_rank[i])

    final = sorted(range(len(chunks)), key=lambda i: -blended(i))
    out = []
    for position, i in enumerate(final[:k]):
        chunk = dict(chunks[i])
        chunk["rerank_score"] = round(float(scores[i]), 4)
        chunk["fusion_position"] = i
        chunk["rerank_position"] = rerank_rank[i]
        chunk["final_position"] = position
        out.append(chunk)
    return out
