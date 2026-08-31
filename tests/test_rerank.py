"""Reranking behaviour.

The cross-encoder is an optimisation, never a dependency: retrieval must
still work when the model is missing or throws. And because neither ranking
dominates, the blend must actually consider both — a bug that silently
returned one or the other would look fine in aggregate and lose answers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend import rerank as rr


def _chunks(n):
    return [{"id": f"c{i}", "text": f"passage number {i}"} for i in range(n)]


def _use(monkeypatch, encoder):
    monkeypatch.setattr(rr, "_probed", True)
    monkeypatch.setattr(rr, "_encoder", encoder)


def test_missing_reranker_degrades_to_fusion_order(monkeypatch):
    _use(monkeypatch, None)
    out = rr.rerank_chunks("q", _chunks(10), 3)
    assert [c["id"] for c in out] == ["c0", "c1", "c2"]


def test_reranker_failure_degrades_to_fusion_order(monkeypatch):
    def boom(query, passages):
        raise RuntimeError("model exploded")

    _use(monkeypatch, boom)
    out = rr.rerank_chunks("q", _chunks(10), 3)
    assert [c["id"] for c in out] == ["c0", "c1", "c2"]


def test_blend_promotes_a_buried_answer(monkeypatch):
    """A chunk fusion ranked last must reach the top when the cross-encoder
    is confident it answers the question — this is the g16 case."""
    chunks = _chunks(20)

    def encoder(query, passages):
        # last passage is the answer; everything else is noise
        return [(10.0 if i == len(passages) - 1 else -5.0) for i in range(len(passages))]

    _use(monkeypatch, encoder)
    monkeypatch.setattr(config, "RERANK_BLEND_K", 5)
    out = rr.rerank_chunks("q", chunks, 7)
    assert "c19" in [c["id"] for c in out], "cross-encoder's answer was not promoted"


def test_blend_keeps_fusion_top_pick_when_reranker_disagrees(monkeypatch):
    """The cross-encoder must not be able to evict a chunk fusion was highly
    confident about — that regression cost hit@7 when rerank replaced fusion
    outright instead of blending."""
    chunks = _chunks(20)

    def encoder(query, passages):
        # cross-encoder hates the fusion favourite, loves a mid-pack chunk
        return [(-9.0 if i == 0 else 1.0) for i in range(len(passages))]

    _use(monkeypatch, encoder)
    monkeypatch.setattr(config, "RERANK_BLEND_K", 5)
    out = rr.rerank_chunks("q", chunks, 7)
    assert "c0" in [c["id"] for c in out], "fusion's top pick was evicted"


def test_rerank_annotates_positions_for_debugging(monkeypatch):
    chunks = _chunks(5)
    _use(monkeypatch, lambda q, p: [1.0, 5.0, 0.0, 0.0, 0.0])
    out = rr.rerank_chunks("q", chunks, 3)
    for c in out:
        assert "rerank_score" in c and "fusion_position" in c
        assert "rerank_position" in c and "final_position" in c
