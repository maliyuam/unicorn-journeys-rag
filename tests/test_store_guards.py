"""Guards against the failure that cost a live corpus.

A fastembed load failure left TF-IDF in place. TF-IDF vectors are ~20,000-d and
~99.5% zeros; stored densely in BSON that is ~289 KB per chunk. The startup
dimension-heal then re-embedded ~2,000 chunks into MongoDB, producing 548 MB,
which exhausted an Atlas free tier and blocked every write on the cluster. The
oversized vectors also stopped matching a $vectorSearch index provisioned for
384 dimensions, so retrieval silently degraded to client-side cosine.

Three separate things had to go wrong together, and none of them said anything.
These tests pin the two that are now loud.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.store import StoreError


class _FakeCollection:
    """Records whether anything was actually written."""

    def __init__(self):
        self.deleted = False
        self.inserted = []

    def delete_many(self, *_args, **_kwargs):
        self.deleted = True

    def insert_many(self, docs, *_args, **_kwargs):
        self.inserted.extend(docs)


def _mongo_store_with_fakes():
    """A MongoStore whose collections are fakes, so nothing touches a server."""
    from backend.store import MongoStore

    store = MongoStore.__new__(MongoStore)  # bypass __init__ and its connection
    store.col = _FakeCollection()
    store.src_col = _FakeCollection()
    store._atlas_search_ok = None
    store._index_provisioned = True
    store.name = "MongoDB (test.chunks)"
    return store


def test_mongo_refuses_oversized_embeddings_before_deleting_anything():
    store = _mongo_store_with_fakes()
    chunks = [{"id": f"c{i}", "text": "x"} for i in range(3)]
    tfidf_like = np.zeros((3, 20000), dtype=np.float32)

    with pytest.raises(StoreError) as excinfo:
        store.replace_all(chunks, tfidf_like, [])

    # The guard must fire BEFORE delete_many, or it does not protect anything.
    assert store.col.deleted is False, "corpus was deleted despite the refusal"
    assert store.col.inserted == []
    message = str(excinfo.value)
    assert "20000" in message
    assert "TF-IDF" in message  # names the actual cause
    assert "VECTOR_STORE=local" in message  # and a way out


def test_mongo_accepts_normal_embeddings():
    store = _mongo_store_with_fakes()
    chunks = [{"id": f"c{i}", "text": "x"} for i in range(3)]
    store.replace_all(chunks, np.zeros((3, 384), dtype=np.float32), [])
    assert store.col.deleted is True
    assert len(store.col.inserted) == 3


def test_empty_corpus_is_not_blocked():
    """Clearing the index must still work; there are no vectors to object to."""
    store = _mongo_store_with_fakes()
    store.replace_all([], np.zeros((0, 0), dtype=np.float32), [])
    assert store.col.deleted is True


def test_heal_refuses_to_re_embed_a_mongo_corpus_into_a_huge_space(monkeypatch, caplog):
    """The heal path is what actually rewrote the corpus, so it gets its own guard."""
    from backend.engine import Engine

    engine = Engine.__new__(Engine)

    class _Store:
        name = "MongoDB (test.chunks)"

        def embedding_dim(self):
            return 384

        def all_chunks(self, founder=None):
            raise AssertionError("heal must not read the corpus to rewrite it")

    class _TfidfLikeEmbedder:
        fixed_dim = False

        def embed_query(self, _text):
            return np.zeros(20000, dtype=np.float32)

    import threading

    engine.store = _Store()
    engine.embedder = _TfidfLikeEmbedder()
    engine._lock = threading.RLock()

    with caplog.at_level("ERROR"):
        engine._heal_dimension_mismatch()

    assert any("NOT re-embedding" in r.message for r in caplog.records), (
        "the refusal must be logged loudly — silence is how this happened"
    )


def test_limit_is_configurable_but_sane():
    assert config.MAX_MONGO_EMBEDDING_DIM >= 384
    assert config.MAX_MONGO_EMBEDDING_DIM <= 4096  # Atlas' own vector-search cap
