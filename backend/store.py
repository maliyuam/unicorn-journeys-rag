"""Vector stores: MongoDB (Atlas Vector Search when available) and a local
persisted store so the system runs with zero infrastructure.

Both expose the same interface; `build_store()` picks one from config.
Alongside chunks + embeddings, each store keeps a *source registry* — one
record per ingested document (content hash, chunk count, timestamps) — which
makes ingestion idempotent and lets sources be listed and deleted.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import config


def _cosine_top(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def _replace_with_retry(src, dst, attempts: int = 5) -> None:
    """os.replace, retried briefly.

    On Windows the destination cannot be replaced while any other handle has
    it open (an antivirus scan, a sync client, or a concurrent reader), which
    surfaces as a transient PermissionError.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))


def _atomic_write_text(path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_with_retry(tmp, path)


class LocalStore:
    """Chunks + embeddings persisted under data/index/. No services needed.
    All file writes are atomic (tmp + rename) so a crash mid-save cannot
    corrupt the index."""

    name = "local (persisted numpy index)"

    def __init__(self):
        self.chunks: list[dict] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.sources: list[dict] = []
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        cpath = config.INDEX_DIR / "chunks.json"
        epath = config.INDEX_DIR / "embeddings.npy"
        spath = config.INDEX_DIR / "sources.json"
        if cpath.exists():
            self.chunks = json.loads(cpath.read_text(encoding="utf-8"))
        if epath.exists():
            self.matrix = np.load(epath)
        if spath.exists():
            self.sources = json.loads(spath.read_text(encoding="utf-8"))

    def _save(self) -> None:
        _atomic_write_text(
            config.INDEX_DIR / "chunks.json",
            json.dumps(self.chunks, ensure_ascii=False),
        )
        tmp = config.INDEX_DIR / "embeddings.npy.tmp"
        with open(tmp, "wb") as f:  # file handle: numpy won't append ".npy"
            np.save(f, self.matrix)
        _replace_with_retry(tmp, config.INDEX_DIR / "embeddings.npy")
        _atomic_write_text(
            config.INDEX_DIR / "sources.json",
            json.dumps(self.sources, ensure_ascii=False),
        )

    # -- interface ---------------------------------------------------------
    def replace_all(
        self, chunks: list[dict], embeddings: np.ndarray, sources: list[dict]
    ) -> None:
        self.chunks = chunks
        self.matrix = embeddings.astype(np.float32)
        self.sources = sources
        self._save()

    def all_chunks(self, founder: str | None = None) -> list[dict]:
        if founder:
            return [c for c in self.chunks if c["founder"] == founder]
        return list(self.chunks)

    def embedding_map(self, ids: list[str]) -> dict[str, np.ndarray]:
        wanted = set(ids)
        out: dict[str, np.ndarray] = {}
        for i, chunk in enumerate(self.chunks):
            if chunk["id"] in wanted and i < len(self.matrix):
                out[chunk["id"]] = self.matrix[i]
        return out

    def embedding_dim(self) -> int | None:
        if self.matrix.size == 0:
            return None
        return int(self.matrix.shape[1])

    def list_sources(self) -> list[dict]:
        return list(self.sources)

    def search(
        self, query_vec: np.ndarray, k: int, founder: str | None = None
    ) -> list[tuple[dict, float]]:
        if founder:
            idxs = [i for i, c in enumerate(self.chunks) if c["founder"] == founder]
            if not idxs:
                return []
            sub = self.matrix[idxs]
            hits = _cosine_top(query_vec, sub, k)
            return [(self.chunks[idxs[i]], s) for i, s in hits]
        hits = _cosine_top(query_vec, self.matrix, k)
        return [(self.chunks[i], s) for i, s in hits]

    def search_mode(self) -> str:
        return "in-memory cosine"

    def stats(self) -> dict:
        founders: dict[str, int] = {}
        for c in self.chunks:
            founders[c["founder"]] = founders.get(c["founder"], 0) + 1
        return {
            "chunks": len(self.chunks),
            "founders": founders,
            "sources": len(self.sources),
        }

    def clear(self) -> None:
        self.replace_all([], np.zeros((0, 0), dtype=np.float32), [])


class MongoStore:
    """MongoDB-backed store.

    With MongoDB Atlas and a vector-search index (config.MONGODB_VECTOR_INDEX)
    it uses native $vectorSearch; against a plain mongod it transparently
    falls back to client-side cosine over founder-filtered documents.
    The source registry lives in a sibling `<collection>_sources` collection.
    """

    def __init__(self):
        from pymongo import MongoClient

        self.client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=4000)
        self.client.admin.command("ping")  # fail fast if unreachable
        db = self.client[config.MONGODB_DB]
        self.col = db[config.MONGODB_COLLECTION]
        self.src_col = db[f"{config.MONGODB_COLLECTION}_sources"]
        self._atlas_search_ok: bool | None = None
        self._index_provisioned = False
        self.name = f"MongoDB ({config.MONGODB_DB}.{config.MONGODB_COLLECTION})"

    def replace_all(
        self, chunks: list[dict], embeddings: np.ndarray, sources: list[dict]
    ) -> None:
        self.col.delete_many({})
        if chunks:
            docs = []
            for chunk, vec in zip(chunks, embeddings, strict=True):
                doc = dict(chunk)
                doc["embedding"] = [float(x) for x in vec]
                docs.append(doc)
            self.col.insert_many(docs)
            self._maybe_create_vector_index(int(embeddings.shape[1]))
        self.src_col.delete_many({})
        if sources:
            self.src_col.insert_many([dict(s) for s in sources])

    def _maybe_create_vector_index(self, dims: int) -> None:
        """Auto-provision the Atlas vector-search index (no-op on plain
        mongod, where search index commands are unsupported)."""
        if self._index_provisioned:
            return
        self._index_provisioned = True
        try:
            from pymongo.operations import SearchIndexModel

            existing = {ix["name"] for ix in self.col.list_search_indexes()}
            if config.MONGODB_VECTOR_INDEX in existing:
                return
            self.col.create_search_index(
                SearchIndexModel(
                    name=config.MONGODB_VECTOR_INDEX,
                    type="vectorSearch",
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": dims,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "founder"},
                        ]
                    },
                )
            )
        except Exception:
            pass  # not Atlas — client-side cosine fallback covers retrieval

    def embedding_dim(self) -> int | None:
        doc = self.col.find_one({}, {"_id": 0, "embedding": 1})
        if not doc or "embedding" not in doc:
            return None
        return len(doc["embedding"])

    def search_mode(self) -> str:
        if self._atlas_search_ok is True:
            return "native $vectorSearch"
        if self._atlas_search_ok is False:
            return "client-side cosine"
        return "client-side cosine (probing for $vectorSearch)"

    def all_chunks(self, founder: str | None = None) -> list[dict]:
        query = {"founder": founder} if founder else {}
        return list(self.col.find(query, {"_id": 0, "embedding": 0}))

    def embedding_map(self, ids: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for doc in self.col.find({"id": {"$in": ids}}, {"_id": 0, "id": 1, "embedding": 1}):
            if "embedding" in doc:
                out[doc["id"]] = np.asarray(doc["embedding"], dtype=np.float32)
        return out

    def list_sources(self) -> list[dict]:
        return list(self.src_col.find({}, {"_id": 0}))

    def _atlas_search(
        self, query_vec: np.ndarray, k: int, founder: str | None
    ) -> list[tuple[dict, float]]:
        stage = {
            "$vectorSearch": {
                "index": config.MONGODB_VECTOR_INDEX,
                "path": "embedding",
                "queryVector": [float(x) for x in query_vec],
                "numCandidates": max(100, k * 15),
                "limit": k,
            }
        }
        if founder:
            stage["$vectorSearch"]["filter"] = {"founder": founder}
        pipeline = [
            stage,
            {"$project": {"_id": 0, "embedding": 0}},
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]
        results = []
        for doc in self.col.aggregate(pipeline):
            score = doc.pop("_score", 0.0)
            results.append((doc, float(score)))
        return results

    def search(
        self, query_vec: np.ndarray, k: int, founder: str | None = None
    ) -> list[tuple[dict, float]]:
        if self._atlas_search_ok is not False:
            try:
                hits = self._atlas_search(query_vec, k, founder)
                if hits:
                    self._atlas_search_ok = True
                    return hits
                if self._atlas_search_ok:
                    return hits  # proven index, genuinely no matches
                # empty + unproven: index likely still building — fall through
            except Exception as e:
                # permanent only when the deployment can't do $vectorSearch at
                # all (plain mongod); transient errors (index building, auth
                # blips) keep the native path eligible for the next query
                msg = str(e).lower()
                if "unrecognized pipeline stage" in msg or "no such command" in msg:
                    self._atlas_search_ok = False
        # client-side cosine fallback (works on any MongoDB)
        query = {"founder": founder} if founder else {}
        docs, vecs = [], []
        for doc in self.col.find(query, {"_id": 0}):
            vec = doc.pop("embedding", None)
            if vec is None:
                continue
            docs.append(doc)
            vecs.append(vec)
        if not docs:
            return []
        matrix = np.asarray(vecs, dtype=np.float32)
        hits = _cosine_top(query_vec, matrix, k)
        return [(docs[i], s) for i, s in hits]

    def stats(self) -> dict:
        founders = {
            d["_id"]: d["n"]
            for d in self.col.aggregate(
                [{"$group": {"_id": "$founder", "n": {"$sum": 1}}}]
            )
        }
        return {
            "chunks": self.col.count_documents({}),
            "founders": founders,
            "sources": self.src_col.count_documents({}),
        }

    def clear(self) -> None:
        self.col.delete_many({})
        self.src_col.delete_many({})


def build_store():
    mode = config.VECTOR_STORE
    if mode in ("auto", "mongodb") and config.MONGODB_URI:
        try:
            return MongoStore()
        except Exception:
            if mode == "mongodb":
                raise
    return LocalStore()
