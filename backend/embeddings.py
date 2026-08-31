"""Pluggable embedding backends.

Priority in "auto" mode: Voyage AI (hosted) > sentence-transformers (local
neural) > TF-IDF (pure scikit-learn, always available — keeps the demo fully
offline-capable). The paper used all-MiniLM-L6-v2 (384-d); the recommended
upgrade is BAAI/bge-small-en-v1.5 or voyage-3.5, both wired here.
"""
from __future__ import annotations

import pickle

import numpy as np

from . import config


class BaseEmbedder:
    name = "base"
    fixed_dim = True  # False for corpus-fitted embedders (TF-IDF)

    def fit_corpus(self, texts: list[str]) -> None:  # optional
        pass

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def save(self, path) -> None:
        pass

    def load(self, path) -> bool:
        return True


class TfidfEmbedder(BaseEmbedder):
    """Corpus-fitted TF-IDF vectors. Zero external dependencies beyond sklearn.

    The whole index is re-fit on every ingest, which is cheap at this scale
    and keeps query vectors consistent with the stored matrix.
    """

    name = "tf-idf (offline)"
    fixed_dim = False

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._make = lambda: TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            stop_words="english",
        )
        self.vectorizer = None

    def fit_corpus(self, texts: list[str]) -> None:
        self.vectorizer = self._make()
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            self.fit_corpus(texts)
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path) -> bool:
        try:
            with open(path, "rb") as f:
                self.vectorizer = pickle.load(f)
            return self.vectorizer is not None
        except (OSError, pickle.PickleError, EOFError):
            return False


class FastEmbedEmbedder(BaseEmbedder):
    """ONNX-based BGE embeddings (fastembed) — neural quality, no torch."""

    name = "fastembed (BAAI/bge-small-en-v1.5, 384-d)"

    def __init__(self):
        import os
        from pathlib import Path

        from fastembed import TextEmbedding

        cache = Path(os.environ.get("LOCALAPPDATA", ".")) / "UnicornRAG" / "fastembed_cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(cache))

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.asarray(list(self.model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class SentenceTransformerEmbedder(BaseEmbedder):
    name = f"sentence-transformers ({config.SENTENCE_TRANSFORMER_MODEL})"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


class VoyageEmbedder(BaseEmbedder):
    name = "voyage-3.5 (hosted)"

    def __init__(self):
        import voyageai

        self.client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    def embed(self, texts: list[str]) -> np.ndarray:
        result = self.client.embed(texts, model="voyage-3.5", input_type="document")
        mat = np.asarray(result.embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def embed_query(self, text: str) -> np.ndarray:
        result = self.client.embed([text], model="voyage-3.5", input_type="query")
        vec = np.asarray(result.embeddings[0], dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n else vec


def build_embedder() -> BaseEmbedder:
    backend = config.EMBEDDING_BACKEND
    if backend in ("auto", "voyage") and config.VOYAGE_API_KEY:
        try:
            return VoyageEmbedder()
        except Exception:
            if backend == "voyage":
                raise
    if backend in ("auto", "fastembed"):
        try:
            return FastEmbedEmbedder()
        except Exception:
            if backend == "fastembed":
                raise
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if backend == "sentence-transformers":
                raise
    return TfidfEmbedder()
