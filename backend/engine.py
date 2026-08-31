"""Engine: owns the embedder + vector store and orchestrates the pipeline
(ingest -> expand -> hybrid retrieve -> generate -> evaluate -> refine).

Ingestion is production-hardened:
  * idempotent — a source whose content hash is unchanged is skipped
  * upsert semantics — re-ingesting a changed source replaces its chunks
  * incremental — fixed-dimension embedders only embed new/changed chunks
    (the TF-IDF fallback refits, since its vector space is corpus-defined)
  * per-document error isolation — one bad document never fails a batch
  * thread-safe — a re-entrant lock serializes all index mutations
"""
from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

import numpy as np

from . import config
from .chunking import chunk_transcript
from .embeddings import build_embedder
from .evaluation import evaluate_ccs, evaluate_faithfulness, refine_narrative
from .generation import citation_map, generate_answer, generate_narrative
from .grounding import assess_support
from .ingest import IngestError, validate_doc
from .llm import get_llm
from .rerank import get_reranker
from .retrieval import expand_query, hybrid_search
from .store import build_store

log = logging.getLogger("unicorn_rag")


class Engine:
    def __init__(self):
        self.embedder = build_embedder()
        self.store = build_store()
        self._lock = threading.RLock()
        if not self.embedder.fixed_dim:
            self.embedder.load(config.INDEX_DIR / "vectorizer.pkl")
        self._heal_dimension_mismatch()

    def _heal_dimension_mismatch(self) -> None:
        """If the embedding backend changed since the index was built (e.g.
        TF-IDF -> fastembed), stored vectors have the wrong dimensionality.
        Detect it and transparently re-embed the whole corpus."""
        with self._lock:
            stored_dim = self.store.embedding_dim()
            if stored_dim is None:
                return
            try:
                current_dim = int(self.embedder.embed_query("dimension probe").shape[0])
            except Exception:
                return  # TF-IDF with no fitted vectorizer: rebuilt on next ingest
            if stored_dim == current_dim:
                return
            log.info(
                "embedding backend changed (%d-d stored vs %d-d current) — re-embedding corpus",
                stored_dim, current_dim,
            )
            chunks = self.store.all_chunks()
            self._rebuild_embeddings(chunks)

    def _rebuild_embeddings(self, chunks: list[dict]) -> None:
        texts = [c["text"] for c in chunks]
        if not texts:
            return
        if not self.embedder.fixed_dim:
            self.embedder.fit_corpus(texts)
            self.embedder.save(config.INDEX_DIR / "vectorizer.pkl")
        self.store.replace_all(chunks, self.embedder.embed(texts), self.store.list_sources())

    # -- ingest ------------------------------------------------------------
    def ingest_documents(self, docs: list[dict], progress=None) -> dict:
        """Validate, upsert and index a batch of documents.

        Returns {ingested, skipped_unchanged, failed, chunks, founders, sources}.
        Bad documents land in `failed` with their reason; the rest proceed.
        """
        if len(docs) > config.MAX_BATCH_DOCS:
            raise IngestError(
                f"batch of {len(docs)} exceeds MAX_BATCH_DOCS={config.MAX_BATCH_DOCS}"
            )

        def report(fraction: float, message: str) -> None:
            if progress:
                progress(fraction, message)

        report(0.02, "validating documents")
        valid: list[dict] = []
        failed: list[dict] = []
        for i, doc in enumerate(docs):
            try:
                valid.append(validate_doc(doc))
            except IngestError as e:
                failed.append(
                    {"source_title": doc.get("source_title") or f"document {i + 1}",
                     "error": str(e)}
                )

        with self._lock:
            registry = {s["source_id"]: s for s in self.store.list_sources()}
            skipped: list[str] = []
            to_ingest: list[dict] = []
            for doc in valid:
                prior = registry.get(doc["source_id"])
                if prior and prior.get("content_hash") == doc["content_hash"]:
                    skipped.append(doc["source_id"])
                else:
                    to_ingest.append(doc)

            if not to_ingest:
                report(1.0, "nothing new to index")
                return {
                    "ingested": 0,
                    "skipped_unchanged": len(skipped),
                    "failed": failed,
                    **self.store.stats(),
                }

            report(0.1, f"chunking {len(to_ingest)} document(s)")
            replaced_ids = {d["source_id"] for d in to_ingest}
            new_chunks: list[dict] = []
            now = datetime.now(UTC).isoformat()
            for doc in to_ingest:
                doc_chunks = [
                    c.to_dict()
                    for c in chunk_transcript(
                        text=doc["text"],
                        founder=doc["founder"],
                        company=doc["company"],
                        source_id=doc["source_id"],
                        source_title=doc["source_title"],
                        source_type=doc["source_type"],
                    )
                ]
                new_chunks.extend(doc_chunks)
                registry[doc["source_id"]] = {
                    "source_id": doc["source_id"],
                    "founder": doc["founder"],
                    "company": doc["company"],
                    "source_title": doc["source_title"],
                    "source_type": doc["source_type"],
                    "content_hash": doc["content_hash"],
                    "chunks": len(doc_chunks),
                    "ingested_at": now,
                }

            kept = [
                c for c in self.store.all_chunks()
                if c["source_id"] not in replaced_ids
            ]
            all_chunks = kept + new_chunks
            report(0.35, f"embedding {len(new_chunks)} new chunk(s)")
            matrix = self._build_matrix(all_chunks, kept, new_chunks, report)
            report(0.9, "persisting index")
            self.store.replace_all(all_chunks, matrix, list(registry.values()))

        log.info(
            "ingest: %d document(s) indexed, %d skipped (unchanged), %d failed",
            len(to_ingest), len(skipped), len(failed),
        )
        report(1.0, "done")
        return {
            "ingested": len(to_ingest),
            "skipped_unchanged": len(skipped),
            "failed": failed,
            **self.store.stats(),
        }

    def _build_matrix(self, all_chunks, kept, new_chunks, report) -> np.ndarray:
        texts = [c["text"] for c in all_chunks]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if not self.embedder.fixed_dim:
            # corpus-defined vector space (TF-IDF): refit + embed everything
            self.embedder.fit_corpus(texts)
            self.embedder.save(config.INDEX_DIR / "vectorizer.pkl")
            return self.embedder.embed(texts)
        # fixed-dimension embedder: reuse stored vectors, embed only the new
        reuse = self.store.embedding_map([c["id"] for c in kept]) if kept else {}
        new_texts = [c["text"] for c in new_chunks]
        new_vecs = self.embedder.embed(new_texts) if new_texts else None
        new_map = {c["id"]: new_vecs[i] for i, c in enumerate(new_chunks)} if new_texts else {}
        rows, missing = [], []
        for chunk in all_chunks:
            vec = reuse.get(chunk["id"])
            if vec is None:
                vec = new_map.get(chunk["id"])
            if vec is None:
                missing.append(chunk)
                rows.append(None)
            else:
                rows.append(vec)
        if missing:  # stored vectors lost (e.g. store swapped) — re-embed
            report(0.6, f"re-embedding {len(missing)} chunk(s) with no stored vector")
            fixed = self.embedder.embed([c["text"] for c in missing])
            it = iter(fixed)
            rows = [r if r is not None else next(it) for r in rows]
        return np.vstack(rows).astype(np.float32)

    def delete_source(self, source_id: str) -> dict:
        with self._lock:
            registry = [
                s for s in self.store.list_sources() if s["source_id"] != source_id
            ]
            remaining = [
                c for c in self.store.all_chunks() if c["source_id"] != source_id
            ]
            if not remaining:
                self.store.replace_all([], np.zeros((0, 0), dtype=np.float32), registry)
            elif self.embedder.fixed_dim:
                matrix = self._build_matrix(remaining, remaining, [], lambda *a: None)
                self.store.replace_all(remaining, matrix, registry)
            else:
                texts = [c["text"] for c in remaining]
                self.embedder.fit_corpus(texts)
                self.embedder.save(config.INDEX_DIR / "vectorizer.pkl")
                self.store.replace_all(remaining, self.embedder.embed(texts), registry)
        log.info("deleted source %s", source_id)
        return self.store.stats()

    def list_sources(self) -> list[dict]:
        return sorted(
            self.store.list_sources(), key=lambda s: s.get("ingested_at", ""), reverse=True
        )

    def clear(self) -> dict:
        with self._lock:
            self.store.clear()
        return self.store.stats()

    # -- query -------------------------------------------------------------
    def ask(self, question: str, founder: str | None = None, k: int | None = None) -> dict:
        llm = get_llm()
        subqueries = expand_query(llm, question, founder)
        chunks = hybrid_search(
            self.store, self.embedder, question, subqueries, k=k, founder=founder
        )

        # Retrieval ranks; it does not judge. Before answering, check that the
        # passages actually carry what was asked — otherwise the pipeline
        # writes a fluent answer from the least-unrelated paragraphs and cites
        # them, which is indistinguishable from a sourced fact.
        company = chunks[0].get("company", "") if chunks else ""
        support = assess_support(
            llm, question, chunks, founder or "", company,
            min_similarity=getattr(self.embedder, "abstain_min_similarity", None),
        )
        if not support.supported:
            return {
                "question": question,
                "subqueries": subqueries,
                # No chunks and no citations. Returning the passages "for
                # context" is how an unsupported answer gets citations anyway:
                # the UI renders them as sources and the distinction is lost.
                "chunks": [],
                "citations": {},
                "answer": (
                    "**This corpus cannot answer that question.**\n\n"
                    f"{support.reason.rstrip('.')}.\n\n"
                    "No sources are cited because none of the retrieved passages "
                    "support an answer. Ingest material that covers this topic, or "
                    "ask something the corpus does cover."
                ),
                "abstained": True,
                "support": support.as_dict(),
                "llm_mode": llm.mode,
            }

        return {
            "question": question,
            "subqueries": subqueries,
            "chunks": chunks,
            "citations": citation_map(chunks),
            "answer": generate_answer(llm, question, chunks),
            "abstained": False,
            "support": support.as_dict(),
            "llm_mode": llm.mode,
        }

    # -- full narrative pipeline --------------------------------------------
    def narrative(self, founder: str, k: int | None = None) -> dict:
        llm = get_llm()
        chunks_for_founder = self.store.all_chunks(founder=founder)
        if not chunks_for_founder:
            return {"error": f"No indexed content for founder '{founder}'."}
        company = chunks_for_founder[0].get("company", "")

        main_query = (
            f"Reconstruct the complete entrepreneurial journey of {founder}"
            + (f" of {company}" if company else "")
        )
        subqueries = expand_query(llm, main_query, founder)
        k_eff = k or max(config.DEFAULT_TOP_K, 8)
        chunks = hybrid_search(
            self.store, self.embedder, main_query, subqueries, k=k_eff, founder=founder
        )

        narrative_text = generate_narrative(llm, founder, company, chunks)
        faith = evaluate_faithfulness(llm, narrative_text, chunks)

        refined_text = None
        if faith["hallucination_rate"] > config.REFINE_HALLUCINATION_THRESHOLD:
            refined_text = refine_narrative(
                llm, narrative_text, faith["unsupported"], founder, company, chunks
            )
            if refined_text:
                faith_refined = evaluate_faithfulness(llm, refined_text, chunks)
                if faith_refined["hallucination_rate"] <= faith["hallucination_rate"]:
                    faith = faith_refined
                else:
                    refined_text = None

        final_text = refined_text or narrative_text
        ccs = evaluate_ccs(llm, final_text)

        return {
            "founder": founder,
            "company": company,
            "subqueries": subqueries,
            "chunks": chunks,
            "citations": citation_map(chunks),
            "narrative": final_text,
            "first_draft": narrative_text if refined_text else None,
            "refined": refined_text is not None,
            "faithfulness": faith,
            "ccs": ccs,
            "llm_mode": llm.mode,
        }

    # -- status --------------------------------------------------------------
    def status(self) -> dict:
        llm = get_llm()
        return {
            "llm": {"mode": llm.mode, "label": llm.label},
            "embedder": self.embedder.name,
            "reranker": (
                config.RERANK_MODEL if get_reranker() is not None else "none (fusion only)"
            ),
            "store": f"{self.store.name} · {self.store.search_mode()}",
            "corpus": self.store.stats(),
            "defaults": {
                "top_k": config.DEFAULT_TOP_K,
                "chunk_target_words": config.CHUNK_TARGET_WORDS,
                "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
                "max_file_mb": config.MAX_FILE_BYTES // (1024 * 1024),
                "max_upload_files": config.MAX_UPLOAD_FILES,
            },
        }


_engine: Engine | None = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
    return _engine


