"""Re-chunk the indexed corpus with the current chunking rules.

Needed after a chunking change — most importantly the word cap that stops
unpunctuated ASR transcripts (YouTube auto-captions) collapsing into one
giant chunk. Source text is reconstructed from the stored chunks by removing
the sentence-overlap between consecutive chunks, so no re-fetching is needed.

Usage:
    python scripts/rechunk.py --dry-run
    python scripts/rechunk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.chunking import chunk_transcript  # noqa: E402
from backend.engine import get_engine  # noqa: E402


def reconstruct(chunks: list[dict]) -> str:
    """Join chunks in order, dropping the repeated overlap between them."""
    ordered = sorted(chunks, key=lambda c: c["chunk_index"])
    words: list[str] = ordered[0]["text"].split()
    for nxt in ordered[1:]:
        nxt_words = nxt["text"].split()
        # longest suffix of the accumulated text that prefixes the next chunk
        max_probe = min(len(words), len(nxt_words))
        overlap = 0
        for size in range(max_probe, 0, -1):
            if words[-size:] == nxt_words[:size]:
                overlap = size
                break
        words.extend(nxt_words[overlap:])
    return " ".join(words)


def main(dry_run: bool = False) -> int:
    engine = get_engine()
    chunks = engine.store.all_chunks()
    if not chunks:
        print("Index is empty — nothing to re-chunk.")
        return 0
    sources = {s["source_id"]: s for s in engine.store.list_sources()}

    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["source_id"], []).append(c)

    rebuilt: list[dict] = []
    changed = 0
    for source_id, source_chunks in by_source.items():
        meta = sources.get(source_id)
        if meta is None:  # orphan chunks: keep as-is
            rebuilt.extend(source_chunks)
            continue
        text = reconstruct(source_chunks)
        new_chunks = [
            c.to_dict()
            for c in chunk_transcript(
                text=text,
                founder=meta["founder"],
                company=meta.get("company", ""),
                source_id=source_id,
                source_title=meta["source_title"],
                source_type=meta.get("source_type", "other"),
            )
        ]
        if len(new_chunks) != len(source_chunks):
            changed += 1
            biggest = max(len(c["text"].split()) for c in source_chunks)
            print(f"  {meta['founder'][:18]:<19}{meta['source_title'][:40]:<42}"
                  f"{len(source_chunks):>4} -> {len(new_chunks):<4} chunks "
                  f"(was {biggest}w max)")
        meta["chunks"] = len(new_chunks)
        rebuilt.extend(new_chunks)

    print(f"\n{changed} source(s) re-chunked | "
          f"{len(chunks)} -> {len(rebuilt)} chunks total")
    if dry_run:
        print("(dry run — nothing written)")
        return 0

    print(f"re-embedding {len(rebuilt)} chunk(s) and persisting…")
    texts = [c["text"] for c in rebuilt]
    if not engine.embedder.fixed_dim:
        engine.embedder.fit_corpus(texts)
        engine.embedder.save(config.INDEX_DIR / "vectorizer.pkl")
    matrix = engine.embedder.embed(texts)
    engine.store.replace_all(rebuilt, matrix, list(sources.values()))
    stats = engine.store.stats()
    print(f"done — {stats['chunks']} chunks / {stats['sources']} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
