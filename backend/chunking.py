"""Sentence-aware chunking.

Upgrade over the paper's fixed 150-word / 50-word-overlap cuts: chunks are
assembled from whole sentences up to a target size, so no sentence is split
mid-thought, and the overlap is also sentence-aligned.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field

from . import config

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    id: str
    text: str
    founder: str
    company: str
    source_id: str
    source_title: str
    source_type: str  # youtube | podcast | report | other
    chunk_index: int
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _cap_length(sentences: list[str], max_words: int) -> list[str]:
    """Break over-long 'sentences' into word windows.

    ASR output (YouTube auto-captions, Whisper without punctuation) can be
    thousands of words with no sentence boundary at all. Without this cap the
    whole transcript collapses into a single chunk and retrieval loses all
    granularity, so fall back to the paper's fixed-window behaviour for those
    stretches.
    """
    out: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            out.append(sentence)
            continue
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            if piece:
                out.append(piece)
    return out


def chunk_transcript(
    text: str,
    founder: str,
    company: str,
    source_id: str,
    source_title: str,
    source_type: str = "other",
    target_words: int | None = None,
    overlap_words: int | None = None,
) -> list[Chunk]:
    target = target_words or config.CHUNK_TARGET_WORDS
    overlap = overlap_words or config.CHUNK_OVERLAP_WORDS
    sentences = _cap_length(split_sentences(text), target)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    idx = 0
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        wc = len(sent.split())
        current.append(sent)
        current_words += wc
        i += 1
        if current_words >= target or i == len(sentences):
            body = " ".join(current)
            cid = hashlib.sha1(
                f"{source_id}:{idx}:{body[:80]}".encode()
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=cid,
                    text=body,
                    founder=founder,
                    company=company,
                    source_id=source_id,
                    source_title=source_title,
                    source_type=source_type,
                    chunk_index=idx,
                )
            )
            idx += 1
            if i < len(sentences):
                # sentence-aligned overlap: carry trailing sentences forward
                carried: list[str] = []
                carried_words = 0
                for s in reversed(current):
                    swc = len(s.split())
                    if carried_words + swc > overlap:
                        break
                    carried.insert(0, s)
                    carried_words += swc
                current = carried
                current_words = carried_words
            else:
                current = []
                current_words = 0
    return chunks
