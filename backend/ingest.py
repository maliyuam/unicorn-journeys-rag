"""Production ingest: file parsing, transcript normalization, validation.

Accepted formats:
  .txt / .md  — plain transcript text
  .json      — {"founder", "company"?, "source_title", "source_type"?, "text"}
               (or minimal {"text": ...}; missing metadata falls back to the
               defaults supplied with the upload)
  .srt       — SubRip captions (YouTube export format)
  .vtt       — WebVTT captions (YouTube API format)

Every document passes through `normalize_transcript` (caption-cue removal,
timestamp stripping, whitespace collapse) and `validate_doc` before chunking.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import config

ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".srt", ".vtt"}


class IngestError(ValueError):
    """User-correctable ingest problem (bad file, missing field, too large)."""


# --- parsing ----------------------------------------------------------------

_SRT_TIMESTAMP = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}.*$"
)
_SRT_INDEX = re.compile(r"^\s*\d+\s*$")
_VTT_HEADER = re.compile(r"^(WEBVTT|Kind:|Language:|NOTE|STYLE|REGION)\b")
_TAG = re.compile(r"<[^>]+>")
_INLINE_TIME = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s+")
_CUE_BRACKET = re.compile(r"\[(?:[A-Za-z ]{1,30})\]|\((?:[a-z ]{1,25})\)")


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8-sig", errors="replace")


def parse_captions(text: str) -> str:
    """SRT/VTT -> plain text. Drops cue numbers, timestamps, headers, tags,
    and consecutive duplicate lines (rolling captions repeat lines)."""
    lines: list[str] = []
    prev = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or _SRT_INDEX.match(line)
            or _SRT_TIMESTAMP.match(line)
            or _VTT_HEADER.match(line)
        ):
            continue
        line = _TAG.sub("", line).strip()
        if not line or line == prev:
            continue
        lines.append(line)
        prev = line
    return " ".join(lines)


def parse_upload(filename: str, raw: bytes) -> dict:
    """Return a partial doc {text, founder?, company?, source_title?, source_type?}."""
    if len(raw) > config.MAX_FILE_BYTES:
        raise IngestError(
            f"{filename}: file exceeds {config.MAX_FILE_BYTES // (1024*1024)} MB limit"
        )
    lowered = filename.lower()
    ext = "." + lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise IngestError(
            f"{filename}: unsupported type '{ext or 'none'}' "
            f"(allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))})"
        )
    text = _decode(raw)
    if ext == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise IngestError(f"{filename}: invalid JSON ({e})") from e
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise IngestError(f'{filename}: JSON must be an object with a "text" field')
        return {
            k: v
            for k, v in data.items()
            if k in ("founder", "company", "source_title", "source_type", "text")
        }
    if ext in (".srt", ".vtt"):
        return {"text": parse_captions(text)}
    return {"text": text}


# --- normalization & validation ---------------------------------------------

def normalize_transcript(text: str) -> str:
    text = _CUE_BRACKET.sub(" ", text)  # [Music], (laughs) ...
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(_INLINE_TIME.sub("", line))
    text = " ".join(cleaned_lines)
    return re.sub(r"\s+", " ", text).strip()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_source_id(founder: str, source_title: str) -> str:
    slug = _SLUG_RE.sub("-", f"{founder}-{source_title}".lower()).strip("-")[:80]
    return slug or "untitled"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def validate_doc(doc: dict) -> dict:
    """Normalize + validate one document dict. Raises IngestError."""
    founder = (doc.get("founder") or "").strip()
    source_title = (doc.get("source_title") or "").strip()
    if not founder:
        raise IngestError("founder is required")
    if not source_title:
        raise IngestError("source_title is required")
    source_type = (doc.get("source_type") or "other").strip().lower()
    if source_type not in ("youtube", "podcast", "report", "other"):
        source_type = "other"
    text = normalize_transcript(doc.get("text") or "")
    if len(text) < config.MIN_TEXT_CHARS:
        raise IngestError(
            f"transcript too short after cleaning "
            f"({len(text)} chars, minimum {config.MIN_TEXT_CHARS})"
        )
    if len(text) > config.MAX_TEXT_CHARS:
        raise IngestError(
            f"transcript too long ({len(text)} chars, maximum {config.MAX_TEXT_CHARS})"
        )
    return {
        "founder": founder,
        "company": (doc.get("company") or "").strip(),
        "source_title": source_title,
        "source_type": source_type,
        "text": text,
        "source_id": (doc.get("source_id") or make_source_id(founder, source_title)),
        "content_hash": content_hash(text),
    }


# --- shipped sample corpus --------------------------------------------------

def load_sample_docs() -> list[dict]:
    """The fictional corpus in `data/samples/`, ready for `ingest_documents`.

    It exists so a fresh clone can retrieve, cite and evaluate before any real
    media has been collected. The founders and companies are invented — see
    `data/samples/README.md`.
    """
    if not config.SAMPLES_DIR.is_dir():
        raise IngestError(
            f"no sample corpus at {config.SAMPLES_DIR} — it ships with the repo"
        )
    docs: list[dict] = []
    for path in sorted(config.SAMPLES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise IngestError(f"{path.name}: invalid JSON ({e})") from e
        docs.append(
            {
                k: v
                for k, v in data.items()
                if k in ("founder", "company", "source_title", "source_type", "text")
            }
        )
    if not docs:
        raise IngestError(f"no *.json sample transcripts found in {config.SAMPLES_DIR}")
    return docs
