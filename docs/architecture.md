# Architecture

How a transcript becomes a cited narrative, and why each stage is built the way
it is.

```
transcript
   ↓  chunking.py      sentence-aware chunks (~220 words, sentence-aligned overlap)
   ↓  embeddings.py    voyage | fastembed (default) | sentence-transformers | tf-idf
   ↓  store.py         MongoDB $vectorSearch → cosine fallback | local numpy index
   ↓  retrieval.py     multi-query expansion → dense + BM25 → reciprocal-rank fusion
   ↓  rerank.py        cross-encoder, blended with the fusion order
   ↓  generation.py    grounded answer/narrative with inline [S1]…[Sn] citations
   ↓  evaluation.py    faithfulness · hallucination rate · Context Completeness Score
   ↓  engine.py        refinement loop: regenerate flagged claims, keep the better draft
```

## Chunking

Sentence-aware, targeting ~220 words with sentence-aligned overlap, so no chunk
ends mid-sentence.

The word cap is not decoration. Machine transcripts frequently arrive with no
punctuation at all, and a purely sentence-based splitter collapses an entire
hour-long interview into one chunk. That bug was invisible in aggregate scores
and only showed up when the harness started printing how many chunks in the
corpus even contained each fact. Do not raise `CHUNK_TARGET_WORDS` without
re-running `scripts/evaluate.py`.

## Embeddings

Selected in this order, first available wins (`EMBEDDING_BACKEND=auto`):

| Backend | Model | Notes |
|---|---|---|
| Voyage | `voyage-3.5` | Hosted, needs `VOYAGE_API_KEY`. Best quality. |
| **fastembed** | `BAAI/bge-small-en-v1.5`, 384-d | **Default.** ONNX, no torch — neural quality at near-TF-IDF startup cost. |
| sentence-transformers | `BAAI/bge-small-en-v1.5` | Needs torch. Same model, heavier install. |
| TF-IDF | scikit-learn | Always available, no downloads. Keeps the offline path working. |

If the backend changes, the engine detects the dimension mismatch at startup
and transparently re-embeds the corpus. Fixed-dimension embedders only embed
new or changed chunks and reuse stored vectors; TF-IDF refits by design,
because its vector space is defined by the corpus.

## Vector store

`backend/store.py` is a small interface with two implementations.

**MongoDB** (`MONGODB_URI` set). On Atlas it uses native `$vectorSearch` with
metadata pre-filtering — the per-founder filter this pipeline needs — and
auto-provisions the `vector_index` (384-d cosine + `founder` filter) on first
write, so there is no console clicking. Against a plain local `mongod`, which
has no `$vectorSearch`, it transparently falls back to client-side cosine:
correct, just slower.

**Local index** (default). Chunks and embeddings persist under `data/index/`
with atomic writes (tmp + rename) so a crash cannot corrupt the store. Needs
nothing installed.

### Why MongoDB, and when to leave

Since the paper was written, Atlas gained native `$vectorSearch` and
`$rankFusion` hybrid search. Because this app also stores transcripts, metadata
and evaluation results, keeping documents and vectors in one database is a
genuine operational win.

If you outgrow it — millions of chunks, heavy filtering, quantization — a
dedicated engine like **Qdrant** is faster and cheaper per query. The store
layer is a small interface, so adding a `QdrantStore` is a contained task. The
recommendation is to stay on Atlas until retrieval latency or index size
actually hurts: one database is simpler than two.

## Retrieval

Three stages, each fixing a failure the previous one leaves behind.

**1. Multi-query expansion.** The question is expanded into sub-queries via
schema-constrained structured output, so it can never mis-parse. Offline, a
deterministic fallback does the same job with fixed templates. The expansion
prompt is instructed to keep the user's original question — an early version
dropped it, which cost real accuracy until the harness caught it.

**2. Hybrid dense + BM25 with reciprocal-rank fusion.** Dense vectors blur
exact names, dates and figures; BM25 recovers them. Fusion depth per sub-query
is `FUSION_POOL_MULTIPLIER` / `FUSION_POOL_MIN`, and the defaults are the
measured optimum — raising them made retrieval *worse* (hit@7 0.882 → 0.824).

**3. Cross-encoder reranking.** Fusion finds candidates well but orders them
poorly, because it ranks on topical similarity. A cross-encoder reads question
and passage together and can tell that a passage *answers* the question. Its
ranking is **blended** with the fusion order rather than replacing it —
replacing it outright measured worse. Set `RERANK_ENABLED=0` to skip the stage;
retrieval still works, just less precisely.

The full error analysis behind this design is in [evaluation.md](evaluation.md).

## Generation and citation

Every narrative and answer carries inline `[S1]…[Sn]` citations, traceable back
to the source in the UI. A generated narrative downloads as a self-contained
Markdown report with its evaluation scores, citations, and a source appendix.

Offline, generation is deterministic and extractive, and the UI labels it
clearly as such — so the pipeline stays inspectable without credentials, and
extractive output is never mistaken for model output.

## Prompts

All model-facing prompts live beside the code that calls them
(`generation.py`, `evaluation.py`, `retrieval.py`, `connectors.py`) and share a
common brief about what the corpus actually is: **machine transcripts**. They
garble proper nouns ("michelle eligbe" for Mitchell Elegbe), carry no
punctuation or casing, and never label speakers.

Prompts that ignore this produce two specific failures: inventing a second
person out of a transcription error, and attributing a host's or panellist's
words to the founder. The generation and judging prompts address both
explicitly, and the judge is told not to penalise a claim for correcting an
obvious ASR misspelling of an entity already in the passage.

Other deliberate choices:

- The narrative prompt forbids uncited sentences and bans filling an empty
  section with generalities.
- The relevance filter is told that a name collision is the most damaging error
  and to reject when uncertain — a missed source costs far less than a
  contaminated corpus.
- The refinement prompt forbids replacing a removed claim with a hedge.

## Ingest

Built for real corpora, not just the demo.

- **Formats:** `.txt`, `.md`, `.json` (with embedded metadata), and `.srt` /
  `.vtt` caption files — YouTube exports are parsed and cleaned automatically
  (cue numbers, timestamps, `[Music]`-style cues, rolling-caption duplicates).
- **Idempotent upserts:** every source gets a stable `source_id` and a content
  hash. Re-ingesting unchanged content is skipped; changed content atomically
  replaces that source's chunks, never duplicating them.
- **Background jobs:** uploads and batches run as jobs with live progress
  (`GET /api/jobs/{id}`) and the UI shows a progress bar.
- **Error isolation:** one malformed document in a batch fails alone, with its
  reason reported; the rest index normally.
- **Safety rails:** per-file (10 MB), per-batch (200 docs) and per-upload
  (50 files) limits, all configurable; a re-entrant lock serializes concurrent
  index mutations.

The in-process job manager suits the default single-worker deployment. Job
state lives in memory, so a second worker would not see jobs the first started
— swap in Celery/RQ for multi-worker setups (interface in `backend/jobs.py`).

## Module map

```
backend/
  chunking.py       sentence-aware chunker with a word cap for unpunctuated ASR
  ingest.py         file parsing (txt/md/json/srt/vtt), cleaning, validation
  connectors.py     YouTube + RSS discovery, relevance filter, transcript fetch
  asr.py            optional local Whisper transcription
  embeddings.py     voyage | fastembed (default) | sentence-transformers | tf-idf
  store.py          MongoDB ($vectorSearch → cosine fallback) | local numpy
  retrieval.py      multi-query expansion, BM25, RRF hybrid fusion
  rerank.py         cross-encoder reranking, blended with fusion
  attribution.py    LLM audit: does the founder actually appear in this source?
  generation.py     cited narratives & answers (+ offline extractive)
  evaluation.py     faithfulness / hallucination / CCS judges (+ offline)
  eval_harness.py   golden-set scoring, k sweep, error analysis
  sweep.py          roster-wide discover → fetch → ingest
  jobs.py           in-process background jobs with progress
  engine.py         pipeline orchestration
  app.py            FastAPI + static UI

frontend/index.html single-file web app (Dashboard · Ask · Narratives ·
                    Ingest · Evaluate · How it works)

data/
  samples/          fictional first-run corpus (shipped)
  founders.json     founder roster
  podcast_feeds.json curated African tech podcast feeds
  eval/             golden sets, saved runs, latest baseline
  transcripts/      media you collect (git-ignored)
  index/            local vector index (git-ignored)

scripts/            evaluate · rechunk · sync_to_mongo · audit_attribution ·
                    audit_attribution_llm · check_cookies · start_mongodb
tests/              70 tests: ingest, connectors, sweep, eval harness, samples
```
