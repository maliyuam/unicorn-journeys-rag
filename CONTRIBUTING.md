# Contributing

Thanks for taking a look. This is a research-grade RAG pipeline, so the bar
that matters most here is **measurement**: a change that improves retrieval
should show it on the golden set, and a change that cannot be measured should
say so plainly.

## Setting up

```bash
git clone https://github.com/maliyuam/unicorn-journeys-rag.git
cd unicorn-journeys-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest ruff
```

No credentials are required to develop. Without them the app runs in offline
extractive mode, and the whole test suite is hermetic — it makes no network
calls, needs no API key, and never touches a configured MongoDB.

Load the bundled sample corpus and you have a working system:

```bash
python -m uvicorn backend.app:app --port 8017
curl -X POST localhost:8017/api/ingest/samples
```

## Before you open a pull request

```bash
ruff check .                # lint (enforced in CI)
python -m pytest tests/ -q  # 70 tests (enforced in CI)
```

`ruff format` is **not** enforced. The codebase is hand-formatted for
readability and a blanket reformat would bury real diffs in noise.

If you touched retrieval, chunking, embeddings, or prompts, also run the
evaluation harness and put the before/after numbers in the PR description:

```bash
EVAL_GOLDENS_FILE=goldens.samples.json python scripts/evaluate.py --retrieval
```

That is the fast path — retrieval only, no LLM calls, seconds to run. Against
your own swept corpus, drop the `EVAL_GOLDENS_FILE` override to use the real
golden set.

## What good looks like here

**Measure retrieval changes.** Two changes have already been rejected by this
harness rather than shipped on intuition — a wider fusion pool that dropped
hit@7 from 0.882 to 0.824, and a cross-encoder that replaced fusion outright
instead of blending with it. "It felt better" is not evidence; `hit@k` is.

**Explain the why in comments, not the what.** The existing comments record
what was tried and what it cost. Keep that. A comment that restates the code
is noise; a comment that says "raising this made retrieval worse, re-measure
before changing" saves the next person a day.

**Grow the golden set from observed failures.** When you fix a retrieval bug,
add the question that exposed it to `data/eval/goldens.json` (or
`goldens.samples.json` if it is reproducible on the sample corpus). Evidence
strings must be read out of the indexed text, not written from memory — the
harness matches them by substring, and a label that matches nothing is
reported as a broken label rather than a model failure.

**Never commit corpus media or credentials.** Collected transcripts are
third-party content and stay in the git-ignored `data/transcripts/`. If you
need demo material, add it to `data/samples/` — and keep it fictional, for the
reason in [data/samples/README.md](data/samples/README.md).

**Tests for anything with a failure mode.** Especially attribution: filing one
person's interview under another founder's name is the worst thing this
pipeline can do, and both real cases of it are pinned by tests.

## Good first contributions

- **A `QdrantStore`.** `backend/store.py` is a small interface with two
  implementations already; a third is a contained piece of work.
- **Provider-agnostic LLM routing.** `backend/llm.py` is Anthropic-plus-offline
  today. `get_llm()` returns one object with `complete`/`complete_json`, so
  routing it through something like `aisuite` would let this run on OpenAI,
  Gemini, or a local Ollama model — which matters for anyone reproducing the
  results without an Anthropic key.
- **More golden questions.** The dev set is deliberately small and honest.
  Adding well-labelled questions makes every future change easier to judge.

## Reporting bugs

Include the output of `GET /api/status` (it reports which store, embedder and
LLM mode are actually in use — most confusing behaviour turns out to be a
silent fallback), what you ran, and what you expected. For transcript-fetch
failures, `GET /api/connectivity` says in seconds whether the network is being
bot-checked, which is the cause more often than not.

Security issues go through [SECURITY.md](SECURITY.md), not the public tracker.
