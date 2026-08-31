## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- What problem this solves, or what it makes possible. -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `python -m pytest tests/ -q` passes
- [ ] No credentials, corpus media, or generated index files are committed

## Retrieval impact

<!--
Delete this section if you did not touch retrieval, chunking, embeddings or
prompts. Otherwise paste before/after numbers — intuition has been wrong here
before:

  EVAL_GOLDENS_FILE=goldens.samples.json python scripts/evaluate.py --retrieval

Remember that Atlas $vectorSearch is approximate: a one-question difference on
a small set is noise, not a result.
-->

| k | hit@k before | hit@k after |
|---|---|---|
|   |              |             |
