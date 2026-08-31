# Sample corpus (fictional)

Four short interview transcripts used as the project's first-run corpus. Load
them from the dashboard (**Load sample corpus**) or with:

```bash
curl -X POST localhost:8017/api/ingest/samples
```

## These are invented

**Every founder and company here is fictional.** Amara Okonkwo/PaySwift, Thabo
Molefe/SolaraGrid, Leila Haddad/MedixCare and Kwame Mensah/AgriLink do not
exist, and no sentence in these files is a record of anything a real person
said. They were written for this repository so that a fresh clone can chunk,
embed, retrieve, cite and evaluate without network access, credentials, or a
discovery sweep — and so that no real person's words are ever paraphrased,
truncated or attributed by a demo.

The real study corpus is not shipped. It is third-party media that belongs to
the people who published it; the connectors collect it into `data/transcripts/`
on your own machine, and that directory is git-ignored.

## Why they look the way they do

The text is written to exercise the pipeline, not to flatter it:

- **Length** — 540–640 words each, so the ~220-word chunker produces three or
  four chunks per source and retrieval has something to rank.
- **Specific facts** — dates, licence names, headcounts, percentages and
  funding rounds give the golden set (`data/eval/goldens.samples.json`)
  verifiable evidence strings to match on.
- **A host and a guest** — the generation and judging prompts must attribute
  claims to the founder rather than the interviewer, which only gets tested if
  someone else is talking too.
- **Facts spread thinly** — most facts appear in exactly one chunk, which is
  the honest case: it caps raw precision@k and is why the eval harness reports
  a precision ceiling alongside the raw number.

## Adding your own

Drop any `{"founder", "company", "source_title", "source_type", "text"}` JSON
file in this directory and it joins the sample set. `source_type` is one of
`youtube`, `podcast`, `report`, `other`. Ingest is idempotent, so re-running is
safe — unchanged sources are skipped.
