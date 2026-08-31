# Prompts

Every prompt this system sends to a model lives in one file:
**[`backend/prompts.py`](../backend/prompts.py)**. Nothing else under
`backend/` contains prompt text — each module imports what it needs. That file
is where you read them, and where you change them.

To see exactly what is sent, without reading code:

```bash
python scripts/show_prompts.py          # the catalogue, with summaries
python scripts/show_prompts.py --full   # complete text of every prompt
python scripts/show_prompts.py ANSWER_SYSTEM
```

That script reads `backend/prompts.py` directly, so it cannot drift out of date
the way a pasted copy in a document would.

## The catalogue

| Prompt | Sent by | Fires when | Without credentials |
|---|---|---|---|
| `EXPANSION_SYSTEM` | `retrieval.py` | Every question, to widen it into sub-queries | Deterministic themed templates |
| `GROUNDING_SYSTEM` | `grounding.py` | Every question, to decide whether to answer at all | Similarity + coverage + cross-encoder heuristic |
| `ANSWER_SYSTEM` | `generation.py` | Answering from retrieved passages | Extractive: most relevant sentences, quoted |
| `NARRATIVE_SYSTEM` | `generation.py` | Writing a founder's full journey | Extractive summary per theme |
| `FAITHFULNESS_SYSTEM` | `evaluation.py` | Judging a draft claim by claim | Lexical overlap against the passages |
| `CCS_SYSTEM` | `evaluation.py` | Scoring Context Completeness | Keyword presence per theme |
| `FILTER_SYSTEM` | `connectors.py` | Deciding if a search result is really this founder | Name + company keyword heuristic |
| `ATTRIBUTION_SYSTEM` | `attribution.py` | Auditing whether a stored source is the right person | String audit only — a review queue, never a delete list |

`SOURCE_CAVEATS` is not a prompt. It is the shared description of what this
corpus *is*, interpolated into every prompt that reads passages. **Editing it
changes several prompts at once** — which is usually what you want, and
occasionally a nasty surprise.

## What `SOURCE_CAVEATS` is for

The corpus is machine transcripts. They garble proper nouns ("michelle eligbe"
for Mitchell Elegbe), carry no punctuation or casing, and never label speakers.
Prompts that ignore this produce two specific failures, both observed:

1. **Inventing a second person out of a transcription error** — the model reads
   a mangled name as somebody new and writes them into the narrative.
2. **Attributing a host's or panellist's words to the founder** — nothing in
   the text says who is speaking, so the model guesses.

Every passage-reading prompt therefore states these properties up front, and
the judge is told not to penalise a claim for correcting an obvious ASR
misspelling of an entity already named in the passage.

## Deliberate choices worth keeping

Each of these exists because its absence caused a specific failure.

**The grounding prompt refuses when uncertain.** It is told that a refusal
costs the reader one question, while a confident answer built on unsupporting
passages puts an invented fact into a study wearing a citation. It is also told
not to use knowledge from outside the passages — it may well know the answer,
and that is exactly the failure mode, because the corpus is what is being
tested, not the model.

**The relevance filter rejects when uncertain.** A missed source costs far less
than a contaminated corpus. Filing one person's interview under another
founder's name is the worst thing this pipeline can do, and it has happened
twice — see
[data-collection.md](data-collection.md#attribution-the-failure-that-actually-costs-you).

**The narrative prompt forbids uncited sentences** and bans filling an empty
section with generalities. Without that, a thin corpus produces confident
padding that looks like findings.

**The refinement prompt forbids replacing a removed claim with a hedge.** The
point of refinement is to drop unsupported claims, not to restate them
vaguely.

**The answer and narrative prompts work in two steps** — first identify
silently which passages actually concern this founder, then write. Skipping
that made the model treat every retrieved passage as relevant.

## Changing a prompt

1. Edit `backend/prompts.py`.
2. **Add or update the offline fallback.** Every prompt has a deterministic
   counterpart in its own module, because the pipeline must work with no
   credentials. A new prompt without a fallback breaks the offline path.
3. Re-run the evaluation harness and put the numbers in your PR:

   ```bash
   EVAL_GOLDENS_FILE=goldens.samples.json python scripts/evaluate.py --retrieval
   ```

   Retrieval-only is enough for `EXPANSION_SYSTEM`. For generation, judging or
   grounding prompts, run the full harness with credentials — those prompts do
   not affect retrieval metrics at all, so a retrieval-only run will show no
   change and prove nothing.

4. For `GROUNDING_SYSTEM` specifically, check both directions: that answerable
   questions are still answered *and* that the negative controls are still
   refused. `tests/test_grounding.py` covers both, and
   `scripts/measure_abstention.py` re-calibrates the offline thresholds.

## A warning about schemas

Four of these prompts are paired with a JSON schema and sent through
`complete_json`, which constrains the model's output structure. The schema
lives next to the call site, not in `prompts.py`, because it is part of the
contract with the code that consumes the result. If you change what a prompt
asks for, check the schema still matches — a prompt asking for a field the
schema forbids fails at validation, not at read time.
