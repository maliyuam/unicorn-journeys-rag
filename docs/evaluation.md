# Evaluation

Built the way Andrew Ng argues LLM systems should be: **error analysis first,
metrics second**, on a small honest dev set, fast enough to run on every change.

```bash
# Against the bundled sample corpus — works on a fresh clone
EVAL_GOLDENS_FILE=goldens.samples.json python scripts/evaluate.py --retrieval

# Against your own swept corpus
python scripts/evaluate.py             # retrieval sweep + generation
python scripts/evaluate.py --retrieval # fast, no LLM calls
python scripts/evaluate.py --no-save   # don't write a baseline
```

Also available as a one-click job in the **Evaluate** tab.

## The two golden sets

| File | Labelled against | Use it for |
|---|---|---|
| `data/eval/goldens.samples.json` | The fictional corpus in `data/samples/`, which ships with the repo | Verifying the harness end-to-end on a fresh clone, and CI |
| `data/eval/goldens.json` | A real corpus collected with the discovery connectors | Measuring actual retrieval quality on real media |

**Do not read the sample scores as a quality result.** On 13 chunks across four
founders, with the per-founder filter applied, retrieval scores hit@3 = 1.000
and misses nothing. That tells you the harness, the store, the embedder and the
retriever are all wired up correctly on your machine — and nothing at all about
how the system performs on a real corpus, where a founder has hundreds of
chunks and the evidence for a question competes with near-duplicates. Real
numbers come from `goldens.json` against real media; the honest ones from that
corpus are further down this page.

`goldens.json` will report every label as **broken** until you have swept the
media it was written against — that is correct behaviour, not a failure. The
harness reports unverifiable goldens separately rather than counting them as
model misses, precisely so a missing corpus cannot quietly depress your scores.

Point the harness at either set with `EVAL_GOLDENS_FILE` (a filename relative
to `data/eval/`, or an absolute path).

## How labels work

Relevance is defined by **evidence text appearing in a retrieved chunk**, not
by chunk ID. That single choice is what makes the labels survive re-chunking
and embedder swaps — the thing that otherwise invalidates a golden set every
time you touch retrieval.

Matching is deliberately strict substring matching after normalising case and
whitespace. A chunk either carries the evidence or it does not; loosening it
would make retrieval scores depend on a fuzzy judgement rather than on the
corpus. Evidence strings must therefore be read out of the *ingested* text
before being written down — including its ASR misspellings, since that is what
is actually in the index.

Answer-level scoring is more forgiving, because a generated answer legitimately
rewords its source ("studied computer science" for "degree in computer
science"). Numbers are the exception: they must survive verbatim, since a wrong
figure is a wrong answer, not a paraphrase.

## What the harness reports

- **Per-component metrics.** Retrieval and generation are scored separately, so
  a wrong answer can be attributed to the stage that caused it.
  `generation_loss_rate` isolates the specific failure of retrieving the
  evidence and then not using it.
- **Honest precision.** Raw precision@k is capped by how few chunks carry each
  fact — a one-chunk fact can never beat 1/k — so the harness reports that
  ceiling and a normalised precision beside it. It selects `best_k` as the
  cheapest k that gets evidence into context. Chasing raw F1 would have picked
  k=3 and starved the generator.
- **The failures, in full.** Every run prints the questions that missed, what
  was retrieved instead, and how many chunks in the corpus even contain the
  evidence. Aggregates alone hid a chunking bug that had collapsed whole
  interviews into single chunks.
- **Label validation.** Goldens whose evidence appears nowhere are reported as
  broken labels, not model failures.
- **Regression tracking.** Each run saves a baseline; the next run prints the
  delta per k.

Two changes were rejected by this harness rather than shipped on intuition: a
wider fusion pool (hit@7 0.882 → 0.824) and, in the other direction, it caught
that LLM query expansion was *dropping the user's original question*, which
cost real accuracy until fixed.

## Reranking, and what error analysis actually showed

The two questions that never retrieved their evidence looked like one problem
and were three.

**1. RRF dilution.** For one, the answer chunk ranked **BM25 #3** on a
sub-query — but with `RRF_K=60` a single strong hit (1/64) loses to mediocre
chunks that appear across every sub-query. Sweeping `RRF_K` alone never lifted
hit@7 above 0.882, so tuning was not the answer.

**2. Ranking, not recall.** Both answer chunks were *in* the candidate pool
(positions 33 and 61), just below the cutoff. That is a reranking problem. A
cross-encoder reads question and passage together and can tell that a passage
answers a question. Replacing fusion with it was worse (hit@7 0.882 → 0.824:
it evicted chunks fusion ranked well), so the two rank lists are **blended**.

Result: misses 2 → 1, hit@3 0.529 → 0.706, hit@5 0.706 → 0.824,
hit@10 0.882 → 0.941, MRR 0.460 → 0.528.

**3. World knowledge.** The last miss — "which ride-hailing company has
Flutterwave worked with?" against "we've worked with Uber" — is not a ranking
bug. All three local cross-encoders rank that passage *last*, because the
question says "ride-hailing" and the passage says "Uber". Bridging that needs
knowledge the small models do not carry, so the fix lives in LLM query
expansion, which is instructed to probe named candidates for "which X"
questions. It needs Anthropic credentials to engage; offline, this class of
question stays unanswerable.

## Reading small deltas

Atlas `$vectorSearch` is **approximate**, so borderline chunks genuinely
flicker in and out between identical runs. On a 17-question set, a difference
of one question is noise, not a regression. If a change looks like it moved
hit@k by a single question, run it again before believing it.

## What this harness cannot tell you

Read these before quoting any number from it. They are limits of the design,
not bugs, and each one bounds a claim you might otherwise make.

**1. Refusal is now measured, but only against 14 questions.**
This used to read "refusal is never measured", because every golden was
answerable by construction and the pipeline had no abstention path at all —
asked about a person absent from the corpus, it returned the top-k passages
as an answer with `[S1]` citations that lent it unearned authority.

Both halves are fixed. `data/eval/goldens.samples.json` now carries
`negative_controls`, and every run reports:

```
ABSTENTION (questions the corpus cannot answer)
  refused correctly    14/14
  false answer rate    0.000
```

`false_answer_rate` is the number to watch: the fraction of unanswerable
questions that came back with an answer and citations. The gate itself is
`backend/grounding.py`.

What this still does not tell you: 14 questions on a fictional corpus is a
smoke test, not a measurement of refusal quality. Write negative controls for
your own corpus — questions it plausibly *should* answer but cannot — and
re-run. `scripts/measure_abstention.py` reports the signal distributions and
suggests thresholds from your data.

And note the offline gate is materially weaker than the LLM one. On the
default embedder it refused 14/14; under the TF-IDF fallback it refused 7/14,
because TF-IDF cosine does not separate answerable from unanswerable at all
(the two distributions overlap, with the unanswerable median *higher*). If
abstention matters to you, do not run TF-IDF.

**2. Retrieval is scored with the founder filter applied.** `evaluate_retrieval`
passes `founder=` to `hybrid_search`, so each question is answered against one
founder's chunks — tens of passages, not the whole corpus. `POST /api/ask`
without a `founder` searches everything, which is a materially harder task.
The reported hit@k is therefore an upper bound on the open-corpus case, and
should not be quoted as the system's general retrieval accuracy.

**3. The real golden set is n=17.** A "hit@3 0.529 → 0.706" improvement is
three questions changing state. On a set this size the confidence interval
swamps the effect, and Atlas `$vectorSearch` is approximate on top of that.
These numbers are useful as a regression tripwire and as error-analysis
prompts. They are not evidence of a general improvement, and nothing here
supports a claim of statistical significance.

**4. The judge and the generator are the same model family.** Faithfulness,
hallucination rate and CCS are scored by Claude on Claude's output, with no
human-labelled subset to calibrate against and no inter-rater agreement
measured. LLM judges are known to favour their own generations. Treat these as
internal consistency checks, not independent evaluation. The refinement loop
compounds it: it keeps whichever draft the same judge prefers.

**5. The environment is pinned; the corpus still is not.** Dependencies are
now exact (`==`) in `requirements.txt` and `requirements-dev.txt`, so a library
release cannot silently move your numbers. Model *names* are pinned in
`backend/config.py` — but a name is not a revision, so an upstream reupload of
`BAAI/bge-small-en-v1.5` would still change retrieval. Pin the revision if you
are publishing.

The corpus remains the real gap: `goldens.json` is labelled against collected
media that is not shipped, so nobody else can reproduce those headline numbers.
The sample corpus and `goldens.samples.json` *are* fully reproducible, which is
what makes them useful for CI and for checking a change did what you think.
Archive your own corpus separately, with its own provenance, if you intend to
publish a result from it.

One thing that was checked and is *not* a problem: the golden questions do not
leak their evidence's vocabulary into the query. Mean content-token overlap
between question and evidence is 0.14, and only 4 of 17 echo half or more of
the evidence tokens — so hybrid retrieval's BM25 half is not being flattered by
questions written to match the passage.

## Growing the set

Grow it from observed failures — error analysis — not from speculation. When
you fix a retrieval bug, add the question that exposed it. Keep it small and
honest: a compact golden set with verified evidence strings beats a large
speculative one, and it stays fast enough to gate every change.
