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

## Growing the set

Grow it from observed failures — error analysis — not from speculation. When
you fix a retrieval bug, add the question that exposed it. Keep it small and
honest: a compact golden set with verified evidence strings beats a large
speculative one, and it stays fast enough to gate every change.
