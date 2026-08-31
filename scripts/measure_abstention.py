"""Calibrate the offline grounding gate against YOUR corpus.

    python scripts/measure_abstention.py

The shipped thresholds (`ABSTAIN_MIN_SIMILARITY`, `ABSTAIN_MIN_COVERAGE`,
`ABSTAIN_MIN_RERANK`) were fitted on 15 answerable and 14 unanswerable
questions against the fictional sample corpus. That is a small set and a
different corpus, so on your own data they are a starting point, not a setting.

This script prints the two distributions and the thresholds that separate them,
so you can set yours from evidence. It reads two things from your golden file:
`goldens` (answerable) and `negative_controls` (not). If you have no negative
controls, write some first — the gate cannot be calibrated without examples of
what it is supposed to refuse.

**Similarity is embedder-specific.** BGE compresses unrelated text into a
narrow high band (~0.5-0.7); TF-IDF puts it near zero. A threshold tuned for
one is meaningless for the other, which is why the default lives on the
embedder class (`BaseEmbedder.abstain_min_similarity`) rather than in a single
global constant. Recalibrate after changing embedders.

This script is READ-ONLY: it retrieves and scores, and never ingests or writes
to the store.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.engine import get_engine  # noqa: E402
from backend.eval_harness import goldens_path  # noqa: E402
from backend.grounding import coverage  # noqa: E402
from backend.llm import get_llm  # noqa: E402
from backend.retrieval import expand_query, hybrid_search  # noqa: E402


def _percentile(values, p):
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def signals_for(engine, llm, question, founder, companies):
    subqueries = expand_query(llm, question, founder)
    chunks = hybrid_search(
        engine.store, engine.embedder, question, subqueries,
        k=config.DEFAULT_TOP_K, founder=founder,
    )
    if not chunks:
        return None
    sims = [c["question_similarity"] for c in chunks if c.get("question_similarity") is not None]
    rrs = [c["rerank_score"] for c in chunks if c.get("rerank_score") is not None]
    return {
        "sim": max(sims) if sims else None,
        "rr": max(rrs) if rrs else None,
        "cov": coverage(question, chunks, founder or "", companies.get(founder, "")),
    }


def main() -> int:
    path = goldens_path()
    if not path.exists():
        print(f"no golden file at {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    positives = data.get("goldens", [])
    negatives = data.get("negative_controls", [])
    if not negatives:
        print(f"{path.name} has no `negative_controls`. The gate cannot be "
              "calibrated without questions it is supposed to refuse — add a "
              "handful your corpus genuinely cannot answer.", file=sys.stderr)
        return 1

    engine = get_engine()
    llm = get_llm()
    companies = {c["founder"]: c.get("company", "") for c in engine.store.all_chunks()}

    print(f"corpus:   {engine.store.stats()['chunks']} chunks")
    print(f"embedder: {engine.embedder.name}")
    print(f"store:    {engine.store.name}")
    print(f"goldens:  {path.name} — {len(positives)} answerable, {len(negatives)} unanswerable\n")

    pos = [s for s in (signals_for(engine, llm, g["question"], g.get("founder"), companies)
                       for g in positives) if s]
    neg = [s for s in (signals_for(engine, llm, n["question"], n.get("founder"), companies)
                       for n in negatives) if s]

    for label, rows in (("ANSWERABLE", pos), ("UNANSWERABLE", neg)):
        sims = [r["sim"] for r in rows if r["sim"] is not None]
        covs = [r["cov"] for r in rows]
        rrs = [r["rr"] for r in rows if r["rr"] is not None]
        print(f"{label} (n={len(rows)})")
        if sims:
            print(f"  similarity  min={min(sims):.3f}  p50={_percentile(sims, .5):.3f}  max={max(sims):.3f}")
        print(f"  coverage    min={min(covs):.2f}  p50={_percentile(covs, .5):.2f}  max={max(covs):.2f}")
        if rrs:
            print(f"  rerank      min={min(rrs):+.2f}  p50={_percentile(rrs, .5):+.2f}  max={max(rrs):+.2f}")
        print()

    # Suggest the rule the gate actually implements:
    #   answer if sim >= S and (cov >= C or rr >= R)
    sims_pos = [r["sim"] for r in pos if r["sim"] is not None]
    if not sims_pos:
        print("no similarity signal available; cannot suggest thresholds")
        return 0

    best = None
    sim_grid = sorted({round(r["sim"], 3) for r in pos + neg if r["sim"] is not None})
    cov_grid = sorted({round(r["cov"], 2) for r in pos + neg})
    rr_grid = sorted({round(r["rr"], 1) for r in pos + neg if r["rr"] is not None}) or [None]
    for S in sim_grid:
        for C in cov_grid:
            for R in rr_grid:
                def answers(r, S=S, C=C, R=R):
                    if r["sim"] is None or r["sim"] < S:
                        return False
                    if r["cov"] >= C:
                        return True
                    return R is not None and r["rr"] is not None and r["rr"] >= R
                kept = sum(1 for r in pos if answers(r))
                leaked = sum(1 for r in neg if answers(r))
                if kept < len(pos):
                    continue  # never trade away a real answer
                if best is None or leaked < best[1]:
                    best = ((S, C, R), leaked, kept)

    if best is None:
        print("No threshold answers every answerable question. Your two "
              "distributions overlap: inspect the weakest answerable questions "
              "above and decide which side to err on.")
        return 0

    (S, C, R), leaked, kept = best
    print("SUGGESTED (answers every answerable question, fewest false answers):")
    print(f"  ABSTAIN_MIN_SIMILARITY={S}")
    print(f"  ABSTAIN_MIN_COVERAGE={C}")
    if R is not None:
        print(f"  ABSTAIN_MIN_RERANK={R}")
    print(f"\n  answers {kept}/{len(pos)} answerable, "
          f"wrongly answers {leaked}/{len(neg)} unanswerable")
    print("\n  Fitting three thresholds on a small set overfits. Prefer round "
          "numbers with margin over the exact optimum, and re-check after "
          "adding goldens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
