"""Run the evaluation harness and print a readable report.

    python scripts/evaluate.py                 # full run (retrieval + generation)
    python scripts/evaluate.py --retrieval     # fast: retrieval sweep only
    python scripts/evaluate.py --no-save       # don't write a baseline

Prints the k sweep (the paper's Table 2 on this corpus), the failures worth
reading, and a diff against the previous saved run so regressions are obvious.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engine import get_engine  # noqa: E402
from backend.eval_harness import (  # noqa: E402
    load_previous_run,
    run_evaluation,
)


def _arrow(cur: float, prev: float | None) -> str:
    if prev is None:
        return ""
    delta = cur - prev
    if abs(delta) < 0.005:
        return "   ="
    return f" {'+' if delta > 0 else ''}{delta:.3f}"


def main() -> int:
    retrieval_only = "--retrieval" in sys.argv
    save = "--no-save" not in sys.argv

    previous = load_previous_run()
    engine = get_engine()
    result = run_evaluation(
        engine, include_generation=not retrieval_only, save=save
    )

    print(f"corpus:   {result['corpus']['chunks']} chunks / "
          f"{result['corpus']['sources']} sources / "
          f"{result['corpus']['founders']} founders")
    print(f"embedder: {result['embedder']}")
    print(f"store:    {result['store']}")
    print(f"llm:      {result['llm_mode']}")
    print(f"goldens:  {result['goldens_scored']}/{result['goldens_total']} scored")

    if result["unverifiable_goldens"]:
        print("\nUNVERIFIABLE GOLDENS (label problem, not a model failure):")
        for p in result["unverifiable_goldens"]:
            print(f"  {p['id']}  {p['question'][:64]}")
            print(f"       {p['reason']} (founder={p['founder']})")

    prev_k = (previous or {}).get("retrieval", {}).get("per_k", {})
    print("\nRETRIEVAL SWEEP")
    print(f"{'k':>4}{'prec':>7}{'ceil':>7}{'prec_n':>8}{'recall':>8}"
          f"{'F1_n':>7}{'hit':>7}{'MRR':>7}   vs prev (hit)")
    for k, m in result["retrieval"]["per_k"].items():
        pk = prev_k.get(str(k)) or prev_k.get(k) or {}
        print(f"{k:>4}{m['precision']:>7.3f}{m['precision_ceiling']:>7.3f}"
              f"{m['precision_normalized']:>8.3f}{m['recall']:>8.3f}"
              f"{m['f1_normalized']:>7.3f}{m['hit_rate']:>7.3f}{m['mrr']:>7.3f}"
              f"{_arrow(m['hit_rate'], pk.get('hit_rate'))}")
    print("\n  prec   = raw precision@k (capped by how few chunks hold the evidence)")
    print("  ceil   = best precision@k reachable given that cap")
    print("  prec_n = precision / ceiling — 1.0 means every reachable hit was retrieved")
    print(f"\nbest k = {result['retrieval']['best_k']} "
          "(cheapest k reaching peak hit rate — evidence in context)")

    best_k = result["retrieval"]["best_k"]
    misses = [
        d for d in result["retrieval"]["details"]
        if d["k"] == best_k and not d["hit"]
    ]
    print(f"\nERROR ANALYSIS — retrieval misses at k={best_k}: {len(misses)}")
    for d in misses:
        print(f"  {d['id']} [{d['origin']}] {d['question'][:60]}")
        print(f"       founder={d['founder']} relevant_in_corpus={d['relevant_in_corpus']}")
        print(f"       retrieved instead: {', '.join(d['top_sources'][:2])}")

    gen = result.get("generation")
    if gen:
        print("\nGENERATION")
        print(f"  answer accuracy      {gen['answer_accuracy']:.3f}")
        print(f"  context recall       {gen['context_recall']:.3f}")
        print(f"  generation loss rate {gen['generation_loss_rate']:.3f}"
              "   (evidence retrieved but missing from the answer)")
        print(f"  faithfulness         {gen['faithfulness']:.3f}")
        print(f"  hallucination rate   {gen['hallucination_rate']:.3f}")
        lost = [r for r in gen["rows"] if r["generation_lost_it"]]
        if lost:
            print(f"\n  generation dropped evidence on {len(lost)} question(s):")
            for r in lost:
                print(f"    {r['id']}  {r['question'][:62]}")

    if save:
        print("\nbaseline saved to data/eval/latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
