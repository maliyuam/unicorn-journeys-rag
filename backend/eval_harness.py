"""Evaluation harness.

Follows the practices Andrew Ng pushes for LLM systems:

* **Look at the data.** Every run emits per-question rows (what was retrieved,
  from which source, whether the evidence was found) so failures can be read,
  not just counted. Aggregate scores alone hide bugs — the chunking collapse
  that flattened whole interviews into single chunks was invisible in a mean.
* **Start small and honest.** A compact golden set with verified evidence
  strings beats a large speculative one. Grow it from observed failures.
* **Component-level metrics.** Retrieval and generation are scored separately,
  so a bad answer can be attributed to the stage that caused it.
* **Fast iteration.** The retrieval sweep runs offline with no LLM calls, in
  seconds, so it can gate every change.
* **Ground truth that survives refactors.** Relevance is defined by evidence
  strings appearing in a chunk, not by chunk IDs, so re-chunking or swapping
  embedders does not invalidate the labels.

Metrics mirror the paper: precision, recall and F1 over a k sweep (its
Table 2), plus faithfulness and hallucination rate for generation.
"""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from . import config
from .evaluation import evaluate_faithfulness
from .llm import get_llm
from .retrieval import build_bm25, expand_query, hybrid_search

EVAL_DIR = config.DATA_DIR / "eval"
RUNS_DIR = EVAL_DIR / "runs"
DEFAULT_K_VALUES = (3, 5, 7, 10, 15, 20)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def goldens_path() -> Path:
    """Resolve EVAL_GOLDENS_FILE — absolute, or relative to data/eval/."""
    configured = Path(config.EVAL_GOLDENS_FILE)
    return configured if configured.is_absolute() else EVAL_DIR / configured


def load_goldens() -> list[dict]:
    path = goldens_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("goldens", [])


def load_negative_controls() -> list[dict]:
    """Questions the corpus cannot answer. Scored separately from `goldens`.

    They are kept out of the golden list on purpose: they have no evidence, and
    `validate_goldens` would report every one as a broken label.
    """
    path = goldens_path()
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("negative_controls", [])


def evaluate_abstention(engine, negatives: list[dict]) -> dict:
    """Does the pipeline refuse questions the corpus cannot answer?

    This measures the failure that every other metric here is blind to. A
    retrieval score says how well the best passages rank; it says nothing about
    whether the system should have answered at all. Precision, recall and
    faithfulness are all computed over questions that *do* have answers, so a
    pipeline that answers everything confidently scores identically to one that
    knows its limits.

    `false_answer_rate` is the number that matters: the fraction of
    unanswerable questions that came back with an answer and citations.
    """
    rows = []
    for n in negatives:
        result = engine.ask(n["question"], founder=n.get("founder"))
        abstained = bool(result.get("abstained"))
        rows.append(
            {
                "id": n["id"],
                "question": n["question"],
                "founder": n.get("founder"),
                "why_unanswerable": n.get("why", ""),
                "abstained": abstained,
                "citations_offered": len(result.get("chunks") or []),
                "method": (result.get("support") or {}).get("method", ""),
                "reason": (result.get("support") or {}).get("reason", ""),
            }
        )
    total = len(rows) or 1
    answered = [r for r in rows if not r["abstained"]]
    return {
        "total": len(rows),
        "abstained": len(rows) - len(answered),
        "false_answers": len(answered),
        "false_answer_rate": round(len(answered) / total, 3),
        "rows": rows,
    }


def chunk_is_relevant(chunk_text: str, evidence: list[str]) -> bool:
    """Retrieval ground truth — deliberately strict substring matching.

    Labels must be mechanical and reproducible: a chunk either carries the
    evidence text or it does not. Loosening this would make retrieval scores
    depend on a fuzzy judgement rather than on the corpus.
    """
    body = _norm(chunk_text)
    return any(_norm(e) in body for e in evidence)


_STOPWORDS = {"a", "an", "the", "in", "of", "to", "and", "with", "for", "on",
              "at", "by", "from", "was", "is", "were", "are", "had", "has"}


def _salient(text: str) -> tuple[set[str], set[str]]:
    """Split text into (numeric tokens, content tokens)."""
    body = _norm(text).replace("%", " percent ").replace("$", " ")
    tokens = re.findall(r"[a-z0-9']+", body)
    numbers = {t for t in tokens if any(ch.isdigit() for ch in t)}
    words = {t for t in tokens if t not in numbers and t not in _STOPWORDS}
    return numbers, words


def answer_contains_evidence(answer: str, evidence: list[str]) -> bool:
    """Answer-level check, tolerant of paraphrase but strict on figures.

    A generated answer legitimately rewords its source ("studied computer
    science" for "degree in computer science"), so exact substring matching
    under-counts correct answers. Numbers, though, must survive verbatim —
    a wrong figure is a wrong answer, not a paraphrase.
    """
    if chunk_is_relevant(answer, evidence):
        return True
    a_numbers, a_words = _salient(answer)
    for ev in evidence:
        e_numbers, e_words = _salient(ev)
        if not e_numbers <= a_numbers:
            continue
        if not e_words:
            return True
        overlap = len(e_words & a_words) / len(e_words)
        if overlap >= 0.6:
            return True
    return False


def validate_goldens(engine, goldens: list[dict]) -> list[dict]:
    """Report goldens whose evidence appears nowhere in the corpus.

    These are not model failures — they are broken labels (or missing
    sources), and counting them as recall misses would quietly depress every
    score. Surfacing them is itself error analysis.
    """
    problems = []
    pools = {f: engine.store.all_chunks(founder=f)
             for f in {g.get("founder") for g in goldens}}
    for g in goldens:
        pool = pools[g.get("founder")]
        hits = sum(1 for c in pool if chunk_is_relevant(c["text"], g["evidence"]))
        if hits == 0:
            problems.append(
                {
                    "id": g["id"],
                    "question": g["question"],
                    "founder": g.get("founder"),
                    "reason": "no chunk in the corpus contains this evidence",
                }
            )
    return problems


def evaluate_retrieval(
    engine, goldens: list[dict], k_values=DEFAULT_K_VALUES
) -> dict:
    """Sweep k, scoring precision/recall/F1 against evidence-based labels.

    Uses the production retrieval path (multi-query expansion + hybrid
    dense/BM25 fusion), so the numbers describe the real system.
    """
    llm = get_llm()
    per_k: dict[int, dict] = {}
    details: list[dict] = []

    # expansion is deterministic per question in offline mode and the dominant
    # cost otherwise, so compute it once and reuse across the sweep
    expanded = {
        g["id"]: expand_query(llm, g["question"], g.get("founder"))
        for g in goldens
    }

    # Load and index each founder's slice ONCE. Doing it per query means a
    # full fetch from the store plus a BM25 rebuild for every (golden, k)
    # pair — minutes against a remote store, and an eval nobody waits for.
    founders = {g.get("founder") for g in goldens}
    pools = {f: engine.store.all_chunks(founder=f) for f in founders}
    indexes = {f: build_bm25(pool) if pool else None for f, pool in pools.items()}
    relevant_totals = {
        g["id"]: sum(
            1 for c in pools[g.get("founder")]
            if chunk_is_relevant(c["text"], g["evidence"])
        )
        for g in goldens
    }

    for k in k_values:
        precisions, recalls, ceilings, norm_precisions, rr = [], [], [], [], []
        hits = 0
        for g in goldens:
            founder = g.get("founder")
            total_relevant = relevant_totals[g["id"]]
            if total_relevant == 0:
                continue  # broken label — reported separately by validate_goldens

            retrieved = hybrid_search(
                engine.store, engine.embedder, g["question"],
                expanded[g["id"]], k=k, founder=founder,
                corpus=pools[founder], bm25=indexes[founder],
            )
            ranks = [
                i for i, c in enumerate(retrieved)
                if chunk_is_relevant(c["text"], g["evidence"])
            ]
            rel = [retrieved[i] for i in ranks]
            precision = len(rel) / len(retrieved) if retrieved else 0.0
            recall = len(rel) / total_relevant
            # A question with only 1 evidence-bearing chunk can never exceed
            # 1/k precision, so raw precision punishes large k for reasons
            # that have nothing to do with retrieval quality. Track the
            # ceiling and normalise against it.
            # denominator must match precision's (retrieved can be < k when a
            # founder's pool is small), otherwise normalised precision > 1
            denom = len(retrieved)
            ceiling = min(total_relevant, denom) / denom if denom else 0.0
            precisions.append(precision)
            recalls.append(recall)
            ceilings.append(ceiling)
            norm_precisions.append(precision / ceiling if ceiling else 0.0)
            rr.append(1.0 / (ranks[0] + 1) if ranks else 0.0)
            if rel:
                hits += 1

            details.append(
                {
                    "k": k,
                    "id": g["id"],
                    "question": g["question"],
                    "founder": founder,
                    "origin": g.get("origin", ""),
                    "relevant_in_corpus": total_relevant,
                    "relevant_retrieved": len(rel),
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "hit": bool(rel),
                    "top_sources": [c["source_title"][:60] for c in retrieved[:3]],
                }
            )

        n = len(precisions) or 1
        avg_p = sum(precisions) / n
        avg_r = sum(recalls) / n
        avg_np = sum(norm_precisions) / n
        f1 = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) else 0.0
        f1_norm = (2 * avg_np * avg_r / (avg_np + avg_r)) if (avg_np + avg_r) else 0.0
        per_k[k] = {
            "precision": round(avg_p, 3),
            "precision_ceiling": round(sum(ceilings) / n, 3),
            "precision_normalized": round(avg_np, 3),
            "recall": round(avg_r, 3),
            "f1": round(f1, 3),
            "f1_normalized": round(f1_norm, 3),
            "hit_rate": round(hits / n, 3),
            "mrr": round(sum(rr) / n, 3),
            "questions_scored": len(precisions),
        }

    # What a RAG pipeline actually needs is the evidence inside the context
    # window, using the smallest k that gets there — extra chunks are cost and
    # distraction. So: maximise hit rate, then take the cheapest k that ties.
    best_k = None
    if per_k:
        top_hit = max(m["hit_rate"] for m in per_k.values())
        contenders = [k for k, m in per_k.items() if m["hit_rate"] >= top_hit - 0.01]
        best_k = min(contenders)
    return {"per_k": per_k, "best_k": best_k, "details": details}


def evaluate_generation(engine, goldens: list[dict], k: int) -> dict:
    """Answer-level scoring: did the answer state the evidence, and is it
    grounded in the retrieved context?"""
    llm = get_llm()
    rows = []
    for g in goldens:
        result = engine.ask(g["question"], founder=g.get("founder"), k=k)
        answer = result["answer"]
        chunks = result["chunks"]
        context_had_it = any(
            chunk_is_relevant(c["text"], g["evidence"]) for c in chunks
        )
        answer_had_it = answer_contains_evidence(answer, g["evidence"])
        faith = evaluate_faithfulness(llm, answer, chunks) if chunks else {
            "faithfulness": 0.0, "hallucination_rate": 0.0, "engine": "n/a",
        }
        rows.append(
            {
                "id": g["id"],
                "question": g["question"],
                "context_had_evidence": context_had_it,
                "answer_had_evidence": answer_had_it,
                # the failure that matters: retrieval worked, generation lost it
                "generation_lost_it": context_had_it and not answer_had_it,
                "faithfulness": faith["faithfulness"],
                "hallucination_rate": faith["hallucination_rate"],
            }
        )
    n = len(rows) or 1
    return {
        "answer_accuracy": round(sum(r["answer_had_evidence"] for r in rows) / n, 3),
        "context_recall": round(sum(r["context_had_evidence"] for r in rows) / n, 3),
        "generation_loss_rate": round(
            sum(r["generation_lost_it"] for r in rows) / n, 3
        ),
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 3),
        "hallucination_rate": round(
            sum(r["hallucination_rate"] for r in rows) / n, 3
        ),
        "rows": rows,
        "judge": rows[0]["faithfulness"] if rows else None,
    }


def run_evaluation(
    engine,
    k_values=DEFAULT_K_VALUES,
    include_generation: bool = True,
    save: bool = True,
) -> dict:
    goldens = load_goldens()
    if not goldens:
        raise ValueError(f"no goldens found — expected {goldens_path()}")

    problems = validate_goldens(engine, goldens)
    usable = [g for g in goldens if g["id"] not in {p["id"] for p in problems}]
    retrieval = evaluate_retrieval(engine, usable, k_values)
    generation = (
        evaluate_generation(engine, usable, retrieval["best_k"] or config.DEFAULT_TOP_K)
        if include_generation and usable
        else None
    )

    stats = engine.store.stats()
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus": {
            "chunks": stats["chunks"],
            "sources": stats["sources"],
            "founders": len(stats["founders"]),
        },
        "embedder": engine.embedder.name,
        "store": engine.store.name,
        "llm_mode": get_llm().mode,
        "goldens_total": len(goldens),
        "goldens_scored": len(usable),
        "unverifiable_goldens": problems,
        "retrieval": retrieval,
        "generation": generation,
        # Scored whenever the golden file supplies negative controls. Retrieval
        # and generation metrics are computed only over answerable questions,
        # so without this a pipeline that answers everything looks perfect.
        "abstention": (
            evaluate_abstention(engine, load_negative_controls())
            if load_negative_controls()
            else None
        ),
    }
    if save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = result["timestamp"].replace(":", "").replace("-", "")[:15]
        (RUNS_DIR / f"eval-{stamp}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        (EVAL_DIR / "latest.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


def load_previous_run() -> dict | None:
    path = EVAL_DIR / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
