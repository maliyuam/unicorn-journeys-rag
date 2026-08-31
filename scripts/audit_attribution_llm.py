"""LLM attribution audit across the whole corpus.

Reads each source's transcript and judges whether the founder it was filed
under actually speaks in it, is discussed in it, or is absent. Unlike the
string-matching audit, this sees through speech-recognition errors and company
renames.

    python scripts/audit_attribution_llm.py                    # report
    python scripts/audit_attribution_llm.py --founder "Jeremy Johnson"
    python scripts/audit_attribution_llm.py --remove           # delete absent
    python scripts/audit_attribution_llm.py --json out.json    # save verdicts

Requires Anthropic credentials; without them the audit cannot run, because
string matching is exactly what fails here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.attribution import audit_corpus  # noqa: E402
from backend.engine import get_engine  # noqa: E402
from backend.llm import LLMUnavailable, get_llm  # noqa: E402

MARK = {"speaker": "OK  ", "discussed": "ABOUT", "absent": "WRONG"}


def main() -> int:
    args = sys.argv[1:]
    remove = "--remove" in args
    founders = None
    if "--founder" in args:
        founders = [args[args.index("--founder") + 1]]
    out_path = None
    if "--json" in args:
        out_path = Path(args[args.index("--json") + 1])

    llm = get_llm()
    if llm.mode != "claude":
        print("This audit needs an LLM and none is configured.")
        print("  Set ANTHROPIC_API_KEY in .env, then re-run.")
        print("  (String matching cannot do this job: the corpus contains")
        print("   'Tosi and La' for Tosin Eniolorunda and 'Helen' for Halan.)")
        return 1

    engine = get_engine()
    print(f"auditing with {llm.label}\n")
    try:
        result = audit_corpus(
            engine, llm, founders=founders,
            report=lambda p, m: print(f"  {m}", flush=True),
        )
    except LLMUnavailable as e:
        print(f"LLM unavailable: {e}")
        return 1

    print(f"\naudited {result['audited']} source(s)")
    c = result["counts"]
    print(f"  founder speaks:    {c.get('speaker', 0)}")
    print(f"  about the founder: {c.get('discussed', 0)}")
    print(f"  founder absent:    {c.get('absent', 0)}")
    if result["failed"]:
        print(f"  audit errors:      {len(result['failed'])}")

    absent = [r for r in result["results"] if r["verdict"] == "absent"]
    if absent:
        print("\nMIS-ATTRIBUTED:")
        for r in absent:
            print(f"  [{r['founder']:<19}] {r['source_title'][:52]}")
            print(f"      confidence={r['confidence']} · actually: {r['identified_as'][:60]}")
            print(f"      {r['reason'][:140]}")
            if r["evidence"]:
                print(f"      evidence: {r['evidence'][:120]!r}")

    low = [r for r in result["results"] if r["confidence"] == "low"]
    if low:
        print(f"\nlow confidence (needs a human): {len(low)}")
        for r in low[:8]:
            print(f"  [{r['founder']:<19}] {r['source_title'][:48]} -> {r['verdict']}")

    if out_path:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nverdicts written to {out_path}")

    removable = result["removable"]
    if not removable:
        print("\nNothing confidently mis-attributed.")
        return 0
    if not remove:
        print(f"\n{len(removable)} source(s) would be removed. Re-run with --remove.")
        return 0

    for r in removable:
        engine.delete_source(r["source_id"])
        print(f"removed {r['source_id']} ({r['founder']}: {r['identified_as'][:40]})")
    stats = engine.store.stats()
    print(f"\ncorpus now {stats['chunks']} chunks / {stats['sources']} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
