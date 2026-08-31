"""Find sources whose transcript never mentions the founder or their company.

The relevance filter judges a video from its title, channel and description —
all it has before fetching. Sometimes the transcript then turns out to be
about someone else entirely: a panel the founder was billed for but never
spoke on, or a different person with the same surname. Such a source cannot
support any claim about that founder, so it is noise at best and false
attribution at worst.

This checks the transcripts themselves, which the pre-fetch filter never sees.

**Treat the output as a review queue, not a delete list.** Speech recognition
mangles exactly the words this check relies on — observed in this corpus:
"Tosin Eniolorunda" transcribed as "Tosi and La", "Moniepoint" as "moneyo Inc
money point", "Halan" as "Helen". Companies also rename (Moniepoint was
TeamApt), so a current name legitimately appears nowhere in an older
recording. Every flag needs a human (or an LLM) to read the opening lines
before anything is removed.

    python scripts/audit_attribution.py            # report only (default)
    python scripts/audit_attribution.py --remove   # only after reviewing
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engine import get_engine  # noqa: E402

# Ignore short/generic company tokens that would match almost any transcript.
STOP_COMPANY_TOKENS = {"the", "and", "group", "bank", "capital", "africa",
                       "african", "tech", "limited", "ltd", "inc", "company"}


def company_tokens(company: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z]+", (company or "").lower())
        if len(t) > 3 and t not in STOP_COMPANY_TOKENS
    ]


def mentions(body: str, needle: str) -> int:
    return len(re.findall(rf"\b{re.escape(needle)}", body, re.I))


def main(remove: bool = False) -> int:
    engine = get_engine()
    by_source: dict[str, list[dict]] = {}
    for c in engine.store.all_chunks():
        by_source.setdefault(c["source_id"], []).append(c)

    orphans = []
    for s in engine.list_sources():
        parts = s["founder"].split()
        surname = parts[-1].lower() if parts else ""
        body = " ".join(c["text"] for c in by_source.get(s["source_id"], []))
        if not body:
            continue
        surname_hits = mentions(body, surname) + mentions(s["source_title"], surname)
        comp_hits = sum(
            mentions(body, t) + mentions(s["source_title"], t)
            for t in company_tokens(s.get("company", ""))
        )
        if surname_hits == 0 and comp_hits == 0:
            orphans.append((s, len(body)))

    total_sources = len(engine.list_sources())
    print(f"checked {total_sources} sources")
    print(f"transcripts naming neither the founder nor their company: {len(orphans)}\n")
    for s, size in orphans:
        print(f"  [{s['founder']:<19}] {s['source_title'][:54]}")
        print(f"      {s['chunks']} chunks / {size} chars — {s['source_id']}")
        snippet = " ".join(
            by_source[s["source_id"]][0]["text"].split()[:24]
        )
        print(f"      opens: {snippet!r}")

    if not orphans:
        print("Nothing flagged.")
        return 0
    if not remove:
        print("\nREVIEW THESE — do not bulk-delete. Speech recognition garbles")
        print("names ('Tosi and La' for Tosin Eniolorunda, 'Helen' for Halan) and")
        print("companies rename (Moniepoint was TeamApt), so genuine sources are")
        print("flagged here. Read the opening lines above, then remove only what")
        print("is clearly someone else, with --remove.")
        return 0

    for s, _ in orphans:
        engine.delete_source(s["source_id"])
        print(f"removed {s['source_id']}")
    stats = engine.store.stats()
    print(f"\ncorpus now {stats['chunks']} chunks / {stats['sources']} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main(remove="--remove" in sys.argv))
