"""Print every prompt this system sends to a model.

    python scripts/show_prompts.py              # list them with a summary
    python scripts/show_prompts.py --full       # print the full text of each
    python scripts/show_prompts.py ANSWER_SYSTEM  # print one

Reads `backend/prompts.py` directly, so what it prints is what actually gets
sent — no copy in a document to drift out of date. To change any of them, edit
`backend/prompts.py`; see `docs/prompts.md` for what each one is for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import prompts  # noqa: E402

# name -> (module that sends it, when it fires, what happens with no credentials)
CATALOGUE = {
    "EXPANSION_SYSTEM": (
        "retrieval.py",
        "every question, to widen it into sub-queries",
        "deterministic themed templates",
    ),
    "GROUNDING_SYSTEM": (
        "grounding.py",
        "every question, to decide whether to answer at all",
        "similarity + word-coverage + cross-encoder heuristic",
    ),
    "ANSWER_SYSTEM": (
        "generation.py",
        "answering a question from retrieved passages",
        "extractive: the most relevant sentences, quoted",
    ),
    "NARRATIVE_SYSTEM": (
        "generation.py",
        "writing a founder's full journey",
        "extractive summary per theme",
    ),
    "FAITHFULNESS_SYSTEM": (
        "evaluation.py",
        "judging a draft claim by claim",
        "lexical overlap check against the passages",
    ),
    "CCS_SYSTEM": (
        "evaluation.py",
        "scoring Context Completeness across the six themes",
        "keyword presence per theme",
    ),
    "FILTER_SYSTEM": (
        "connectors.py",
        "deciding whether a search result is really this founder",
        "name + company keyword heuristic",
    ),
    "ATTRIBUTION_SYSTEM": (
        "attribution.py",
        "auditing whether a stored source is the right person",
        "string audit only (a review queue, never a delete list)",
    ),
    "SOURCE_CAVEATS": (
        "shared",
        "interpolated into the prompts that read passages",
        "n/a - it is a fragment, not a prompt",
    ),
}


def _names() -> list[str]:
    return [n for n in dir(prompts) if n.isupper() and isinstance(getattr(prompts, n), str)]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    full = "--full" in sys.argv

    available = _names()
    if args:
        for name in args:
            if name not in available:
                print(f"unknown prompt {name!r}. Available: {', '.join(sorted(available))}",
                      file=sys.stderr)
                return 1
            print(f"{'=' * 78}\n{name}\n{'=' * 78}")
            print(getattr(prompts, name))
            print()
        return 0

    print(f"{len(available)} prompts in backend/prompts.py — edit that file to change them.\n")
    for name in sorted(available):
        sender, when, offline = CATALOGUE.get(name, ("?", "?", "?"))
        text = getattr(prompts, name)
        print(f"{name}")
        print(f"  sent by     backend/{sender}")
        print(f"  fires       {when}")
        print(f"  no creds    {offline}")
        print(f"  length      {len(text)} chars")
        if full:
            print("  " + "-" * 70)
            for line in text.splitlines():
                print(f"  | {line}")
        else:
            first = next((ln for ln in text.splitlines() if ln.strip()), "")
            print(f"  starts      {first[:66]}...")
        print()

    undocumented = set(available) - set(CATALOGUE)
    if undocumented:
        print(f"NOTE: not in this script's catalogue: {', '.join(sorted(undocumented))}")
        print("Add them here and in docs/prompts.md.")
    if not full:
        print("Run with --full to print the complete text of every prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
