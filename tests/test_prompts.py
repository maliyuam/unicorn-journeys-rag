"""Keep the prompt catalogue honest.

The point of `backend/prompts.py` is that one file answers "what does this
system actually ask the model?". That promise breaks quietly: someone adds a
prompt inline in a module, or adds one to `prompts.py` and never documents it,
and the single-source-of-truth claim becomes false without anything failing.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import prompts

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
PROMPT_DOC = ROOT / "docs" / "prompts.md"


def prompt_names() -> list[str]:
    return [n for n in dir(prompts) if n.isupper() and isinstance(getattr(prompts, n), str)]


def test_prompts_module_exposes_the_expected_set():
    expected = {
        "SOURCE_CAVEATS", "NARRATIVE_SYSTEM", "ANSWER_SYSTEM", "EXPANSION_SYSTEM",
        "FAITHFULNESS_SYSTEM", "CCS_SYSTEM", "FILTER_SYSTEM", "ATTRIBUTION_SYSTEM",
        "GROUNDING_SYSTEM",
    }
    assert expected <= set(prompt_names())


def test_every_prompt_is_documented():
    doc = PROMPT_DOC.read_text(encoding="utf-8")
    missing = [n for n in prompt_names() if n not in doc]
    assert not missing, (
        f"prompts absent from docs/prompts.md: {missing}. The file is meant to be "
        "the complete catalogue."
    )


def test_every_prompt_is_listed_in_show_prompts():
    """`scripts/show_prompts.py` is how people read the prompts without code."""
    script = (ROOT / "scripts" / "show_prompts.py").read_text(encoding="utf-8")
    missing = [n for n in prompt_names() if n not in script]
    assert not missing, f"prompts missing from the show_prompts catalogue: {missing}"


def test_no_prompt_text_lives_outside_the_prompts_module():
    """No module should define its own system prompt.

    Detects a long triple-quoted constant whose name looks like a prompt, which
    is how prompt text creeps back into individual modules.
    """
    offenders = []
    pattern = re.compile(
        r'^_?[A-Z][A-Z0-9_]*(?:SYSTEM|PROMPT|CAVEATS)\s*=\s*\(?f?"""', re.MULTILINE
    )
    for path in sorted(BACKEND.glob("*.py")):
        if path.name == "prompts.py":
            continue
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(0)[:48]}")
    assert not offenders, (
        "prompt text defined outside backend/prompts.py:\n  " + "\n  ".join(offenders)
    )


def test_source_caveats_reaches_the_prompts_that_read_passages():
    """The transcript caveats must actually be interpolated, not just defined.

    They exist to stop two observed failures: inventing a person out of an ASR
    error, and attributing a host's words to the founder.
    """
    marker = "Speech recognition errors are common"
    assert marker in prompts.SOURCE_CAVEATS
    for name in ("NARRATIVE_SYSTEM", "ANSWER_SYSTEM", "GROUNDING_SYSTEM"):
        assert marker in getattr(prompts, name), f"{name} lost SOURCE_CAVEATS"


def test_grounding_prompt_states_the_rules_that_matter():
    """These instructions are the difference between a gate and a rubber stamp."""
    text = prompts.GROUNDING_SYSTEM.lower()
    assert "do not use knowledge from outside the passages" in text
    assert "answer false" in text  # refuse when uncertain
