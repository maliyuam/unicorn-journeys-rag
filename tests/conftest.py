"""Test-wide isolation.

Two guarantees every test in this suite depends on, applied here rather than
repeated per-file (where they were previously easy to forget):

1. **No configured MongoDB is touched.** A developer with `MONGODB_URI` in
   their `.env` would otherwise run the suite against their real corpus —
   `replace_all()` starts with `delete_many({})`, so a stray `Engine()` in a
   test can empty a live collection.

2. **No Anthropic credentials are spent.** `get_llm()` probes for credentials
   with a real API call. With a key in the environment, running the tests
   would bill the developer and make results depend on a model's mood.

Note what is deliberately *not* forced here: the embedding backend. Several
tests exercise incremental-embedding behaviour that only fixed-dimension
backends have, so pinning everything to TF-IDF would quietly stop testing it.
The consequence is that a cold machine downloads the ~130 MB fastembed model
once (CI caches it) — so the suite needs no credentials and no database, but
it is not network-free on first run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config


@pytest.fixture(autouse=True)
def _isolate_from_real_services(monkeypatch):
    monkeypatch.setattr(config, "MONGODB_URI", "", raising=False)

    from backend import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_credentials_might_exist", lambda: False, raising=False)
    monkeypatch.setattr(llm_mod, "_llm_instance", None, raising=False)
    monkeypatch.setattr(llm_mod, "_offline_since", 0.0, raising=False)
    yield
