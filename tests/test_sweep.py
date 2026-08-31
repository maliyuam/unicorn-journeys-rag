"""Sweep tests: roster loading, per-founder result shape, and the
result-key collision that once clobbered the sweep report."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.llm import OfflineLLM
from backend.sweep import load_roster, sweep

LONG_TEXT = (
    "The founder launched the company in 2016 after leaving a large bank. "
    "They raised a Series A of twelve million dollars in 2019. "
    "Regulators were slow to approve the licence, which delayed expansion. "
) * 8


def test_roster_has_the_papers_sixteen_founders():
    roster = load_roster()
    names = {f["name"] for f in roster}
    from_paper = [f for f in roster if not f.get("added_beyond_paper")]
    assert len(from_paper) == 16, "the paper's cohort must stay at sixteen founders"
    # spot-check one founder per unicorn in the study
    for expected in [
        "Mounir Nakhla",        # MNT-Halan
        "Tosin Eniolorunda",    # Moniepoint
        "Coenraad Jonker",      # Tyme
        "Olugbenga Agboola",    # Flutterwave
        "Mitchell Elegbe",      # Interswitch
        "Jeremy Johnson",       # Andela
    ]:
        assert expected in names
    assert all(f.get("company") for f in roster)
    assert len(names) == len(roster), "roster must not contain duplicate names"
    # additions beyond the paper's 2024 cohort
    assert {"Ladi Delano", "Jide Odunsi"} <= names, "Moove founders missing"
    moove = [f for f in roster if f["company"] == "Moove"]
    assert len(moove) == 2 and all(f["inception"] == 2020 for f in moove)


def test_podcast_discovery_rejects_substring_name_collisions(monkeypatch):
    """A surname inside another word must not match. "Ike" appears in "like"
    and "strike" — naive substring matching filed an NGO-funding episode
    under Felix Ike's name and put wrong facts in his corpus."""
    import backend.sweep as sw

    episodes = [
        {"kind": "rss", "id": "1", "url": "u", "audio_url": "a.mp3",
         "title": "How to Fund a Hospital in a Warzone", "channel": "The Flip",
         "description": "Aid money and how it moves. People would like to strike deals.",
         "published": "", "duration_sec": None},
        {"kind": "rss", "id": "2", "url": "u", "audio_url": "a.mp3",
         "title": "Felix Ike on building Moniepoint's agent network",
         "channel": "The Flip", "description": "An interview with the co-founder.",
         "published": "", "duration_sec": None},
    ]
    monkeypatch.setattr(sw, "fetch_rss", lambda url, max_items=40: episodes)
    relevant = sw.discover_podcast_episodes(
        OfflineLLM(), "Felix Ike", "Moniepoint",
        feeds=[{"name": "The Flip", "feed_url": "http://x"}],
    )
    ids = [e["id"] for e in relevant]
    assert "1" not in ids, "substring collision leaked into the corpus"
    assert "2" in ids, "the genuine interview was rejected"


def test_podcast_discovery_survives_a_dead_feed(monkeypatch):
    """One unreachable feed must not abort the scan across the others."""
    import backend.sweep as sw

    good = [{"kind": "rss", "id": "9", "url": "u", "audio_url": "a.mp3",
             "title": "Iyinoluwa Aboyeji interview", "channel": "c",
             "description": "Andela co-founder", "published": "", "duration_sec": None}]

    def flaky(url, max_items=40):
        if "dead" in url:
            raise RuntimeError("connection refused")
        return good

    monkeypatch.setattr(sw, "fetch_rss", flaky)
    out = sw.discover_podcast_episodes(
        OfflineLLM(), "Iyinoluwa Aboyeji", "Andela",
        feeds=[{"name": "dead", "feed_url": "http://dead"},
               {"name": "live", "feed_url": "http://live"}],
    )
    assert [e["id"] for e in out] == ["9"]


def test_podcast_discovery_skips_episodes_without_audio(monkeypatch):
    import backend.sweep as sw

    monkeypatch.setattr(sw, "fetch_rss", lambda url, max_items=40: [
        {"kind": "rss", "id": "n", "url": "u", "audio_url": None,
         "title": "Iyinoluwa Aboyeji interview", "channel": "c",
         "description": "no audio enclosure", "published": "", "duration_sec": None}
    ])
    out = sw.discover_podcast_episodes(
        OfflineLLM(), "Iyinoluwa Aboyeji", "Andela",
        feeds=[{"name": "x", "feed_url": "http://x"}],
    )
    assert out == [], "an episode with no audio cannot be transcribed"


@pytest.fixture()
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(config, "MONGODB_URI", "")
    from backend.engine import Engine

    return Engine()


def test_sweep_reports_per_founder_results(engine, monkeypatch):
    """The result's "founders" key must stay the per-founder list — the
    store's stats() also has a "founders" key and previously overwrote it."""
    import backend.sweep as sw

    def fake_search(query, n):
        # distinct video per founder — a shared id would (correctly) be
        # deduped by source_id and only ingest once
        vid = "elegbe" if "Elegbe" in query else "eniolorunda"
        return [
            {
                "kind": "youtube", "id": vid, "url": "u", "title": "An interview",
                "channel": "c", "published": "", "description": "", "duration_sec": 60,
            }
        ]

    def fake_fetch(video_id, fallback_meta=None):
        return {
            "title": "An interview", "channel": "c", "published": "",
            "text": LONG_TEXT, "method": "captions (test)",
        }

    monkeypatch.setattr(sw, "search_youtube", fake_search)
    monkeypatch.setattr(sw, "fetch_youtube_transcript", fake_fetch)
    monkeypatch.setattr(sw, "score_candidates",
                        lambda llm, n, c, cands: [{**x, "relevant": True} for x in cands])
    monkeypatch.setattr(sw, "PAUSE_BETWEEN_FOUNDERS_SEC", 0)

    messages = []
    result = sweep(
        engine, OfflineLLM(), ["Mitchell Elegbe", "Tosin Eniolorunda"],
        max_candidates=3, search_results=5,
        report=lambda p, m: messages.append(m),
        # hermetic: no live preflight, no podcast feed fetches
        include_podcasts=False, preflight=False,
    )

    assert isinstance(result["founders"], list), "per-founder results were clobbered"
    assert [f["founder"] for f in result["founders"]] == [
        "Tosin Eniolorunda", "Mitchell Elegbe",
    ] or {f["founder"] for f in result["founders"]} == {
        "Mitchell Elegbe", "Tosin Eniolorunda",
    }
    assert all(f["ingested"] == 1 for f in result["founders"])
    assert result["chunks"] > 0
    assert result["founders_indexed"] == 2
    assert any("searching" in m for m in messages)


def test_sweep_skips_already_ingested_sources(engine, monkeypatch):
    import backend.sweep as sw

    monkeypatch.setattr(sw, "PAUSE_BETWEEN_FOUNDERS_SEC", 0)
    monkeypatch.setattr(sw, "search_youtube", lambda q, n: [
        {"kind": "youtube", "id": "dup", "url": "u", "title": "t", "channel": "c",
         "published": "", "description": "", "duration_sec": 1}
    ])
    monkeypatch.setattr(sw, "score_candidates",
                        lambda llm, n, c, cands: [{**x, "relevant": True} for x in cands])
    monkeypatch.setattr(sw, "fetch_youtube_transcript", lambda vid, fallback_meta=None: {
        "title": "t", "channel": "c", "published": "", "text": LONG_TEXT, "method": "m"})

    kw = {"include_podcasts": False, "preflight": False}
    first = sweep(engine, OfflineLLM(), ["Mitchell Elegbe"], 3, 5, lambda p, m: None, **kw)
    assert first["founders"][0]["ingested"] == 1
    second = sweep(engine, OfflineLLM(), ["Mitchell Elegbe"], 3, 5, lambda p, m: None, **kw)
    assert second["founders"][0]["approved"] == 0, "already-ingested source re-fetched"
