"""Connector tests that need no network: relevance heuristics, caption-track
selection, RSS parsing from raw XML, and YouTube auto-caption VTT cleaning."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.connectors import (
    ConnectorError,
    _pick_caption_url,
    _pick_proxy_caption,
    check_youtube_access,
    fetch_rss,
    looks_blocked,
    score_candidates,
    search_youtube,
)
from backend.ingest import parse_captions
from backend.llm import OfflineLLM


def _cand(id, title, channel="Some Channel", description=""):
    return {
        "kind": "youtube", "id": id, "url": f"https://youtu.be/{id}",
        "title": title, "channel": channel, "description": description,
        "published": "", "duration_sec": None,
    }


def test_heuristic_filter_matches_names_and_interviews():
    """Surname-only matches are deliberately rejected.

    Interview-style words appear in almost every description, so accepting
    "surname + interview words" was effectively accepting any surname — which
    is how another person's interview entered the corpus. The trade is
    explicit: a surname needs a company signal to pass, losing some genuine
    surname-only titles rather than admitting the wrong person. The LLM
    filter, when credentials are available, judges these cases properly.
    """
    llm = OfflineLLM()
    cands = score_candidates(
        llm, "Mitchell Elegbe", "Interswitch",
        [
            _cand("a", "Mitchell Elegbe on building Interswitch"),
            _cand("b", "Elegbe keynote at payments summit"),
            _cand("c", "Top 10 richest musicians in Lagos"),
            _cand("d", "Cooking with Mitchell"),  # first name only
            _cand("e", "Elegbe on the Interswitch story"),  # surname + company
        ],
    )
    verdicts = {c["id"]: c["relevant"] for c in cands}
    assert verdicts["a"] is True
    assert verdicts["b"] is False         # surname alone is not enough
    assert verdicts["c"] is False
    assert verdicts["d"] is False         # first name only, no company
    assert verdicts["e"] is True          # surname + company clears the bar
    assert all(c["filter_engine"].startswith("keyword") for c in cands)


def test_pick_caption_url_prefers_english_vtt():
    tracks = {
        "fr": [{"ext": "vtt", "url": "http://x/fr.vtt"}],
        "en": [{"ext": "srv3", "url": "http://x/en.srv3"},
               {"ext": "vtt", "url": "http://x/en.vtt"}],
    }
    assert _pick_caption_url(tracks) == "http://x/en.vtt"
    assert _pick_caption_url({"de": [{"ext": "vtt", "url": "http://x/de.vtt"}]}) \
        == "http://x/de.vtt"
    assert _pick_caption_url({"en": [{"ext": "srv3", "url": "u"}]}) is None


BLOCK_ERROR = (
    "Could not retrieve a transcript for the video https://www.youtube.com/watch?v=x! "
    "This is most likely caused by: YouTube is blocking requests from your IP. "
    "This usually is due to one of the following reasons: too many requests."
)


def test_connectivity_check_detects_a_block_past_the_display_cutoff(monkeypatch):
    """Block markers sit deep inside a long library error. Truncating each
    error for display BEFORE testing for markers reported blocked=False on a
    genuinely blocked network."""
    import backend.connectors as conn

    for name in ("_captions_via_transcript_api", "_captions_via_invidious",
                 "_captions_via_piped"):
        monkeypatch.setattr(conn, name, lambda vid: (_ for _ in ()).throw(
            RuntimeError(BLOCK_ERROR)))
    monkeypatch.setattr(conn, "_captions_via_ytdlp", lambda vid: (_ for _ in ()).throw(
        RuntimeError(BLOCK_ERROR)))

    result = check_youtube_access("someid")
    assert result["ok"] is False
    assert result["blocked"] is True, "block hidden behind the display truncation"
    assert "cookies.txt" in result["hint"]
    assert len(result["detail"]) < 800, "detail should stay readable"


def test_connectivity_check_reports_success(monkeypatch):
    import backend.connectors as conn

    monkeypatch.setattr(conn, "_captions_via_transcript_api",
                        lambda vid: "a real transcript " * 20)
    result = check_youtube_access("someid")
    assert result["ok"] is True and result["method"] == "transcript API"


def test_looks_blocked_matches_both_apostrophe_forms():
    assert looks_blocked("Sign in to confirm you're not a bot")
    assert looks_blocked("Sign in to confirm you’re not a bot")
    assert not looks_blocked("video unavailable")


def test_filter_rejects_a_different_person_with_the_same_surname():
    """The most damaging filter error. "Dr Omobola Johnson (TLcom)" was filed
    under Jeremy Johnson of Andela and put 29 chunks about someone else's
    career in his corpus."""
    llm = OfflineLLM()
    cands = score_candidates(llm, "Jeremy Johnson", "Andela", [
        _cand("bad", "The New Rules of African Tech | Dr Omobola Johnson (TLcom)",
              channel="The Wimbart Way",
              description="A conversation about storytelling, trust and due diligence."),
        _cand("good", "Jeremy Johnson on scaling Andela across Africa",
              description="Interview with the Andela co-founder."),
    ])
    verdicts = {c["id"]: c["relevant"] for c in cands}
    assert verdicts["bad"] is False, "name collision leaked into the corpus"
    assert verdicts["good"] is True, "the genuine interview was rejected"
    reason = next(c["reason"] for c in cands if c["id"] == "bad")
    assert "different person" in reason


def test_filter_accepts_shortened_given_names_and_initials():
    """"Iyin 'E' Aboyeji" is the same person as Iyinoluwa Aboyeji — a
    collision guard that rejected shortened forms would lose real sources."""
    llm = OfflineLLM()
    cands = score_candidates(llm, "Iyinoluwa Aboyeji", "Andela Flutterwave", [
        _cand("short", "Iyin Aboyeji on Andela and Flutterwave",
              description="Founder interview."),
        _cand("initial", "Iyin E Aboyeji : Tech Press Is Over",
              description="Andela co-founder on the African tech press."),
    ])
    for c in cands:
        assert c["relevant"] is True, f"{c['id']} wrongly rejected: {c['reason']}"


def test_surname_plus_interview_words_alone_is_not_enough():
    """Interview-style words appear in nearly every podcast description, so
    surname + those words collapsed to a bare surname match."""
    llm = OfflineLLM()
    cands = score_candidates(llm, "Felix Ike", "Moniepoint", [
        _cand("weak", "A fireside chat about payments",
              description="An interview with a founder named Ike about markets."),
    ])
    assert cands[0]["relevant"] is False


def test_pick_proxy_caption_handles_camelcase_language_field():
    """Instances disagree: some return language_code, others languageCode and
    null the former. Reading only one spelling made every track look non-English."""
    tracks = [
        {"languageCode": "fr", "language_code": None, "label": "French", "url": "/fr"},
        {"languageCode": "en", "language_code": None, "label": "English", "url": "/en"},
    ]
    assert _pick_proxy_caption(tracks, "language_code", "auto")["url"] == "/en"


def test_pick_proxy_caption_prefers_english_manual():
    # Invidious shape: language_code + label
    tracks = [
        {"language_code": "fr", "label": "French", "url": "/fr"},
        {"language_code": "en", "label": "English (auto-generated)", "url": "/en-auto"},
        {"language_code": "en", "label": "English", "url": "/en"},
    ]
    assert _pick_proxy_caption(tracks, "language_code", "auto")["url"] == "/en"
    # falls back to English auto when no manual English exists
    assert _pick_proxy_caption(tracks[:2], "language_code", "auto")["url"] == "/en-auto"
    # falls back to first track when no English at all
    assert _pick_proxy_caption(tracks[:1], "language_code", "auto")["url"] == "/fr"
    assert _pick_proxy_caption([], "language_code", "auto") is None
    # Piped shape: code + autoGenerated flag
    piped = [
        {"code": "en", "autoGenerated": True, "url": "p-auto"},
        {"code": "en", "autoGenerated": False, "url": "p-man"},
    ]
    assert _pick_proxy_caption(piped, "code", "auto")["url"] == "p-man"


def test_search_strategy_chain_falls_through(monkeypatch):
    import backend.connectors as conn

    calls = []

    def fail_search(query, n):
        calls.append("yt-dlp")
        raise RuntimeError("boom")

    def ok_invidious(query, n):
        calls.append("invidious")
        return [{"kind": "youtube", "id": "x", "url": "u", "title": "t",
                 "channel": "c", "published": "", "description": "", "duration_sec": 1}]

    monkeypatch.setattr(conn, "_yt_dlp_search", fail_search)
    monkeypatch.setattr(conn, "_invidious_search", ok_invidious)
    monkeypatch.setattr(conn.config, "YOUTUBE_API_KEY", "")
    results = search_youtube("anything", 5)
    assert calls == ["yt-dlp", "invidious"]
    assert results[0]["id"] == "x"


def test_fetch_rss_parses_raw_xml():
    xml = """<?xml version="1.0"?>
    <rss version="2.0"><channel><title>Founder Talks</title>
      <item><title>Ep 1: Banking the unbanked</title>
        <link>https://pod.example/ep1</link>
        <description>A chat about &lt;b&gt;fintech&lt;/b&gt; in Egypt.</description>
        <enclosure url="https://pod.example/ep1.mp3" type="audio/mpeg" length="1"/>
      </item>
      <item><title>Ep 2: No audio here</title>
        <link>https://pod.example/ep2</link>
        <description>Text only.</description>
      </item>
    </channel></rss>"""
    episodes = fetch_rss(xml, max_items=10)
    assert len(episodes) == 2
    assert episodes[0]["channel"] == "Founder Talks"
    assert episodes[0]["audio_url"] == "https://pod.example/ep1.mp3"
    assert "<b>" not in episodes[0]["description"]
    assert episodes[1]["audio_url"] is None


def test_parse_youtube_autocaption_vtt_with_word_timing():
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
so<00:00:00.480><c> today</c><00:00:00.880><c> we</c><00:00:01.280><c> talk</c>

00:00:02.000 --> 00:00:04.000
so today we talk
about African fintech
"""
    text = parse_captions(vtt)
    assert "<c>" not in text and "00:00" not in text
    assert text.count("so today we talk") == 1
    assert "about African fintech" in text


def test_fetch_rss_refuses_non_http_locators():
    """SSRF / local-file guard.

    `feed_url` comes straight from an API caller and feedparser will happily
    treat it as a local path or a file:// URL. Only http(s) locators may cause
    a fetch; inline feed XML stays allowed because it is parsed, not fetched.
    """
    for hostile in (
        "file:///etc/passwd",
        "file://C:/Windows/win.ini",
        "/etc/passwd",
        r"C:\Windows\win.ini",
        "ftp://example.com/feed.xml",
        "http://169.254.169.254/latest/meta-data/".replace("http://", "gopher://"),
    ):
        with pytest.raises(ConnectorError, match="http"):
            fetch_rss(hostile)

    # inline XML is still accepted — it is content, not a locator
    assert fetch_rss("<rss version='2.0'><channel><title>T</title></channel></rss>") == []
