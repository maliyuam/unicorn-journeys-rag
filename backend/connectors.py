"""Discovery connectors: YouTube search and podcast RSS feeds.

Pipeline (mirrors the paper's methodology, upgraded):
  1. search    — YouTube Data API v3 when YOUTUBE_API_KEY is set (the paper's
                 approach), else yt-dlp keyless search; RSS via feedparser.
  2. filter    — relevance verdict per candidate: Claude with a structured
                 schema (the paper's Appendix-A prompt, made machine-readable)
                 or a deterministic keyword heuristic offline.
  3. review    — HUMAN IN THE LOOP: candidates are returned to the UI for
                 approval; nothing is ingested without an explicit selection.
  4. fetch     — captions via yt-dlp (manual first, then auto-generated),
                 or local ASR (backend/asr.py) for audio-only sources.
  5. ingest    — through the production upsert path (idempotent, hashed).
"""
from __future__ import annotations

import hashlib
import logging
import re

import requests

from . import config
from .ingest import parse_captions
from .llm import LLMUnavailable
from .retrieval import tokenize

log = logging.getLogger("unicorn_rag.connectors")


class ConnectorError(RuntimeError):
    pass


# --- search: YouTube ---------------------------------------------------------

def _yt_api_search(query: str, max_results: int) -> list[dict]:
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results, 50),
            "key": config.YOUTUBE_API_KEY,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise ConnectorError(f"YouTube API error {resp.status_code}: {resp.text[:200]}")
    out = []
    for item in resp.json().get("items", []):
        vid = item["id"].get("videoId")
        if not vid:
            continue
        sn = item["snippet"]
        out.append(
            {
                "kind": "youtube",
                "id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": sn.get("title", ""),
                "channel": sn.get("channelTitle", ""),
                "published": sn.get("publishedAt", ""),
                "description": sn.get("description", ""),
                "duration_sec": None,
            }
        )
    return out


def _ydl_opts(**extra) -> dict:
    opts = {"quiet": True, "no_warnings": True, **extra}
    if config.YTDLP_COOKIES_FILE:
        opts["cookiefile"] = config.YTDLP_COOKIES_FILE
    elif config.YTDLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.YTDLP_COOKIES_FROM_BROWSER,)
    return opts


_BLOCK_MARKERS = ("confirm you're not a bot", "confirm you’re not a bot",
                  "blocking requests from your ip", "ipblocked", "requestblocked")


def looks_blocked(error_text: str) -> bool:
    return any(m in error_text.lower() for m in _BLOCK_MARKERS)


def _block_hint(error_text: str) -> str:
    if looks_blocked(error_text):
        return (
            " — YouTube appears to be bot-checking this network's IP. Fix: set "
            "YTDLP_COOKIES_FROM_BROWSER=chrome (or edge/firefox) in .env so "
            "requests authenticate with your browser session, or run from a "
            "non-blocked network."
        )
    return ""


def _cookie_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "cookie" in msg or "dpapi" in msg or "decrypt" in msg


def _extract(target: str, opts_extra: dict, download: bool = False):
    """yt-dlp extract_info with cookie-failure resilience: if configured
    browser cookies can't be read (locked DB, app-bound encryption), retry
    once without them rather than failing the whole operation."""
    from yt_dlp import YoutubeDL

    opts = _ydl_opts(**opts_extra)
    try:
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(target, download=download)
    except Exception as e:
        has_cookies = "cookiefile" in opts or "cookiesfrombrowser" in opts
        if has_cookies and _cookie_error(e):
            log.warning(
                "browser-cookie extraction failed (%s) — retrying without cookies",
                str(e)[:100],
            )
            bare = {k: v for k, v in opts.items()
                    if k not in ("cookiefile", "cookiesfrombrowser")}
            with YoutubeDL(bare) as ydl:
                return ydl.extract_info(target, download=download)
        raise


def _yt_dlp_search(query: str, max_results: int) -> list[dict]:
    info = _extract(
        f"ytsearch{max_results}:{query}",
        {"extract_flat": True, "skip_download": True},
    )
    out = []
    for entry in (info or {}).get("entries", []) or []:
        if not entry or not entry.get("id"):
            continue
        out.append(
            {
                "kind": "youtube",
                "id": entry["id"],
                "url": entry.get("url")
                or f"https://www.youtube.com/watch?v={entry['id']}",
                "title": entry.get("title", ""),
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "published": "",
                "description": (entry.get("description") or "")[:500],
                "duration_sec": entry.get("duration"),
            }
        )
    return out


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UnicornRAG/2.1"}


def _invidious_search(query: str, n: int) -> list[dict]:
    errors = []
    for inst in config.INVIDIOUS_INSTANCES:
        base = inst.rstrip("/")
        try:
            resp = requests.get(
                f"{base}/api/v1/search",
                params={"q": query, "type": "video"},
                headers=_UA,
                timeout=12,
            )
            if resp.status_code != 200:
                errors.append(f"{inst}: HTTP {resp.status_code}")
                continue
            out = []
            for item in resp.json():
                vid = item.get("videoId")
                if not vid or item.get("type", "video") != "video":
                    continue
                out.append(
                    {
                        "kind": "youtube",
                        "id": vid,
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "title": item.get("title", ""),
                        "channel": item.get("author", ""),
                        "published": item.get("publishedText", ""),
                        "description": (item.get("description") or "")[:500],
                        "duration_sec": item.get("lengthSeconds"),
                    }
                )
                if len(out) >= n:
                    break
            if out:
                return out
            errors.append(f"{inst}: 0 results")
        except Exception as e:
            errors.append(f"{inst}: {e.__class__.__name__}")
    raise ConnectorError("; ".join(errors))


def search_youtube(query: str, max_results: int | None = None) -> list[dict]:
    """Multi-strategy search, non-API first (user preference):
    1. yt-dlp keyless search
    2. YouTube Data API v3 (only when YOUTUBE_API_KEY is set)
    3. Invidious public instances
    """
    n = max_results or config.DISCOVERY_MAX_RESULTS
    strategies = [("yt-dlp", lambda: _yt_dlp_search(query, n))]
    if config.YOUTUBE_API_KEY:
        strategies.append(("data-api", lambda: _yt_api_search(query, n)))
    strategies.append(("invidious", lambda: _invidious_search(query, n)))

    errors = []
    for name, fn in strategies:
        try:
            results = fn()
            if results:
                log.info("search via %s: %d result(s)", name, len(results))
                return results
            errors.append(f"{name}: no results")
        except Exception as e:
            errors.append(f"{name}: {str(e)[:160]}")
    raise ConnectorError("all search strategies failed — " + " | ".join(errors))


# --- search: RSS -------------------------------------------------------------

def search_podcasts(term: str, limit: int = 10) -> list[dict]:
    """Find podcast RSS feeds by name via Apple's public search API.

    Podcast RSS is the one ingestion channel YouTube's rate limiting cannot
    touch: feeds are plain XML and episode audio is served by the publisher's
    own CDN. Requiring users to already know a feed URL was the only thing
    making that channel hard to use.
    """
    resp = requests.get(
        "https://itunes.apple.com/search",
        params={"term": term, "entity": "podcast", "limit": min(limit, 50)},
        headers=_UA,
        timeout=20,
    )
    if resp.status_code != 200:
        raise ConnectorError(f"podcast search failed: HTTP {resp.status_code}")
    out = []
    for item in resp.json().get("results", []):
        feed = item.get("feedUrl")
        if not feed:
            continue
        out.append(
            {
                "name": item.get("collectionName", ""),
                "feed_url": feed,
                "publisher": item.get("artistName", ""),
                "episodes": item.get("trackCount"),
                "genre": item.get("primaryGenreName", ""),
            }
        )
    return out


def fetch_rss(feed_url: str, max_items: int | None = None) -> list[dict]:
    import feedparser

    # feedparser accepts a URL, a local path, or raw feed content. Inline
    # content is inert — it is just parsed — but anything treated as a locator
    # causes a *fetch*, and this value arrives straight from an API caller. Left
    # unchecked the endpoint reads local files (file://, bare paths) and proxies
    # requests to whatever the host can reach, cloud metadata included.
    source = feed_url.strip()
    is_inline_document = source.startswith("<")
    if not is_inline_document and not re.match(r"^https?://", source, re.IGNORECASE):
        raise ConnectorError(
            "feed_url must be an http(s) URL (or inline feed XML); "
            "local paths and other schemes are refused"
        )

    n = max_items or config.DISCOVERY_MAX_RESULTS
    parsed = feedparser.parse(feed_url)
    if parsed.bozo and not parsed.entries:
        raise ConnectorError(f"could not parse RSS feed: {parsed.bozo_exception}")
    channel = parsed.feed.get("title", feed_url)
    out = []
    for entry in parsed.entries[:n]:
        audio_url = None
        for enc in entry.get("enclosures", []):
            if "audio" in (enc.get("type") or ""):
                audio_url = enc.get("href")
                break
        link = entry.get("link") or audio_url or ""
        eid = hashlib.sha1((link or entry.get("title", "")).encode()).hexdigest()[:12]
        out.append(
            {
                "kind": "rss",
                "id": eid,
                "url": link,
                "audio_url": audio_url,
                "title": entry.get("title", ""),
                "channel": channel,
                "published": entry.get("published", ""),
                "description": re.sub(r"<[^>]+>", " ", entry.get("summary", ""))[:500],
                "duration_sec": None,
            }
        )
    return out


# --- relevance filter --------------------------------------------------------

_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "relevant", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

_FILTER_SYSTEM = """You screen media candidates for an academic research \
corpus about a specific startup founder. Every candidate you approve will be \
transcribed and mined for facts about that founder's entrepreneurial journey, \
so a wrong approval injects another person's biography into their record.

Mark a candidate RELEVANT when the named founder is a speaker in it, or is \
discussed substantively in it: interviews, fireside chats, keynotes, panels, \
podcast episodes, documentary segments, and company profiles that centre on \
them or the company they built.

Mark it NOT relevant when:
- the match is only a name collision with a different person — this is the \
most damaging error, so weigh the country, industry and company signals, not \
the name alone;
- the founder is a passing mention in a roundup, listicle, or news bulletin \
about something else;
- it is commentary, reaction, or AI-generated content about the founder rather \
than the founder's own account;
- the company is named but the founder is neither present nor the subject.

Judge from the title, channel and description only — you cannot watch the \
video. Titles are often incomplete, so use the description to decide: a \
generic title on a channel that clearly interviews founders, with a \
description naming the person, is relevant. When the evidence genuinely does \
not let you tell, mark it NOT relevant; a missed source costs far less than a \
contaminated corpus.

Give a specific reason per candidate — name the signal you used ("description \
names him as Moove co-founder", "different Ladi Delano, a musician")."""

_INTERVIEW_WORDS = (
    "interview", "podcast", "fireside", "keynote", "panel", "conversation",
    "talks", "founder", "ceo", "story", "journey", "startup",
)


def score_candidates(llm, founder: str, company: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return candidates
    if llm.mode == "claude":
        try:
            listing = "\n".join(
                f"- id={c['id']} | title={c['title']!r} | channel={c['channel']!r}"
                f" | description={c['description'][:200]!r}"
                for c in candidates
            )
            result = llm.complete_json(
                system=_FILTER_SYSTEM,
                user=(
                    f"Founder: {founder}" + (f" (company: {company})" if company else "")
                    + f"\n\nCandidates:\n{listing}"
                ),
                schema=_FILTER_SCHEMA,
                max_tokens=6000,
                effort="low",
            )
            verdicts = {v["id"]: v for v in result.get("verdicts", [])}
            for c in candidates:
                v = verdicts.get(c["id"])
                c["relevant"] = bool(v and v["relevant"])
                c["reason"] = (v or {}).get("reason", "not judged")
                c["filter_engine"] = "claude"
            return candidates
        except (LLMUnavailable, Exception) as e:  # fall through to heuristic
            log.warning("LLM filter failed (%s); using keyword heuristic", e)
    name_tokens = set(tokenize(founder))
    parts = founder.split()
    first_name = parts[0].lower() if parts else ""
    last_name = parts[-1].lower() if parts else ""
    company_tokens = set(tokenize(company)) if company else set()
    for c in candidates:
        hay = f"{c['title']} {c['channel']} {c['description']}".lower()
        hay_tokens = set(tokenize(hay))
        full_name_hit = name_tokens and name_tokens <= hay_tokens
        last_name_hit = last_name and last_name in hay_tokens
        company_hit = bool(company_tokens and company_tokens & hay_tokens)
        interview_hit = any(w in hay for w in _INTERVIEW_WORDS)
        collision = (
            _different_person(hay, first_name, last_name)
            if last_name_hit and not full_name_hit else False
        )
        # Surname + "interview-style content" is far too weak on its own:
        # almost every podcast description contains one of those words, so
        # the rule reduced to a bare surname match and filed an interview
        # with Dr Omobola Johnson under Jeremy Johnson of Andela. Require a
        # company signal, and reject outright when a different given name
        # sits in front of the surname.
        c["relevant"] = bool(
            not collision
            and (full_name_hit or (last_name_hit and company_hit))
        )
        bits = []
        if full_name_hit:
            bits.append("full name match")
        elif last_name_hit:
            bits.append("surname match")
        if company_hit:
            bits.append("company match")
        if interview_hit:
            bits.append("interview-style content")
        if collision:
            bits.append(f"REJECTED: surname belongs to a different person ({collision})")
        c["reason"] = ", ".join(bits) or "no name/company signals in metadata"
        c["filter_engine"] = "keyword heuristic (offline)"
    return candidates


def _different_person(hay: str, first_name: str, last_name: str) -> str | None:
    """Detect "<other given name> <surname>" — a name collision.

    Returns the conflicting given name, or None. An initial ("Iyin 'E'
    Aboyeji") or a shortened form ("Iyin" for "Iyinoluwa") is the same
    person, so those do not count.
    """
    if not last_name:
        return None
    for match in re.finditer(rf"([a-z]+)\s+{re.escape(last_name)}\b", hay):
        prev = match.group(1)
        if prev in _NAME_PREFIXES or len(prev) <= 1:
            continue
        if prev == first_name:
            continue
        if first_name.startswith(prev) or prev.startswith(first_name):
            continue  # shortened or extended form of the same given name
        return prev
    return None


_NAME_PREFIXES = {
    "dr", "mr", "mrs", "ms", "prof", "professor", "sir", "with", "and", "the",
    "by", "of", "featuring", "ft", "feat", "guest", "host", "interview",
}


# --- transcript fetching -----------------------------------------------------

def _pick_caption_url(tracks: dict) -> str | None:
    """Choose an English (or first available) VTT caption track URL."""
    for lang in sorted(tracks, key=lambda t: (not t.startswith("en"), t)):
        for fmt in tracks[lang]:
            if fmt.get("ext") == "vtt" and fmt.get("url"):
                return fmt["url"]
    return None


def _captions_via_transcript_api(video_id: str) -> str:
    """Lightweight InnerTube caption fetch (no full page extraction)."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    listing = api.list(video_id)
    transcript = None
    try:
        transcript = listing.find_manually_created_transcript(["en", "en-US", "en-GB"])
    except Exception:
        try:
            transcript = listing.find_transcript(["en", "en-US", "en-GB"])
        except Exception:
            for t in listing:  # any language beats nothing
                transcript = t
                break
    if transcript is None:
        raise ConnectorError("no transcript tracks")
    text = " ".join(s.text.strip() for s in transcript.fetch().snippets)
    return re.sub(r"\s+", " ", text).strip()


# The web client is what bot-checks hit hardest; Android/iOS player clients
# talk to different endpoints and frequently still work on flagged IPs.
_YTDLP_CLIENT_VARIANTS: tuple[tuple[str, ...] | None, ...] = (
    None, ("android",), ("ios",)
)


def _captions_via_ytdlp(video_id: str) -> tuple[str, dict, str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    last_error: Exception = ConnectorError("no client variants attempted")
    for clients in _YTDLP_CLIENT_VARIANTS:
        label = f"yt-dlp {clients[0]}" if clients else "yt-dlp web"
        # We only want subtitle tracks, never media. Without
        # ignore_no_formats_error yt-dlp aborts the whole extraction with
        # "Requested format is not available" on videos whose streams it
        # cannot resolve — throwing away captions that were right there.
        extra: dict = {
            "skip_download": True,
            "ignore_no_formats_error": True,
            "format": None,
        }
        if clients:
            extra["extractor_args"] = {"youtube": {"player_client": list(clients)}}
        try:
            info = _extract(url, extra)
        except Exception as e:
            last_error = e
            continue
        meta = {
            "title": info.get("title", ""),
            "channel": info.get("channel") or info.get("uploader") or "",
            "published": info.get("upload_date", ""),
        }
        for tracks in (info.get("subtitles") or {}, info.get("automatic_captions") or {}):
            cap_url = _pick_caption_url(tracks)
            if not cap_url:
                continue
            resp = requests.get(cap_url, timeout=60)
            if resp.status_code != 200:
                continue
            text = parse_captions(resp.text)
            if len(text) >= config.MIN_TEXT_CHARS:
                return text, meta, label
        last_error = ConnectorError(f"{label}: no usable caption tracks")
    raise last_error


def _pick_proxy_caption(tracks: list[dict], lang_key: str, auto_marker: str) -> dict | None:
    """Prefer English manual > English auto > first track. Pure helper,
    shared by the Invidious and Piped routes.

    Instances disagree on the language field name — some return
    `language_code`, others `languageCode` (and may null out the other) — so
    check every spelling rather than trusting one.
    """
    lang_keys = (lang_key, "language_code", "languageCode", "code", "lang")

    def is_en(t):
        for key in lang_keys:
            value = t.get(key)
            if value and str(value).lower().startswith("en"):
                return True
        return False

    def is_auto(t):
        label = (t.get("label") or t.get("name") or "").lower()
        return t.get("autoGenerated") is True or auto_marker in label

    for track in tracks:
        if is_en(track) and not is_auto(track):
            return track
    for track in tracks:
        if is_en(track):
            return track
    return tracks[0] if tracks else None


def _captions_via_invidious(video_id: str) -> str:
    errors = []
    for inst in config.INVIDIOUS_INSTANCES:
        base = inst.rstrip("/")
        try:
            resp = requests.get(
                f"{base}/api/v1/captions/{video_id}", headers=_UA, timeout=12
            )
            if resp.status_code != 200:
                errors.append(f"{inst}: HTTP {resp.status_code}")
                continue
            track = _pick_proxy_caption(
                resp.json().get("captions", []), "language_code", "auto"
            )
            if not track or not track.get("url"):
                errors.append(f"{inst}: no caption tracks")
                continue
            cap = requests.get(base + track["url"], headers=_UA, timeout=20)
            if cap.status_code != 200:
                errors.append(f"{inst}: caption HTTP {cap.status_code}")
                continue
            text = parse_captions(cap.text)
            if len(text) >= config.MIN_TEXT_CHARS:
                return text
            errors.append(f"{inst}: caption text too short")
        except Exception as e:
            errors.append(f"{inst}: {e.__class__.__name__}")
    raise ConnectorError("; ".join(errors))


def _captions_via_piped(video_id: str) -> str:
    errors = []
    for inst in config.PIPED_INSTANCES:
        base = inst.rstrip("/")
        try:
            resp = requests.get(f"{base}/streams/{video_id}", headers=_UA, timeout=12)
            if resp.status_code != 200:
                errors.append(f"{inst}: HTTP {resp.status_code}")
                continue
            track = _pick_proxy_caption(resp.json().get("subtitles", []), "code", "auto")
            if not track or not track.get("url"):
                errors.append(f"{inst}: no subtitle tracks")
                continue
            cap = requests.get(track["url"], headers=_UA, timeout=20)
            if cap.status_code != 200:
                errors.append(f"{inst}: subtitle HTTP {cap.status_code}")
                continue
            text = parse_captions(cap.text)
            if len(text) >= config.MIN_TEXT_CHARS:
                return text
            errors.append(f"{inst}: subtitle text too short")
        except Exception as e:
            errors.append(f"{inst}: {e.__class__.__name__}")
    raise ConnectorError("; ".join(errors))


def _asr_via_ytdlp(video_id: str) -> str:
    from .asr import get_transcriber

    transcriber = get_transcriber()
    if transcriber is None:
        raise ConnectorError(
            "no ASR backend installed (`pip install faster-whisper` enables "
            "direct audio transcription)"
        )
    import tempfile
    from pathlib import Path

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmp:
        _extract(
            url,
            {"format": "bestaudio/best", "outtmpl": str(Path(tmp) / "audio.%(ext)s")},
            download=True,
        )
        audio_files = list(Path(tmp).glob("audio.*"))
        if not audio_files:
            raise ConnectorError("audio download failed")
        return transcriber(str(audio_files[0]))


def fetch_youtube_transcript(video_id: str, fallback_meta: dict | None = None) -> dict:
    """Return {text, title, channel, published, method} for one video.

    Non-API strategy chain — each step tolerates the previous one's failure:
      1. youtube-transcript-api      (lightest; avoids full-page extraction)
      2. yt-dlp captions             (web, then android/ios player clients —
                                      alternate endpoints often dodge bot-checks)
      3. Invidious public instances  (fetched from THEIR servers, not this IP)
      4. Piped public instances      (same idea, different network)
      5. yt-dlp audio + local ASR    ("read the video directly")
    Metadata comes from the search candidate (`fallback_meta`) so ingestion
    never depends on a page extraction succeeding.
    """
    meta = {
        "title": (fallback_meta or {}).get("title", "") or video_id,
        "channel": (fallback_meta or {}).get("channel", ""),
        "published": (fallback_meta or {}).get("published", ""),
    }
    errors: list[str] = []

    def _short(e) -> str:  # library errors can be multi-paragraph essays
        return re.sub(r"\s+", " ", str(e)).strip()[:180]

    try:
        text = _captions_via_transcript_api(video_id)
        if len(text) >= config.MIN_TEXT_CHARS:
            return {**meta, "text": text, "method": "captions (transcript API)"}
        errors.append("transcript API: text too short")
    except Exception as e:
        errors.append(f"transcript API: {_short(e)}")

    try:
        text, ydl_meta, label = _captions_via_ytdlp(video_id)
        meta = {k: (ydl_meta.get(k) or meta[k]) for k in meta}
        return {**meta, "text": text, "method": f"captions ({label})"}
    except Exception as e:
        errors.append(f"yt-dlp captions: {_short(e)}")

    for name, fn in (("invidious", _captions_via_invidious),
                     ("piped", _captions_via_piped)):
        try:
            return {**meta, "text": fn(video_id), "method": f"captions ({name})"}
        except Exception as e:
            errors.append(f"{name}: {_short(e)}")

    try:
        return {**meta, "text": _asr_via_ytdlp(video_id), "method": "ASR"}
    except Exception as e:
        errors.append(f"ASR: {_short(e)}")

    combined = " | ".join(errors)
    raise ConnectorError(f"all transcript strategies failed: {combined}"
                         + _block_hint(combined))


def check_youtube_access(probe_video_id: str | None = None) -> dict:
    """Can this machine fetch a transcript at all right now?

    A sweep can spend twenty minutes discovering that every fetch is refused.
    This answers the same question in seconds against one known-captioned
    video, so callers can abort early and say why.
    """
    vid = probe_video_id or config.ACCESS_PROBE_VIDEO_ID
    errors = []
    raw_errors = []  # untruncated: block markers often sit past the cut-off
    for name, fn in (
        ("transcript API", lambda: _captions_via_transcript_api(vid)),
        ("yt-dlp", lambda: _captions_via_ytdlp(vid)[0]),
        ("invidious", lambda: _captions_via_invidious(vid)),
        ("piped", lambda: _captions_via_piped(vid)),
    ):
        try:
            text = fn()
            if text and len(text) >= config.MIN_TEXT_CHARS:
                return {"ok": True, "method": name, "detail": f"{len(text)} chars"}
            errors.append(f"{name}: empty")
        except Exception as e:  # noqa: BLE001 — probing, every failure is data
            flat = re.sub(r"\s+", " ", str(e))
            raw_errors.append(flat)
            errors.append(f"{name}: {flat[:110]}")
    combined = " | ".join(errors)
    blocked = looks_blocked(" ".join(raw_errors))
    return {
        "ok": False,
        "method": None,
        "detail": combined,
        "blocked": blocked,
        "hint": (
            "YouTube is refusing transcript requests from this network. Wait for "
            "the rate limit to clear, set YTDLP_COOKIES_FILE to an exported "
            "cookies.txt, or run from a different network. Search and relevance "
            "filtering still work; only transcript fetching is affected."
            if blocked else
            "No transcript route succeeded for the probe video."
        ),
    }


def fetch_rss_transcript(candidate: dict) -> dict:
    """Transcribe a podcast episode's audio enclosure via local ASR."""
    from .asr import transcribe_url

    audio_url = candidate.get("audio_url")
    if not audio_url:
        raise ConnectorError("episode has no audio enclosure")
    text = transcribe_url(audio_url)
    return {
        "title": candidate.get("title", ""),
        "channel": candidate.get("channel", ""),
        "published": candidate.get("published", ""),
        "text": text,
        "method": "ASR",
    }
