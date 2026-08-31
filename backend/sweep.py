"""Corpus sweep: run discovery → relevance filter → fetch → ingest across a
roster of founders (the paper's sixteen by default).

The relevance filter is what makes this safe to run unattended: only
candidates the filter judges relevant are fetched, capped per founder. Every
result lands in the source registry, so anything wrong can be reviewed and
deleted afterwards in the UI.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .connectors import (
    check_youtube_access,
    fetch_rss,
    fetch_rss_transcript,
    fetch_youtube_transcript,
    looks_blocked,
    score_candidates,
    search_youtube,
)

log = logging.getLogger("unicorn_rag.sweep")

# Fetch politeness. Volume is what trips YouTube's bot-check: a 4-wide burst
# across dozens of videos got every request in a wave rejected, while a
# narrower, paced sweep keeps succeeding. Slower and complete beats fast and
# empty, since the point of the sweep is maximum coverage per founder.
FETCH_WAVE = 2
PAUSE_BETWEEN_WAVES_SEC = 1.5
PAUSE_BETWEEN_FOUNDERS_SEC = 3.0
# One retry for items rejected by a bot-check, after a longer cool-off. The
# block is rate-driven and intermittent, so a paused retry recovers many.
BLOCKED_RETRY_PAUSE_SEC = 20.0
# Abandon the retry pass after this many consecutive block errors — the
# cool-off clearly did not work and the rest of the queue will fail the same.
RETRY_GIVE_UP_AFTER = 3


def load_roster() -> list[dict]:
    path = config.DATA_DIR / "founders.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("founders", [])


def load_podcast_feeds() -> list[dict]:
    path = config.DATA_DIR / "podcast_feeds.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("feeds", [])


def discover_podcast_episodes(
    llm, founder: str, company: str, feeds: list[dict] | None = None,
    per_feed: int = 40,
) -> list[dict]:
    """Find podcast episodes featuring a founder across the curated feeds.

    Runs candidates through the same relevance filter as YouTube. That
    matters: naive substring matching on a surname is actively dangerous —
    "Ike" appears inside "like" and "strike", which is enough to file an
    episode about NGO funding under the wrong founder's name. The filter
    matches on whole tokens and weighs company and interview signals.
    """
    feeds = feeds if feeds is not None else load_podcast_feeds()
    candidates: list[dict] = []
    for feed in feeds:
        try:
            episodes = fetch_rss(feed["feed_url"], max_items=per_feed)
        except Exception as e:  # noqa: BLE001 — one dead feed must not stop the scan
            log.warning("feed failed %s: %s", feed.get("name", "?"), str(e)[:90])
            continue
        for ep in episodes:
            if ep.get("audio_url"):
                candidates.append(ep)
    if not candidates:
        return []
    scored = score_candidates(llm, founder, company, candidates)
    return [c for c in scored if c.get("relevant")]


def _fetch_one(item: dict, founder: str, company: str) -> tuple[dict, str]:
    if item["kind"] == "youtube":
        fetched = fetch_youtube_transcript(
            item["id"],
            fallback_meta={
                "title": item.get("title", ""),
                "channel": item.get("channel", ""),
                "published": item.get("published", ""),
            },
        )
        source_type = "youtube"
    else:
        fetched = fetch_rss_transcript(item)
        source_type = "podcast"
    doc = {
        "founder": founder,
        "company": company,
        "source_title": fetched["title"] or item.get("title", ""),
        "source_type": source_type,
        "text": fetched["text"],
        "source_id": f"{item['kind']}-{item['id']}",
    }
    return doc, fetched.get("method", "unknown")


def fetch_many(
    items: list[dict], founder: str, company: str, on_progress=None
) -> tuple[list[dict], list[dict], list[dict]]:
    """Fetch transcripts for approved candidates, FETCH_WAVE at a time.

    Returns (docs, failed, methods). Aborts early when the first wave is
    entirely bot-check failures — the rest of the queue would fail the same
    way, and reporting that beats grinding through it.
    """
    docs: list[dict] = []
    failed: list[dict] = []
    methods: list[dict] = []
    done = 0
    for start in range(0, len(items), FETCH_WAVE):
        wave = items[start:start + FETCH_WAVE]
        with ThreadPoolExecutor(max_workers=FETCH_WAVE) as pool:
            futures = {
                pool.submit(_fetch_one, item, founder, company): item for item in wave
            }
            for fut in as_completed(futures):
                item = futures[fut]
                done += 1
                if on_progress:
                    on_progress(done, len(items))
                try:
                    doc, method = fut.result()
                    docs.append(doc)
                    methods.append(
                        {"source_title": doc["source_title"], "method": method}
                    )
                except Exception as e:  # noqa: BLE001 — per-item isolation
                    failed.append(
                        {
                            "source_title": item.get("title") or item.get("id", "?"),
                            "error": str(e),
                        }
                    )
        remaining = items[start + FETCH_WAVE:]
        if (
            remaining
            and not docs
            and failed
            and all(looks_blocked(f["error"]) for f in failed)
        ):
            failed.append(
                {
                    "source_title": f"{len(remaining)} remaining item(s)",
                    "error": (
                        "skipped without trying — YouTube is bot-checking this "
                        "network. Set YTDLP_COOKIES_FROM_BROWSER (browser fully "
                        "closed) or YTDLP_COOKIES_FILE in .env, or retry later."
                    ),
                }
            )
            break
        if remaining:
            time.sleep(PAUSE_BETWEEN_WAVES_SEC)

    # Second pass: bot-checks are rate-driven and intermittent, so items
    # rejected during a burst often succeed after a pause. Retry those once.
    # Retry even when nothing succeeded: a burst can get every item in the
    # first wave rejected, and gating the retry on a prior success meant the
    # fully-blocked case — the one that most needs a cool-off — never retried.
    blocked = [f for f in failed if looks_blocked(f["error"])]
    if blocked:
        retryable = [
            item for item in items
            if any(f["source_title"] in (item.get("title"), item.get("id"))
                   for f in blocked)
        ]
        if retryable:
            if on_progress:
                on_progress(len(items), len(items))
            time.sleep(BLOCKED_RETRY_PAUSE_SEC)
            recovered_titles = set()
            consecutive_blocks = 0
            for item in retryable:
                try:
                    doc, method = _fetch_one(item, founder, company)
                except Exception as e:  # noqa: BLE001 — keep the original failure
                    if looks_blocked(str(e)):
                        consecutive_blocks += 1
                        # The cool-off did not help; the block is still on.
                        # Walking the rest of the queue just burns minutes to
                        # collect identical errors.
                        if consecutive_blocks >= RETRY_GIVE_UP_AFTER:
                            log.info(
                                "retry abandoned after %d consecutive blocks",
                                consecutive_blocks,
                            )
                            break
                    continue
                consecutive_blocks = 0
                docs.append(doc)
                methods.append(
                    {"source_title": doc["source_title"], "method": method + " (retry)"}
                )
                recovered_titles.add(item.get("title") or item.get("id"))
                time.sleep(PAUSE_BETWEEN_WAVES_SEC)
            if recovered_titles:
                failed = [f for f in failed
                          if f["source_title"] not in recovered_titles]
                log.info("recovered %d blocked item(s) on retry", len(recovered_titles))
    return docs, failed, methods


def sweep(
    engine,
    llm,
    names: list[str] | None,
    max_candidates: int,
    search_results: int,
    report,
    include_podcasts: bool = True,
    always_podcasts: bool = False,
    preflight: bool = True,
) -> dict:
    """Discover + ingest for each founder in turn, indexing per founder so
    partial progress survives a mid-sweep failure."""
    roster = load_roster()
    if names:
        wanted = {n.strip().lower() for n in names}
        roster = [f for f in roster if f["name"].lower() in wanted]
    if not roster:
        raise ValueError("no founders matched the roster")

    # Preflight: a blocked network turns a sweep into twenty minutes of
    # guaranteed failures. Check once, up front, and say so plainly.
    # Callers that supply their own fetchers (tests) disable it — otherwise
    # this is an unmockable network call in the middle of the unit under test.
    if preflight:
        report(0.0, "checking transcript access")
        access = check_youtube_access()
        if not access["ok"]:
            log.warning("sweep preflight failed: %s", access["detail"][:200])
    else:
        access = {"ok": True, "method": "preflight skipped", "detail": ""}

    total = len(roster)
    results: list[dict] = []
    for i, entry_def in enumerate(roster):
        name = entry_def["name"]
        company = entry_def.get("company", "")
        base = i / total
        step = 1.0 / total
        entry: dict = {
            "founder": name,
            "company": company,
            "candidates": 0,
            "approved": 0,
            "ingested": 0,
            "failed": [],
            "methods": [],
        }
        report(base, f"[{i + 1}/{total}] searching {name}")
        try:
            query = entry_def.get("query") or (
                f"{name} {company} interview OR podcast OR keynote"
            )
            candidates = search_youtube(query, search_results)
        except Exception as e:  # noqa: BLE001 — one founder must not stop the sweep
            entry["error"] = str(e)[:300]
            results.append(entry)
            log.warning("sweep: search failed for %s: %s", name, str(e)[:160])
            continue

        candidates = score_candidates(llm, name, company, candidates)

        # Podcast RSS is served by publishers, not YouTube, so it keeps
        # working when YouTube rate-limits this network. Always include it
        # when transcript access is blocked; it is the only channel that can
        # still add material.
        if include_podcasts and (not access["ok"] or always_podcasts):
            report(base + step * 0.15, f"[{i + 1}/{total}] scanning podcasts for {name}")
            try:
                candidates += discover_podcast_episodes(llm, name, company)
            except Exception as e:  # noqa: BLE001
                log.warning("podcast scan failed for %s: %s", name, str(e)[:90])

        known = {s["source_id"] for s in engine.list_sources()}
        approved = [
            c
            for c in candidates
            if c.get("relevant") and f"{c['kind']}-{c['id']}" not in known
        ][:max_candidates]
        entry["candidates"] = len(candidates)
        entry["approved"] = len(approved)
        entry["podcast_approved"] = sum(1 for c in approved if c["kind"] == "rss")

        if approved:
            report(
                base + step * 0.25,
                f"[{i + 1}/{total}] fetching {len(approved)} transcript(s) for {name}",
            )
            docs, failed, methods = fetch_many(
                approved,
                name,
                company,
                on_progress=lambda d, t, base=base, step=step, i=i, name=name: report(
                    base + step * (0.25 + 0.5 * d / t),
                    f"[{i + 1}/{total}] {name}: fetched {d}/{t}",
                ),
            )
            entry["failed"] = failed
            entry["methods"] = methods
            if docs:
                report(
                    base + step * 0.85,
                    f"[{i + 1}/{total}] indexing {len(docs)} transcript(s) for {name}",
                )
                res = engine.ingest_documents(docs)
                entry["ingested"] = res.get("ingested", 0)
                entry["failed"] = entry["failed"] + res.get("failed", [])
        results.append(entry)
        log.info(
            "sweep %s: %d candidates, %d approved, %d ingested, %d failed",
            name, entry["candidates"], entry["approved"],
            entry["ingested"], len(entry["failed"]),
        )
        if i + 1 < total:
            time.sleep(PAUSE_BETWEEN_FOUNDERS_SEC)

    report(1.0, "sweep complete")
    stats = engine.store.stats()
    # NB: do not splat stats here — it also carries a "founders" key (chunk
    # counts per founder) which would clobber the per-founder sweep results.
    return {
        "founders": results,
        "chunks": stats["chunks"],
        "sources": stats["sources"],
        "founders_indexed": len(stats["founders"]),
        "access": access,
    }
