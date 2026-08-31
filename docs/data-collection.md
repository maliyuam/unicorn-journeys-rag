# Collecting a corpus

The bundled sample corpus exists so the pipeline runs on a fresh clone. To
study real people you have to collect real media, and that is what the
**Discover** and **Sweep** flows do.

Collected media belongs to whoever published it. It lands in the git-ignored
`data/transcripts/` and is never committed. Respect the terms of the platforms
you fetch from.

## The human-in-the-loop discovery flow

The **Discover** tab automates find → filter → fetch, with a human approval
step in the middle:

1. **Search (multi-strategy, non-API first)** — ① yt-dlp keyless search,
   ② YouTube Data API v3 (only if `YOUTUBE_API_KEY` is set — an optional
   booster, never required), ③ Invidious public instances. Podcast RSS via
   feedparser (episode metadata + audio enclosures).
2. **Relevance filter** — a schema-constrained Claude call returning a
   per-candidate verdict and reason; offline, a keyword heuristic on
   name/company/interview signals.
3. **Human review** — candidates appear with relevance badges and reasons;
   relevant-and-new items are pre-checked. *Nothing is ingested without
   explicit approval.*
4. **Transcript fetch (five strategies per video, all keyless)** —
   ① `youtube-transcript-api` (lightest), ② yt-dlp caption tracks via the web
   client then **android/ios player clients** (alternate endpoints that often
   dodge IP bot-checks), ③ Invidious instances, ④ Piped instances (both proxy
   the fetch through *their* servers, not your IP — `INVIDIOUS_INSTANCES` /
   `PIPED_INSTANCES` override the defaults), ⑤ **local ASR**
   (`pip install faster-whisper`), which downloads the audio and reads the
   video directly — also the only path for caption-disabled videos and RSS
   audio.
5. **Ingest** — through the same idempotent upsert path as file uploads
   (`source_id = youtube-<videoId>` / `rss-<hash>`), so re-discovering an
   already-ingested video shows as *already ingested* and is never duplicated.

Fetches run 4-at-a-time; each item fails independently with its reason, and the
job shows which strategy served each success.

## Rate limiting is the main operational constraint

YouTube bot-checks caption fetches ("Sign in to confirm you're not a bot"), and
**volume triggers it**: a wide parallel burst across dozens of videos gets a
whole wave rejected, and heavy sweeping can rate-limit the IP for hours.

The sweep therefore fetches 2 at a time with pauses, retries blocked items once
after a cool-off, and fails fast rather than grinding through a queue that
cannot succeed. If you hit it:

- set `YTDLP_COOKIES_FROM_BROWSER=chrome` (or `edge`/`firefox`, with the
  browser fully closed) in `.env` to authenticate with your own session;
- or wait for the block to clear and re-run — **the sweep is idempotent**, so
  it picks up exactly the sources that are still missing and re-ingests
  nothing;
- or run from a different network.

Search and relevance filtering are unaffected; only transcript fetching is.

**Check before you sweep.** `GET /api/connectivity` fetches one known-captioned
video through every strategy and reports whether transcripts are reachable
right now — seconds rather than a twenty-minute sweep. The sweep runs the same
probe as a preflight and reports the verdict under `access`, so a run that
collected nothing tells you *why*.

Note that the public Invidious and Piped instances are mostly defunct. When
tested, one of twelve Invidious hosts responded and it returned an empty
caption body, and no Piped host worked. Treat those routes as a bonus, not a
reliable bypass.

## Authenticating with cookies

Signing in restores access on a rate-limited network. Export your YouTube
cookies in Netscape format to `secrets/cookies.txt` (the directory is
git-ignored) and the app picks them up with no `.env` change. Full steps are in
[secrets/README.md](../secrets/README.md). Verify with:

```bash
python scripts/check_cookies.py
```

It checks the format, confirms the export captured a signed-in session, warns
on expiry, then does one real fetch — and never prints a cookie value, so the
output is safe to share.

Automatic extraction (`YTDLP_COOKIES_FROM_BROWSER`) often fails on Windows:
Edge holds a lock on its cookie database while running, and Chrome's app-bound
encryption defeats yt-dlp's DPAPI decryption. A manual export avoids both.

**These cookies are a live signed-in Google session — treat the file like a
password.** See [SECURITY.md](../SECURITY.md).

## The channel that cannot be rate-limited: podcasts

Podcast RSS is served by publishers, and episode audio by their own CDNs, so
this path keeps working when YouTube refuses everything.

Episodes have no captions, so they are transcribed locally with
`faster-whisper` (`pip install faster-whisper`, roughly 2 minutes per 30-minute
episode on CPU).

- `data/podcast_feeds.json` holds eight verified African tech feeds.
- `GET /api/podcasts/search?term=...` finds more feeds by name through Apple's
  public directory, so you never need to hunt for a feed URL.
- The **sweep automatically scans podcasts when YouTube access is blocked**,
  and `always_podcasts: true` scans them regardless.

Episodes go through the same relevance filter as YouTube, which matters more
than it sounds: a naive surname match files an episode about NGO funding under
"Felix **Ike**" because his surname sits inside "l*ike*" and "str*ike*". That
exact mistake put wrong facts in a corpus during development; the filter now
matches whole tokens, and a test pins it.

## Attribution: the failure that actually costs you

Filing one person's interview under another founder's name is the worst thing
this pipeline can do — it puts a stranger's career into a founder's narrative,
and the citations look perfectly legitimate. Two real cases, both caught and
removed:

- **Name collision.** "Dr Omobola Johnson (TLcom)" was ingested under **Jeremy
  Johnson** of Andela — 29 chunks, zero mentions of "Jeremy". The filter had
  accepted "surname + interview-style content", but interview words appear in
  nearly every description, so that rule was really just "surname". It now
  requires a **company** signal alongside a surname, and rejects outright when
  a different given name precedes the surname ("Dr *Omobola* Johnson"), while
  still accepting shortened forms and initials ("Iyin 'E' Aboyeji").
- **Billed but absent.** A panel opening "my name is Jenny Fielding" was filed
  under a founder with no mention of him or his company in 37k characters.
  Nothing visible before fetching could have caught this: the title and
  description were plausible.

Two audits check the *transcripts*, which the pre-fetch filter never sees.

### LLM audit (recommended)

`scripts/audit_attribution_llm.py`, also the **Attribution audit** card on the
Ingest tab and `POST /api/audit/attribution`. Claude reads excerpts from each
stored transcript and returns one of three verdicts — the founder **speaks**,
is **discussed**, or is **absent** — with a confidence, who the transcript is
actually about, and a supporting quote.

```bash
python scripts/audit_attribution_llm.py                 # report
python scripts/audit_attribution_llm.py --remove        # delete confident "absent"
python scripts/audit_attribution_llm.py --json out.json
```

This is the audit that can be trusted to delete, because it reads through the
noise that defeats string matching — it recognises "Tosi and La" as Tosin
Eniolorunda and "Helen" as Halan, and knows a company can have renamed. Only
**high or medium** confidence `absent` verdicts are removable; a low-confidence
guess is reported for a human and never deletes anything.

Excerpts are sampled rather than sending whole transcripts: the opening chunks
(where speakers introduce themselves, and where "my name is Jenny Fielding"
appears), the passages densest in name mentions, and one from the middle.

### String audit (no LLM needed)

`scripts/audit_attribution.py` flags sources that never name the founder or
their company. **It is a review queue, not a delete list**, precisely because
ASR garbling and company renames make it over-flag — running it against a real
corpus flagged nine sources of which most were genuine. Use it only when no
credentials are available.

## Corpus sweep

`data/founders.json` holds the roster (name, company, country, and a tuned
search query each). **Sweep all founders** in the Ingest tab runs the whole
pipeline for every one of them: search → relevance filter → fetch → ingest,
capped at N videos per founder, indexing after each founder so partial progress
survives an interruption. A per-founder results table reports candidates found,
approved, ingested, and failed.

The sweep auto-approves only filter-relevant candidates — the human-in-the-loop
step becomes *post-hoc review*: every result appears in the sources table where
anything wrong can be deleted. Use the single-founder Discover flow when you
want to approve each video before it is fetched.

```bash
curl -X POST localhost:8017/api/sweep -H "Content-Type: application/json" \
  -d '{"max_candidates": 6, "search_results": 10}'
```

Manual uploads (Ingest tab) remain the path for **reports** — company filings,
local news articles — which have no connector.
