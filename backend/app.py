"""FastAPI app: JSON API + static single-page UI."""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config
from .asr import asr_available
from .attribution import audit_corpus
from .connectors import (
    ConnectorError,
    check_youtube_access,
    fetch_rss,
    score_candidates,
    search_podcasts,
    search_youtube,
)
from .engine import get_engine
from .eval_harness import load_previous_run, run_evaluation
from .ingest import IngestError, load_sample_docs, parse_upload
from .jobs import jobs
from .llm import get_llm
from .sweep import fetch_many, load_roster, sweep

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("unicorn_rag.api")

app = FastAPI(title="Unicorn Journeys RAG", version="2.1")


class IngestDoc(BaseModel):
    founder: str
    company: str = ""
    source_title: str
    source_type: str = "other"
    text: str = Field(min_length=1)
    source_id: str | None = None


class IngestBatch(BaseModel):
    docs: list[IngestDoc] = Field(min_length=1, max_length=config.MAX_BATCH_DOCS)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    founder: str | None = None
    k: int | None = Field(default=None, ge=1, le=25)


class NarrativeRequest(BaseModel):
    founder: str
    k: int | None = Field(default=None, ge=1, le=25)


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/api/status")
def status():
    return get_engine().status()


@app.get("/api/founders")
def founders():
    stats = get_engine().store.stats()
    return {
        "founders": [
            {"name": name, "chunks": count}
            for name, count in sorted(stats["founders"].items())
        ]
    }


# --- ingest -----------------------------------------------------------------

@app.post("/api/ingest/samples")
def ingest_samples():
    """Ingest the fictional sample corpus bundled in `data/samples/`.

    The first-run path: it gives a fresh clone something to retrieve over
    without network access, credentials, or a discovery sweep. Idempotent —
    re-running skips unchanged sources.
    """
    try:
        return get_engine().ingest_documents(load_sample_docs())
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/ingest")
def ingest(doc: IngestDoc):
    """Synchronous single-document ingest (form paste)."""
    try:
        return get_engine().ingest_documents([doc.model_dump()])
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/ingest/batch")
def ingest_batch(batch: IngestBatch):
    """Asynchronous batch ingest — returns a job to poll."""
    docs = [d.model_dump() for d in batch.docs]
    job = jobs.submit(
        "ingest-batch",
        lambda report: get_engine().ingest_documents(docs, progress=report),
    )
    return {"job_id": job.id}


@app.post("/api/ingest/upload")
async def ingest_upload(
    files: list[UploadFile] = File(...),
    founder: str = Form(""),
    company: str = Form(""),
    source_type: str = Form("other"),
):
    """Multi-file upload (.txt/.md/.json/.srt/.vtt). Form fields provide
    default metadata for files that don't carry their own (JSON can embed it).
    Parsing happens inline so format errors surface immediately; chunking and
    embedding run as a background job."""
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"too many files ({len(files)} > {config.MAX_UPLOAD_FILES})",
        )
    docs, failed = [], []
    for f in files:
        raw = await f.read()
        name = f.filename or "unnamed"
        try:
            partial = parse_upload(name, raw)
        except IngestError as e:
            failed.append({"source_title": name, "error": str(e)})
            continue
        stem = name.rsplit(".", 1)[0]
        docs.append(
            {
                "founder": partial.get("founder") or founder,
                "company": partial.get("company") or company,
                "source_title": partial.get("source_title") or stem,
                "source_type": partial.get("source_type") or source_type,
                "text": partial.get("text", ""),
            }
        )
    if not docs:
        raise HTTPException(
            status_code=422,
            detail="no ingestable files: "
            + "; ".join(f"{f['source_title']}: {f['error']}" for f in failed),
        )

    def run(report):
        result = get_engine().ingest_documents(docs, progress=report)
        result["failed"] = failed + result.get("failed", [])
        return result

    job = jobs.submit("ingest-upload", run)
    log.info("upload accepted: %d file(s), %d unparsable", len(docs), len(failed))
    return {"job_id": job.id, "parsed": len(docs), "unparsable": failed}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job.to_dict()


# --- discovery (YouTube / RSS, human-in-the-loop) ---------------------------

class DiscoverRequest(BaseModel):
    mode: str = Field(pattern="^(youtube|rss)$")
    founder: str = Field(min_length=2)
    company: str = ""
    query: str = ""
    feed_url: str = ""
    max_results: int = Field(default=0, ge=0, le=50)


class DiscoverCandidate(BaseModel):
    kind: str
    id: str
    url: str = ""
    audio_url: str | None = None
    title: str = ""
    channel: str = ""
    published: str = ""


class DiscoverIngestRequest(BaseModel):
    founder: str = Field(min_length=2)
    company: str = ""
    items: list[DiscoverCandidate] = Field(min_length=1, max_length=50)


@app.post("/api/discover")
def discover(req: DiscoverRequest):
    """Search + relevance-filter candidates. Returns a job; NOTHING is
    ingested here — the user reviews and approves candidates in the UI."""
    engine = get_engine()

    def run(report):
        report(0.1, "searching")
        if req.mode == "youtube":
            query = req.query.strip() or (
                f"{req.founder} {req.company} interview OR podcast OR keynote".strip()
            )
            candidates = search_youtube(query, req.max_results or None)
        else:
            if not req.feed_url.strip():
                raise ConnectorError("feed_url is required for RSS discovery")
            candidates = fetch_rss(req.feed_url.strip(), req.max_results or None)
        report(0.5, f"filtering {len(candidates)} candidate(s) for relevance")
        candidates = score_candidates(get_llm(), req.founder, req.company, candidates)
        ingested_ids = {s["source_id"] for s in engine.list_sources()}
        for c in candidates:
            c["already_ingested"] = f"{c['kind']}-{c['id']}" in ingested_ids
        report(1.0, "done")
        return {
            "candidates": candidates,
            "asr_available": asr_available(),
        }

    job = jobs.submit("discover", run)
    return {"job_id": job.id}


@app.post("/api/discover/ingest")
def discover_ingest(req: DiscoverIngestRequest):
    """Fetch transcripts for user-approved candidates and ingest them."""
    engine = get_engine()
    items = [c.model_dump() for c in req.items]

    def run(report):
        docs, failed, methods = fetch_many(
            items,
            req.founder,
            req.company,
            on_progress=lambda d, t: report(0.05 + 0.75 * d / t, f"fetched {d}/{t}"),
        )
        if docs:
            report(0.85, f"indexing {len(docs)} transcript(s)")
            result = engine.ingest_documents(docs)
            result["failed"] = failed + result.get("failed", [])
        else:
            result = {
                "ingested": 0,
                "skipped_unchanged": 0,
                "failed": failed,
                **engine.store.stats(),
            }
        result["fetched_methods"] = methods
        report(1.0, "done")
        return result

    job = jobs.submit("discover-ingest", run)
    log.info("discover-ingest accepted: %d approved item(s)", len(items))
    return {"job_id": job.id}


# --- corpus sweep (all founders) --------------------------------------------

class SweepRequest(BaseModel):
    founders: list[str] | None = None  # names; omit for the full roster
    # Defaults collect the maximum media the search layer will return per
    # founder. Coverage is the point: the paper's own finding is that
    # narrative completeness depends on thematic breadth, and thin corpora
    # are what produced its lowest CCS scores.
    max_candidates: int = Field(default=50, ge=1, le=50)
    search_results: int = Field(default=50, ge=1, le=50)
    # Podcast RSS keeps working when YouTube blocks this network. Included
    # automatically when access is blocked; set always_podcasts to scan feeds
    # even when YouTube is reachable.
    include_podcasts: bool = True
    always_podcasts: bool = False


@app.get("/api/roster")
def roster():
    return {"founders": load_roster()}


@app.get("/api/podcasts/search")
def podcast_search(term: str, limit: int = 10):
    """Find podcast RSS feeds by name. Podcast audio is served by the
    publisher, so this channel is unaffected by YouTube rate limiting."""
    try:
        return {"podcasts": search_podcasts(term, limit)}
    except ConnectorError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/api/podcasts/curated")
def podcast_curated():
    path = config.DATA_DIR / "podcast_feeds.json"
    if not path.exists():
        return {"feeds": []}
    return {"feeds": json.loads(path.read_text(encoding="utf-8")).get("feeds", [])}


@app.get("/api/connectivity")
def connectivity():
    """Can transcripts be fetched from this network right now? Seconds, not
    a twenty-minute sweep."""
    return check_youtube_access()


@app.post("/api/sweep")
def start_sweep(req: SweepRequest):
    """Run discovery → filter → fetch → ingest across the founder roster.

    Auto-approves only filter-relevant candidates (capped per founder);
    everything lands in the source registry for post-hoc review/deletion.
    """
    engine = get_engine()
    llm = get_llm()

    def run(report):
        return sweep(
            engine,
            llm,
            req.founders,
            req.max_candidates,
            req.search_results,
            report,
            include_podcasts=req.include_podcasts,
            always_podcasts=req.always_podcasts,
        )

    job = jobs.submit("sweep", run)
    log.info(
        "sweep started: %s founder(s), max %d candidate(s) each",
        len(req.founders) if req.founders else "all roster",
        req.max_candidates,
    )
    return {"job_id": job.id}


# --- evaluation --------------------------------------------------------------

class EvalRequest(BaseModel):
    include_generation: bool = True
    save: bool = True


@app.post("/api/eval")
def start_eval(req: EvalRequest):
    """Run the golden-set evaluation as a background job."""
    engine = get_engine()

    def run(report):
        report(0.05, "loading goldens")
        result = run_evaluation(
            engine, include_generation=req.include_generation, save=req.save
        )
        report(1.0, "evaluation complete")
        return result

    job = jobs.submit("eval", run)
    return {"job_id": job.id}


@app.get("/api/eval/latest")
def latest_eval():
    previous = load_previous_run()
    if previous is None:
        raise HTTPException(status_code=404, detail="no evaluation has been run yet")
    return previous


# --- attribution audit --------------------------------------------------------

class AuditRequest(BaseModel):
    founders: list[str] | None = None
    remove: bool = False


@app.post("/api/audit/attribution")
def audit_attribution(req: AuditRequest):
    """Have the LLM read each transcript and judge whether the founder it was
    filed under actually appears in it."""
    engine = get_engine()
    llm = get_llm()
    if llm.mode != "claude":
        raise HTTPException(
            status_code=503,
            detail=(
                "attribution auditing requires an LLM; string matching cannot "
                "see through speech-recognition errors. Set ANTHROPIC_API_KEY."
            ),
        )

    def run(report):
        result = audit_corpus(engine, llm, founders=req.founders, report=report)
        if req.remove and result["removable"]:
            for r in result["removable"]:
                engine.delete_source(r["source_id"])
            result["removed"] = [r["source_id"] for r in result["removable"]]
            result["corpus"] = engine.store.stats()
        return result

    job = jobs.submit("attribution-audit", run)
    return {"job_id": job.id}


# --- sources ----------------------------------------------------------------

@app.get("/api/sources")
def list_sources():
    return {"sources": get_engine().list_sources()}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str):
    engine = get_engine()
    known = {s["source_id"] for s in engine.list_sources()}
    if source_id not in known:
        raise HTTPException(status_code=404, detail="unknown source")
    return engine.delete_source(source_id)


@app.post("/api/clear")
def clear():
    return get_engine().clear()


# --- query ------------------------------------------------------------------

@app.post("/api/ask")
def ask(req: AskRequest):
    return get_engine().ask(req.question, founder=req.founder, k=req.k)


@app.post("/api/narrative")
def narrative(req: NarrativeRequest):
    result = get_engine().narrative(req.founder, k=req.k)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
