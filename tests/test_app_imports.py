"""The API layer must import, and its routes must exist.

This file exists because of a bug that reached a published `main`: FastAPI
needs `python-multipart` to parse the `File()`/`Form()` parameters on
`/api/ingest/upload`, and it was missing from `requirements.txt`. FastAPI
imports it internally, so no `import` statement in this codebase named it —
static analysis could not see it either.

The failure mode is as bad as it gets: FastAPI raises while *defining the
route*, so `backend.app` cannot be imported and the whole server dies on
startup. Not the upload endpoint — everything. A clean
`pip install -r requirements.txt` followed by the documented
`uvicorn backend.app:app` crashed for every new user.

Nothing caught it. The suite never imported `backend.app` (every other test
reaches the engine, ingest and connectors directly), so 76 passing tests said
nothing about whether the application starts. Only the Docker smoke test in CI
failed, and only because it actually runs the container.

So: import the app, and assert the routes the README documents are really
registered. It is a cheap test that pins an entire class of "works on the
maintainer's machine" packaging failures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_app_module_imports():
    """If a runtime dependency is undeclared, this is where it surfaces."""
    from backend.app import app

    assert app.title == "Unicorn Journeys RAG"


def test_multipart_upload_route_is_registered():
    """The route whose Form()/File() params need python-multipart."""
    from backend.app import app

    routes = {getattr(r, "path", None) for r in app.routes}
    assert "/api/ingest/upload" in routes


def test_documented_routes_exist():
    """Guards the API table in README.md against silent drift.

    A previous README documented `POST /api/ingest/demo`, which had been
    deleted — so the docs promised an endpoint that returned 404.
    """
    from backend.app import app

    documented = {
        ("GET", "/api/status"),
        ("GET", "/api/founders"),
        ("POST", "/api/ingest/samples"),
        ("POST", "/api/ingest"),
        ("POST", "/api/ingest/upload"),
        ("POST", "/api/ingest/batch"),
        ("POST", "/api/discover"),
        ("POST", "/api/discover/ingest"),
        ("POST", "/api/sweep"),
        ("GET", "/api/roster"),
        ("GET", "/api/connectivity"),
        ("GET", "/api/podcasts/search"),
        ("GET", "/api/podcasts/curated"),
        ("POST", "/api/audit/attribution"),
        ("GET", "/api/jobs/{job_id}"),
        ("GET", "/api/sources"),
        ("DELETE", "/api/sources/{source_id}"),
        ("POST", "/api/ask"),
        ("POST", "/api/narrative"),
        ("POST", "/api/eval"),
        ("GET", "/api/eval/latest"),
        ("POST", "/api/clear"),
    }
    actual = {
        (method, r.path)
        for r in app.routes
        for method in getattr(r, "methods", set()) or set()
    }
    missing = documented - actual
    assert not missing, f"documented endpoints that do not exist: {sorted(missing)}"
