# Deployment

Three ways to run it, in increasing order of setup.

> **Read this first.** The app ships **no authentication**. Every endpoint —
> including `POST /api/clear`, which drops the index — is open to anyone who
> can reach the port. That is fine on `localhost` and wrong on the public
> internet. Put it behind an authenticating reverse proxy or your platform's
> access control before exposing it. See [SECURITY.md](../SECURITY.md).

## 1. Local Python

```bash
pip install -r requirements.txt
python -m uvicorn backend.app:app --port 8017
```

Open <http://localhost:8017> and click **Load sample corpus**.

Needs no credentials and no database: the app uses a local file-based vector
index under `data/index/` and runs generation in offline extractive mode.

## 2. Docker

```bash
docker build -t unicorn-rag .
docker run --rm -p 8017:8017 unicorn-rag
```

With credentials, pass them at run time — never bake them into an image layer:

```bash
docker run --rm -p 8017:8017 --env-file .env unicorn-rag
```

The image runs as a non-root user (uid 10001). First boot downloads the
embedding model (~130 MB), so mount a cache volume to make restarts instant:

```bash
docker run --rm -p 8017:8017 \
  -v unicorn-models:/home/app/.cache \
  -v unicorn-index:/app/data/index \
  unicorn-rag
```

## 3. Docker Compose

```bash
docker compose up                    # app only, local file index
docker compose --profile mongo up    # app + a local MongoDB
```

Compose wires up the volumes for you: `index`, `transcripts`, `models` and
(with the profile) `mongo`. `docker compose down` keeps them; `down -v`
discards them.

`.env` is optional — Compose loads it if present and starts without it if not.

Note that the bundled `mongo` service is a plain `mongod`: it has **no**
`$vectorSearch`, so the store falls back to client-side cosine. That is correct
but slower, and it is not what you want in production. Atlas provides the
native index.

## Single worker, on purpose

Both the Dockerfile and the documented commands run **one** uvicorn worker. The
in-process job manager (`backend/jobs.py`) holds job state in memory, so a
second worker would not see jobs the first one started — uploads and sweeps
would appear to vanish. To scale out, swap in a shared queue (Celery/RQ) behind
the same interface first.

## MongoDB Atlas

The recommended store for anything beyond a laptop.

1. Create a cluster and a database user.
2. Put the connection string in `.env`:
   ```
   MONGODB_URI=mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net
   MONGODB_DB=unicorn_rag
   ```
3. Start the app and ingest something. On first write the store
   **auto-provisions the `vector_index`** (`$vectorSearch`, 384-d cosine with a
   `founder` filter) — no console clicking needed.

Retrieval then runs on native `$vectorSearch`, which the status bar reports.

Operational notes:

- **`.env` holds your database password.** It is git-ignored. If it ever leaks,
  rotate the password in Atlas → Database Access.
- **Network Access allowlists your IP.** If your IP changes (VPN, new network),
  add the new one in Atlas → Network Access, or connections time out. A **TLS
  handshake error almost always means this**.
- **If Atlas is unreachable the app silently falls back to the local file
  store**, so ingestion never stops — but new data lands locally instead of in
  Atlas.

### Recovering after an outage

Fix access first, then merge the local index back into MongoDB and restart:

```bash
python scripts/sync_to_mongo.py --dry-run   # inspect the plan
python scripts/sync_to_mongo.py             # apply it
```

The sync merges by `source_id` (local wins, MongoDB-only sources preserved), so
nothing collected during the outage is lost.

### A local mongod without admin rights

`scripts/start_mongodb.ps1` fetches binaries into `%LOCALAPPDATA%\UnicornRAG`
and starts a user-level `mongod`. Point `MONGODB_URI` at
`mongodb://localhost:27017` to use it. Same caveat as the Compose service: no
`$vectorSearch`, so retrieval uses client-side cosine.

## Choosing an embedding backend

The default (`fastembed`, `BAAI/bge-small-en-v1.5`) needs no API key and
downloads ~130 MB once. In a constrained or fully air-gapped environment, force
the always-available fallback:

```
EMBEDDING_BACKEND=tfidf
```

Retrieval quality drops, but nothing downloads and nothing breaks. If the
backend changes, the engine detects the dimension mismatch at startup and
re-embeds the corpus automatically.

## Health check

`GET /api/status` returns the store, embedder and LLM mode actually in use.
Most confusing behaviour turns out to be a silent fallback — Atlas unreachable,
credentials unresolved — and this endpoint is where that shows up. It is what
the Compose health check polls.
