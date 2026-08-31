"""Push the local index into MongoDB once the cluster is reachable again.

The app falls back to the local file store whenever MongoDB is unreachable
(e.g. an Atlas IP-access-list rejection), so ingestion never stops. This
script merges the two by source_id — local wins on conflicts, MongoDB-only
sources are preserved — and writes the union back to MongoDB.

Usage:
    python scripts/sync_to_mongo.py            # merge local -> MongoDB
    python scripts/sync_to_mongo.py --dry-run  # report what would change
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.store import LocalStore, MongoStore  # noqa: E402


def main(dry_run: bool = False) -> int:
    if not config.MONGODB_URI:
        print("MONGODB_URI is not set in .env — nothing to sync to.")
        return 1

    local = LocalStore()
    local_chunks = local.all_chunks()
    local_sources = {s["source_id"]: s for s in local.list_sources()}
    if not local_chunks:
        print("Local index is empty — nothing to sync.")
        return 0

    try:
        mongo = MongoStore()
    except Exception as e:
        print(f"MongoDB unreachable: {type(e).__name__}: {str(e)[:200]}")
        print("\nIf this is Atlas, check Network Access — a TLS handshake error "
              "usually means this machine's public IP is not on the access list.")
        return 1

    mongo_chunks = mongo.all_chunks()
    mongo_sources = {s["source_id"]: s for s in mongo.list_sources()}

    local_ids = set(local_sources)
    mongo_only = [sid for sid in mongo_sources if sid not in local_ids]

    print(f"local:  {len(local_chunks):>5} chunks / {len(local_sources)} sources")
    print(f"mongo:  {len(mongo_chunks):>5} chunks / {len(mongo_sources)} sources")
    print(f"merge:  {len(local_sources)} from local + {len(mongo_only)} mongo-only")

    if dry_run:
        for sid in mongo_only:
            print(f"  would keep from mongo: {sid}")
        return 0

    # local chunks + their embeddings (aligned by construction)
    merged_chunks = list(local_chunks)
    local_vecs = local.embedding_map([c["id"] for c in local_chunks])
    missing_local = [c["id"] for c in local_chunks if c["id"] not in local_vecs]
    if missing_local:
        print(f"ERROR: {len(missing_local)} local chunk(s) have no embedding — "
              "re-run an ingest to rebuild the index before syncing.")
        return 1
    rows = [local_vecs[c["id"]] for c in merged_chunks]

    # preserve sources that exist only in MongoDB
    if mongo_only:
        keep = [c for c in mongo_chunks if c["source_id"] in set(mongo_only)]
        keep_vecs = mongo.embedding_map([c["id"] for c in keep])
        for chunk in keep:
            vec = keep_vecs.get(chunk["id"])
            if vec is None:
                continue
            merged_chunks.append(chunk)
            rows.append(vec)

    matrix = np.vstack(rows).astype(np.float32)
    merged_sources = list(local_sources.values()) + [
        mongo_sources[sid] for sid in mongo_only
    ]
    mongo.replace_all(merged_chunks, matrix, merged_sources)
    print(f"\nSynced: {len(merged_chunks)} chunks / {len(merged_sources)} sources "
          f"now in {config.MONGODB_DB}.{config.MONGODB_COLLECTION}")
    print("Restart the app to switch it back onto MongoDB.")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
