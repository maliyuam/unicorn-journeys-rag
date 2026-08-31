"""Minimal in-process background jobs with progress reporting.

Good for a single-worker deployment (the default `uvicorn backend.app:app`).
For multi-worker or multi-host deployments, replace with a shared queue
(Celery/RQ/arq) — the JobManager interface is deliberately tiny to make that
swap easy.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from datetime import UTC, datetime

_MAX_KEPT = 100


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"  # queued | running | done | error
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created_at = datetime.now(UTC).isoformat()
        self.finished_at = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def submit(self, kind: str, fn) -> Job:
        """fn(report) -> result; report(progress: float, message: str)."""
        job = Job(kind)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > _MAX_KEPT:
                dead = self._order.pop(0)
                self._jobs.pop(dead, None)

        def report(progress: float, message: str = "") -> None:
            job.progress = max(0.0, min(1.0, progress))
            if message:
                job.message = message

        def runner() -> None:
            job.status = "running"
            try:
                job.result = fn(report)
                job.progress = 1.0
                job.status = "done"
            except Exception as e:  # noqa: BLE001 — job errors must be captured
                job.status = "error"
                job.error = str(e)
                job.message = "failed"
                traceback.print_exc()
            finally:
                job.finished_at = datetime.now(UTC).isoformat()

        threading.Thread(target=runner, daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)


jobs = JobManager()
