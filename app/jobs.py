"""MoveMatch background jobs -- long CPU work with a progress bar.

Analysing a song video (or scoring a full-length game) takes minutes of CPU YOLO,
far longer than a request should block. A job runs in a daemon thread of the
server process that received it; its state lives in ``jobs/<id>.json`` so ANY
worker (gunicorn runs several processes, waitress several threads) can answer the
browser's progress polls. Writes go through a temp file + rename, so a reader never
sees half a file.

Limits, accepted for a local single-machine app: a job dies with its worker
process (its file then stays "running"; ``STALE_S`` without an update marks it
failed), and nothing caps how many jobs run at once.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

JOB_DIR = Path(__file__).resolve().parent / "jobs"
STALE_S = 600       # a running job silent this long is reported as failed
REPORT_EVERY = 0.5  # s between progress writes (polls don't need more)


def _path(job_id: str) -> Path:
    if not job_id.isalnum():
        raise ValueError(f"bad job id {job_id!r}")
    return JOB_DIR / f"{job_id}.json"


def _write(state: dict) -> None:
    JOB_DIR.mkdir(exist_ok=True)
    path = _path(state["id"])
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def get(job_id: str) -> dict | None:
    """The job's state, or None if unknown."""
    try:
        state = json.loads(_path(job_id).read_text())
    except (OSError, ValueError):
        return None
    if state["status"] == "running" and time.time() - state["updated"] > STALE_S:
        state.update(status="error", error="The job stopped responding (server restart?).")
    return state


def start(kind: str, fn: Callable[..., Any], *args, job_id: str | None = None,
          meta: dict | None = None) -> str:
    """Run ``fn(report, *args)`` in the background; returns the job id. ``fn`` calls
    ``report(progress 0..1, message)`` as it goes and returns a JSON-able result."""
    job_id = job_id or uuid.uuid4().hex
    state = {"id": job_id, "kind": kind, "status": "running", "progress": 0.0,
             "message": "Starting...", "result": None, "error": None,
             "meta": meta or {}, "updated": time.time()}
    _write(state)
    last = [0.0]

    def report(progress: float, message: str | None = None) -> None:
        now = time.time()
        if now - last[0] < REPORT_EVERY and progress < 1:
            return
        last[0] = now
        state.update(progress=round(float(progress), 4), updated=now)
        if message:
            state["message"] = message
        _write(state)

    def run() -> None:
        try:
            result = fn(report, *args)
            state.update(status="done", progress=1.0, message="Done", result=result)
        except Exception as exc:
            log.exception("Job %s (%s) failed", job_id, kind)
            state.update(status="error", error=str(exc) or exc.__class__.__name__,
                         trace=traceback.format_exc(limit=5))
        state["updated"] = time.time()
        _write(state)

    threading.Thread(target=run, name=f"job-{kind}-{job_id[:8]}", daemon=True).start()
    return job_id
