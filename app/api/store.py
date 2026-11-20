"""In-memory sessions and one fit worker (PR-11, PR-12, PR-13).

Nothing is written to disk and sample data is never logged: logs carry ids, counts and
timings only. Sessions are forgotten after `session_idle_s` without a request. Fits run one at
a time on a single worker thread; at most `queue_size` wait, and one more is refused so the
API can answer 429 with a retry time.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field

from app.pipeline.swipe_format import Swipe

from .config import SETTINGS
from .fit_job import FitState, run_fit

log = logging.getLogger("magnet_eye")


class QueueFull(Exception):
    def __init__(self, retry_after_s: int):
        self.retry_after_s = retry_after_s


@dataclass
class Session:
    id: str
    app_version: str
    created: float
    touched: float
    swipes: list[Swipe] = field(default_factory=list)
    fit: FitState = field(default_factory=FitState)
    result: dict | None = None
    result_swipes: int = 0


@dataclass
class Job:
    id: str
    session_id: str
    swipes: int  # how many swipes this fit includes
    status: str = "queued"  # queued | running | done | failed
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None
    error: str | None = None


class Store:
    def __init__(self, settings=SETTINGS):
        self.settings = settings
        self.lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[str] = queue.Queue(maxsize=settings.queue_size)
        self.fit_seconds = 1.0  # running estimate, for Retry-After
        self._worker = threading.Thread(target=self._run, name="fit-worker", daemon=True)
        self._worker.start()

    # ---- sessions
    def sweep(self) -> None:
        now = time.monotonic()
        with self.lock:
            for sid in [
                s.id
                for s in self.sessions.values()
                if now - s.touched > self.settings.session_idle_s
            ]:
                del self.sessions[sid]
                log.info("session %s expired", sid)
            for jid in [
                j.id
                for j in self.jobs.values()
                if j.finished and now - j.finished > self.settings.job_keep_s
            ]:
                del self.jobs[jid]

    def create(self, app_version: str) -> Session:
        self.sweep()
        now = time.monotonic()
        s = Session(id=secrets.token_urlsafe(12), app_version=app_version, created=now, touched=now)
        with self.lock:
            self.sessions[s.id] = s
        log.info("session %s created (app %s)", s.id, app_version)
        return s

    def get(self, sid: str) -> Session | None:
        self.sweep()
        with self.lock:
            s = self.sessions.get(sid)
            if s:
                s.touched = time.monotonic()
            return s

    def delete(self, sid: str) -> bool:
        with self.lock:
            return self.sessions.pop(sid, None) is not None

    # ---- jobs
    def add_swipe(self, s: Session, swipe: Swipe) -> Job:
        """Store the swipe and queue a refit of the whole session."""
        with self.lock:
            s.swipes.append(swipe)
            job = Job(id=secrets.token_urlsafe(12), session_id=s.id, swipes=len(s.swipes))
            try:
                self.queue.put_nowait(job.id)
            except queue.Full:
                s.swipes.pop()
                wait = int((self.queue.qsize() + 1) * self.fit_seconds) + 1
                raise QueueFull(wait) from None
            self.jobs[job.id] = job
        log.info("session %s: swipe %d queued as job %s", s.id, len(s.swipes), job.id)
        return job

    def job(self, jid: str) -> Job | None:
        with self.lock:
            return self.jobs.get(jid)

    def _run(self) -> None:
        while True:
            jid = self.queue.get()
            with self.lock:
                job = self.jobs.get(jid)
                sess = self.sessions.get(job.session_id) if job else None
                swipes = list(sess.swipes[: job.swipes]) if sess else None
                if job:
                    job.status = "running"
            if not job or sess is None:
                if job:
                    job.status, job.error, job.finished = (
                        "failed",
                        "session ended",
                        time.monotonic(),
                    )
                continue
            t0 = time.perf_counter()
            try:
                result = run_fit(swipes, sess.fit)
            except Exception as e:  # report the reason, never the data
                with self.lock:
                    job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                    job.finished = time.monotonic()
                log.warning("job %s failed: %s", jid, type(e).__name__)
                continue
            dt = time.perf_counter() - t0
            self.fit_seconds = 0.7 * self.fit_seconds + 0.3 * dt
            with self.lock:
                if job.swipes >= sess.result_swipes:
                    sess.result, sess.result_swipes = result, job.swipes
                job.status, job.finished = "done", time.monotonic()
            log.info("job %s done: %d swipes in %.2f s", jid, job.swipes, dt)
