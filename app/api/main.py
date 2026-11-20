"""Magnet Eye backend API (SRS 4.3). All endpoints under /v1.

Run for development (reloads on save, reachable from a phone on the same Wi-Fi):
    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app.pipeline.swipe_format import SwipeFormatError, parse_swipe

from .config import API_VERSION, CALIBRATION_SCHEMA_VERSION, SETTINGS
from .store import QueueFull, Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

try:
    SERVER_VERSION = version("magnet-eye-backend")
except PackageNotFoundError:  # running from a checkout without install
    SERVER_VERSION = "dev"

app = FastAPI(title="Magnet Eye backend", version=SERVER_VERSION)
store = Store()
v1 = APIRouter(prefix="/v1")


def _error(status: int, code: str, message: str, headers: dict | None = None):
    raise HTTPException(status, detail={"code": code, "message": message}, headers=headers)


class NewSession(BaseModel):
    app_version: str
    api_version: str
    calibration_schema_version: int


@v1.get("/health")
def health():
    return {
        "status": "ok",
        "server_version": SERVER_VERSION,
        "api_version": API_VERSION,
        "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
        "queue": {"waiting": store.queue.qsize(), "size": SETTINGS.queue_size},
        "limits": {
            "max_swipes": SETTINGS.max_swipes,
            "max_samples_per_swipe": SETTINGS.max_samples,
            "session_idle_s": SETTINGS.session_idle_s,
        },
    }


def _check_versions(api_version: str, schema: object) -> None:
    """PR-21: refuse what this server cannot read, saying what it expected."""
    if api_version != API_VERSION:
        _error(
            409,
            "api_version",
            f"This server speaks API {API_VERSION}; the app sent {api_version}. Update the app "
            "or point it at a matching server.",
        )
    if schema != CALIBRATION_SCHEMA_VERSION:
        _error(
            409,
            "calibration_schema_version",
            f"This server reads calibration schema {CALIBRATION_SCHEMA_VERSION}; the app sent "
            f"{schema}. Update the app or the server.",
        )


@v1.post("/sessions", status_code=201)
def create_session(body: NewSession):
    _check_versions(body.api_version, body.calibration_schema_version)
    s = store.create(body.app_version)
    return {"session_id": s.id, "idle_timeout_s": SETTINGS.session_idle_s}


def _session(sid: str):
    s = store.get(sid)
    if s is None:
        _error(
            404, "session_not_found", "No such session (it may have expired after 30 minutes idle)."
        )
    return s


@v1.post("/sessions/{sid}/swipes", status_code=202)
async def upload_swipe(sid: str, request: Request):
    length = int(request.headers.get("content-length") or 0)
    if length > SETTINGS.max_body_bytes:
        _error(413, "too_large", f"A swipe may be at most {SETTINGS.max_body_bytes} bytes.")
    try:
        data = await request.json()
    except ValueError:
        _error(400, "not_json", "The body must be one swipe format v1 JSON object.")
    if not isinstance(data, dict):
        _error(400, "not_json", "The body must be one swipe format v1 JSON object.")
    _check_versions(str(data.get("api_version")), data.get("calibration_schema_version"))
    rows = ((data.get("streams") or {}).get("mag") or {}).get("rows") or []
    if len(rows) > SETTINGS.max_samples:
        _error(
            413, "too_many_samples", f"A swipe may have at most {SETTINGS.max_samples} readings."
        )
    try:
        swipe = parse_swipe(data)
    except SwipeFormatError as e:
        _error(422, "invalid_swipe", "; ".join(e.problems))
    s = _session(sid)
    if len(s.swipes) >= SETTINGS.max_swipes:
        _error(409, "too_many_swipes", f"A session may have at most {SETTINGS.max_swipes} swipes.")
    if not s.swipes and swipe.baseline_start_ut is None:
        _error(422, "invalid_swipe", "The first swipe must carry the session's start baseline.")
    try:
        job = store.add_swipe(s, swipe)
    except QueueFull as q:
        _error(
            429,
            "busy",
            "The server is busy; your swipe is safe on the phone. Try again shortly.",
            headers={"Retry-After": str(q.retry_after_s)},
        )
    return {"job_id": job.id, "swipes": job.swipes}


@v1.get("/jobs/{jid}")
def get_job(jid: str):
    job = store.job(jid)
    if job is None:
        _error(404, "job_not_found", "No such job (finished jobs are kept for 10 minutes).")
    out = {"job_id": job.id, "status": job.status, "swipes": job.swipes}
    if job.status == "failed":
        out["error"] = job.error
    if job.status == "done":
        s = store.get(job.session_id)
        out["result"] = s.result if s else None
    return out


@v1.get("/sessions/{sid}/result")
def get_result(sid: str):
    s = _session(sid)
    if s.result is None:
        _error(404, "no_result_yet", "No fit has finished for this session yet.")
    return {"swipes": s.result_swipes, "result": s.result}


@v1.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: str):
    if not store.delete(sid):
        _error(404, "session_not_found", "No such session.")
    return Response(status_code=204)


app.include_router(v1)


@app.exception_handler(HTTPException)
async def _http_error(_: Request, exc: HTTPException):
    body = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": exc.detail}
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)
