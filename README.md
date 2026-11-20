# magnet-eye-backend

Reconstruction backend for **Magnet Eye**, a free, open source Android app that shows the 3D
magnetic field around any magnet from a few phone swipes. This repository holds the maths
and the FastAPI service around it (Phases 1 and 4 of the roadmap).

## What it does

0. Reads each uploaded swipe file (swipe format v1), puts every stream on the magnetometer's
   timeline, recomputes the iron correction and subtracts the session baseline in the world
   frame (`app/pipeline/swipe_format.py`, `align.py`).
1. Rotates each reading into a fixed world frame and places the sensor inside the phone
   (`app/pipeline/frames.py`).
2. Bins readings along each swipe and gives each one a noise level from sensor noise, tracking
   jitter and orientation jitter (`observations.py`).
3. Finds the magnet with a single-dipole fit, then sizes a voxel grid from the data and the
   table plane (`locate.py`).
4. Solves for a free magnetic moment in every voxel, with no shape assumed. Errors shared by
   a whole swipe (position, orientation and field offsets) are modelled as correlated noise,
   and the regularisation strength comes from the data (`inverse.py`).
5. Reports the field, its uncertainty and the next swipe that would reduce that uncertainty
   most while staying clear of sensor saturation (`uncertainty.py`, `reconstruct.py`).

Units everywhere: metres, microtesla, A·m², seconds (from integer nanoseconds), radians,
right-handed world frame with z up.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                          # unit + Magpylib accuracy tests
.venv/bin/python scripts/phase1_report.py    # full table -> docs/phase1-results.md
```

API server for development (reloads on every save; `--host 0.0.0.0` lets a phone on the same
Wi-Fi reach it, port 8001 keeps clear of other local servers):

```bash
.venv/bin/uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8001
```

Endpoints, all under `/v1` (SRS 4.3): `GET /health`, `POST /sessions`,
`POST /sessions/{id}/swipes` (one swipe format v1 file, returns a job), `GET /jobs/{id}`,
`GET /sessions/{id}/result`, `DELETE /sessions/{id}`. Interactive docs at `/docs`.

The service keeps sessions in memory only and forgets them after 30 minutes idle. It runs
one fit at a time; when its queue is full it answers 429 with `Retry-After`. It refuses files
from a different API or calibration schema version with 409, and never logs sample data.
Limits are set with environment variables (`MAGNET_EYE_QUEUE_SIZE`, `MAGNET_EYE_MAX_SWIPES`,
`MAGNET_EYE_MAX_SAMPLES`, `MAGNET_EYE_SESSION_IDLE_S`, `MAGNET_EYE_MAX_VOXELS`; see
`app/api/config.py`).

## Layout

```
app/api/        FastAPI service: endpoints, in-memory sessions, the single fit worker
app/pipeline/   production maths (pure functions, no FastAPI)
app/sim/        Magpylib test magnets, a simulated phone and simulated swipe files (tests only)
scripts/        phase1_report.py
tests/          unit tests and pass-bar tests
docs/           phase1-pass-bar.md, phase1-results.md
```

## Licence

MIT
