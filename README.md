# magnet-eye-backend

Reconstruction backend for **Magnet Eye**, a free, open source Android app that shows the 3D
magnetic field around any magnet from a few phone swipes. This repository holds the maths
(Phase 1 of the roadmap). The FastAPI service around it comes in Phase 4.

## What it does

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

Units everywhere: metres, microtesla, A·m², right-handed world frame with z up.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                          # unit + Magpylib accuracy tests
.venv/bin/python scripts/phase1_report.py    # full table -> docs/phase1-results.md
```

## Layout

```
app/pipeline/   production maths (pure functions, no FastAPI)
app/sim/        Magpylib test magnets and a simulated phone (tests and notebooks only)
scripts/        phase1_report.py
tests/          unit tests and pass-bar tests
docs/           phase1-pass-bar.md, phase1-results.md
```

## Licence

MIT
