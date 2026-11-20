"""Phase 4 load test: several phones scanning at once against a running server.

Each simulated phone opens a session and uploads swipe format v1 files of a Magpylib magnet,
one every `--gap` seconds (a real swipe takes a few seconds), obeys 429 Retry-After, and
waits for its last fit. Meanwhile the container's memory is sampled with `docker stats`.

  docker run -d --name me-load --cpus=1 --memory=1g -p 8001:8001 magnet-eye-backend
  .venv/bin/python scripts/load_test.py --url http://127.0.0.1:8001 --phones 6 --swipes 10

Units: seconds, MiB.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from app.sim import magnets
from app.sim.phone import NOISE
from app.sim.session import scan_height
from app.sim.swipe_file import ruler_lines, swipe_file

NEW = {"app_version": "load-test", "api_version": "v1", "calibration_schema_version": 1}


def call(url: str, method: str = "GET", body: bytes | None = None):
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read() or b"null"), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null"), e.headers


def session_files(name: str, n: int, seed: int) -> list[bytes]:
    m = magnets.ALL[name]()
    phone = NOISE["realistic"]
    rng = np.random.default_rng(seed)
    h = scan_height(m, phone, m.radius + 0.03)
    z, half = m.top + h, m.radius + 2.5 * h
    lines = ruler_lines(half, z, 0.6 * h, n - 1) + [([0, -half, z], [0, half, z])]
    return [
        json.dumps(swipe_file(m, a, b, i, phone, rng)).encode() for i, (a, b) in enumerate(lines)
    ]


def phone(url: str, files: list[bytes], gap: float, out: dict) -> None:
    status, body, _ = call(f"{url}/v1/sessions", "POST", json.dumps(NEW).encode())
    sid = body["session_id"]
    busy = 0
    job = None
    for f in files:
        while True:
            t0 = time.perf_counter()
            status, body, headers = call(f"{url}/v1/sessions/{sid}/swipes", "POST", f)
            out.setdefault("upload_s", []).append(time.perf_counter() - t0)
            if status == 429:
                busy += 1
                time.sleep(float(headers.get("Retry-After", 2)))
                continue
            assert status == 202, (status, body)
            job = body["job_id"]
            break
        time.sleep(gap)
    t0 = time.perf_counter()
    while True:
        status, body, _ = call(f"{url}/v1/jobs/{job}")
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    out["wait_last_s"] = time.perf_counter() - t0
    out["status"] = body["status"]
    out["busy"] = busy
    if body["status"] == "done":
        r = body["result"]
        out["fit_s"] = r["fit_seconds"]
        out["rel_unc"] = r["relative_uncertainty"]
        out["warnings"] = r["warnings"]
    else:
        out["error"] = body.get("error")
    call(f"{url}/v1/sessions/{sid}", "DELETE")


def sample_memory(container: str, stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        r = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True,
            text=True,
        )
        used = r.stdout.split("/")[0].strip()
        if used:
            v = float(used[:-3])
            samples.append(
                v / 1024 if used.endswith("KiB") else v * 1024 if used.endswith("GiB") else v
            )
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--container", default="me-load")
    ap.add_argument("--phones", type=int, default=6)
    ap.add_argument("--swipes", type=int, default=10)
    ap.add_argument("--gap", type=float, default=3.0, help="seconds between a phone's swipes")
    args = ap.parse_args()

    names = ["disc", "ring", "bar", "broken"]
    print(f"preparing {args.phones} sessions of {args.swipes} swipes...", flush=True)
    sessions = [session_files(names[i % len(names)], args.swipes, i) for i in range(args.phones)]
    mb = sum(len(f) for s in sessions for f in s) / 2**20
    print(f"{mb:.1f} MiB of swipe files", flush=True)

    mem: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(args.container, stop, mem), daemon=True)
    sampler.start()
    time.sleep(1.5)
    idle = min(mem) if mem else float("nan")

    results = [{} for _ in sessions]
    t0 = time.perf_counter()
    threads = [
        threading.Thread(target=phone, args=(args.url, files, args.gap, results[i]))
        for i, files in enumerate(sessions)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    time.sleep(1.0)
    stop.set()

    done = [r for r in results if r.get("status") == "done"]
    failed = [r.get("error") for r in results if r.get("status") != "done"]
    uploads = [u for r in results for u in r.get("upload_s", [])]
    print(f"\nphones {args.phones} x swipes {args.swipes}, gap {args.gap}s, wall {wall:.1f}s")
    print(f"finished {len(done)}/{len(results)}; failed: {failed}")
    print(f"429 answers: {sum(r.get('busy', 0) for r in results)}")
    print(
        f"upload request: median {st.median(uploads):.3f}s, max {max(uploads):.3f}s; "
        f"last fit wait: median {st.median(r['wait_last_s'] for r in done):.2f}s, "
        f"max {max(r['wait_last_s'] for r in done):.2f}s"
    )
    print(f"fit time (final fits): max {max(r['fit_s'] for r in done):.2f}s")
    print(f"memory: idle {idle:.0f} MiB, peak {max(mem):.0f} MiB ({len(mem)} samples)")
    print(f"warnings: {sorted({w for r in done for w in r['warnings']})}")


if __name__ == "__main__":
    main()
