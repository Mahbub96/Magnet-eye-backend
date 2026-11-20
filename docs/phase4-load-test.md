# Phase 4 load test

`scripts/load_test.py` against the Docker image with the VPS limits: `docker run --cpus=1
--memory=1g --memory-swap=1g`. Each simulated phone uploads swipe format v1 files of a Magpylib
magnet (disc, ring, bar, broken in turn), one every 3 s, obeys 429 Retry-After and waits for
its last fit. Memory is sampled with `docker stats`. 2026-11-20, Apple silicon host.

| Run | Phones × swipes | Finished | 429 | Slowest final fit | Memory idle → peak |
| --- | --- | --- | --- | --- | --- |
| Before fixes (queue 4) | 6 × 10 | 6/6 | 11 | 1.59 s | 70 → 191 MiB |
| After fixes (queue 12) | 8 × 10 | 8/8 | 0 | 1.14 s | 67 → 198 MiB |

Fixes made from what the runs showed:
- A session has at most one waiting refit (new swipes join it), so the queue size is how many
  phones can scan at once. The default went from 4 to 12.
- The next-swipe search scores every third candidate line first and then refines around the
  best (same line as scoring all, or one at least 98 % as good; `tests/test_next_swipe.py`).
  It took 60 % of a large fit; a 20-swipe two-magnet fit went from 2.7 s to 1.6 s (host,
  one thread).

Single sessions in the container: 10 swipes take 0.3–0.6 s for one simple magnet and about
1.6 s for the broken magnet; 40 swipes or two magnets take several seconds. Those run as jobs
(NF-1 allows it), but one long fit delays every other phone. Deferred (values, not structure):
a fit-time limit, a coarse-then-fine voxel grid, and warm starts (PR-8).

Memory stays under 20 % of the limit, so the 1 GB server has room.
