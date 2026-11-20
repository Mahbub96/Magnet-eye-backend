"""Service limits (PR-13), read once from the environment so a deployment can tune them.

Defaults suit a 1-core, 1 GB server. Every name is prefixed MAGNET_EYE_.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

API_VERSION = "v1"
CALIBRATION_SCHEMA_VERSION = 1


def _int(name: str, default: int) -> int:
    return int(os.environ.get(f"MAGNET_EYE_{name}", default))


@dataclass(frozen=True)
class Settings:
    queue_size: int = _int("QUEUE_SIZE", 4)  # waiting fits; one more gets HTTP 429
    session_idle_s: int = _int("SESSION_IDLE_S", 30 * 60)  # PR-12: forget idle sessions
    max_swipes: int = _int("MAX_SWIPES", 40)  # per session
    max_samples: int = _int("MAX_SAMPLES", 30_000)  # magnetometer rows per swipe
    max_body_bytes: int = _int("MAX_BODY_BYTES", 16 * 1024 * 1024)
    max_voxels: int = _int("MAX_VOXELS", 600)  # PR-25
    job_keep_s: int = _int("JOB_KEEP_S", 10 * 60)  # finished jobs stay pollable this long


SETTINGS = Settings()
