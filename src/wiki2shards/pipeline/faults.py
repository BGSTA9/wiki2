"""Fault injection: W2S_FAULT="<stage>:<point>:<n>" kills the coordinator at the n-th hit.

os._exit skips finally blocks, atexit handlers and buffered flushes, exactly like SIGKILL, so the
crash tests see the same on-disk state a real kill would leave behind.
"""

from __future__ import annotations

import os

FAULT_EXIT_CODE = 137
_hits: dict[tuple[str, str], int] = {}


def hit(stage: str, point: str) -> None:
    spec = os.environ.get("W2S_FAULT")
    if not spec:
        return
    want_stage, want_point, n = spec.split(":")
    if (stage, point) != (want_stage, want_point):
        return
    count = _hits.get((stage, point), 0) + 1
    _hits[(stage, point)] = count
    if count == int(n):
        os._exit(FAULT_EXIT_CODE)
