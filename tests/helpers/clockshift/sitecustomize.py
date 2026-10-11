"""Shift the wall clock of a subprocess by W2S_TEST_CLOCK_OFFSET_S seconds.

Put this directory on PYTHONPATH for the second build in test_build_twice_identical: any wall-clock
value that leaks into an artifact (a timestamp, a gzip mtime) then differs between the two builds
every time, instead of only when they happen to straddle a second boundary.
"""

import os
import time

_offset = int(os.environ.get("W2S_TEST_CLOCK_OFFSET_S", "0"))
if _offset:
    _time, _time_ns = time.time, time.time_ns
    time.time = lambda: _time() + _offset
    time.time_ns = lambda: _time_ns() + _offset * 1_000_000_000
