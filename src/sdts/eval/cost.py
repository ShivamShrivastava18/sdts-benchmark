"""Cost: wall-clock seconds for fit and sample, and peak resident set size.

``track()`` is a context manager that records elapsed wall time and the
process peak RSS (``ru_maxrss``, normalised to MB on both Linux and
macOS) at exit. Peak RSS is a process-lifetime maximum, so it is the peak
up to and including the tracked block."""

from __future__ import annotations

import contextlib
import resource
import sys
import time
from typing import Iterator


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


class Timing:
    seconds: float = 0.0
    peak_rss_mb: float = 0.0


@contextlib.contextmanager
def track() -> Iterator[Timing]:
    t = Timing()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start
        t.peak_rss_mb = peak_rss_mb()
