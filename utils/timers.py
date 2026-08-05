"""
utils/timers.py
Lightweight performance timers.
"""

import time
from contextlib import contextmanager
from typing import Optional

from utils.logger import get_logger

logger = get_logger("timers")


class Timer:
    def __init__(self, label: str = "op"):
        self.label = label
        self.start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - (self.start or time.perf_counter())
        logger.info("%s took %.3fs", self.label, self.elapsed)


@contextmanager
def timed(label: str = "op"):
    t = Timer(label)
    with t:
        yield t
