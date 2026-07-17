"""Coarse-grained method locking shared by the components.

Each component owns one ``threading.RLock`` (``self._lock``) and serializes
every public method behind it. Reentrant because public methods call each
other (``kill_losers`` -> ``kill``, ``reconcile`` -> ``reap_expired``).
Correctness, not parallel throughput: concurrent callers are safe but
serialized.
"""

from __future__ import annotations

import functools


def locked(method):
    """Run ``method`` while holding ``self._lock``."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper
