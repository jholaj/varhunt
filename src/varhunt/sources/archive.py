"""Patience with the public archives, which time out now and then."""

import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

ATTEMPTS = 4
BACKOFF_SECONDS = 15.0


def retried[**P, T](call: Callable[P, T]) -> Callable[P, T]:
    """Repeats a query that failed on the network, waiting longer every time."""

    @functools.wraps(call)
    def patient(*args: P.args, **kwargs: P.kwargs) -> T:
        for attempt in range(1, ATTEMPTS):
            try:
                return call(*args, **kwargs)
            except FileNotFoundError:
                # Data the archive does not have will not appear by asking again.
                raise
            except OSError as exc:
                log.warning("%s failed (attempt %d): %s", call.__name__, attempt, exc)
                time.sleep(BACKOFF_SECONDS * attempt)
        return call(*args, **kwargs)

    return patient
