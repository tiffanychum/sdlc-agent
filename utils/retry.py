"""
utils/retry.py
--------------
Provides the `retry` decorator for automatically retrying a function call
on exception.

Usage example::

    @retry(times=3, delay=0.5, exceptions=(ValueError, IOError))
    def flaky_operation():
        ...

"""

import time
import functools
import logging
from typing import Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    times: int = 3,
    delay: float = 0.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """Decorator that retries the wrapped function up to *times* attempts.

    Parameters
    ----------
    times : int
        Maximum number of attempts (must be >= 1).  The first call counts as
        attempt #1, so the function is retried at most ``times - 1`` extra
        times.
    delay : float
        Seconds to wait between attempts.  Must be >= 0.  Defaults to 0.
    exceptions : tuple of exception types
        Only retry when one of these exception types is raised.  Any other
        exception propagates immediately without retrying.  Defaults to
        ``(Exception,)``.

    Returns
    -------
    Callable
        A decorator that wraps the target function with retry logic.

    Raises
    ------
    ValueError
        If *times* < 1 or *delay* < 0.
    TypeError
        If *exceptions* is not a tuple of exception types.
    """
    if not isinstance(times, int) or times < 1:
        raise ValueError(f"`times` must be an integer >= 1, got {times!r}")
    if delay < 0:
        raise ValueError(f"`delay` must be >= 0, got {delay!r}")
    if not isinstance(exceptions, tuple) or not exceptions:
        raise TypeError(
            "`exceptions` must be a non-empty tuple of exception types, "
            f"got {exceptions!r}"
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: BaseException | None = None

            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "Attempt %d/%d for %r failed with %s: %s",
                        attempt,
                        times,
                        func.__qualname__,
                        type(exc).__name__,
                        exc,
                    )
                    if attempt < times:
                        if delay > 0:
                            time.sleep(delay)
                    # Non-matching exceptions propagate immediately (not caught)

            # All attempts exhausted — re-raise the last captured exception
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
