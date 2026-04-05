"""
utils/retry.py
==============
Provides a ``retry`` decorator that automatically retries a function call
up to *max_attempts* times when it raises an exception, with an optional
fixed delay between attempts.

Typical usage
-------------
::

    from utils.retry import retry

    @retry(max_attempts=3, delay=1.0)
    def flaky_network_call():
        ...
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Optional, Type, Tuple, TypeVar, Any

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 0.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator – retry *func* up to *max_attempts* times on exception.

    Parameters
    ----------
    max_attempts:
        Total number of times the decorated function may be called.
        Must be >= 1.  A value of 1 means no retries (one attempt only).
    delay:
        Seconds to wait between attempts.  Must be >= 0.
    exceptions:
        Tuple of exception types that trigger a retry.  Defaults to
        ``(Exception,)``.  Only exceptions whose type is a subclass of one
        of these will be caught; all others propagate immediately.

    Returns
    -------
    Callable
        The decorated function with retry behaviour applied.

    Raises
    ------
    ValueError
        If *max_attempts* < 1 or *delay* < 0.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay!r}")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[BaseException] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    logger.warning(
                        "Attempt %d/%d for '%s' failed: %s: %s",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        type(exc).__name__,
                        exc,
                    )
                    if attempt < max_attempts and delay > 0:
                        time.sleep(delay)

            raise last_exception from None  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
