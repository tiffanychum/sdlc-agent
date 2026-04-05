"""
utils/retry.py
--------------
Provides the `retry` decorator for automatically retrying a function call
on exception, with configurable maximum attempts and inter-attempt delay.

Typical usage
-------------
    from utils.retry import retry

    @retry(n=3, delay=0.5)
    def fetch_data(url: str) -> dict:
        ...

    # Or applied programmatically:
    result = retry(n=5)(some_flaky_function)(arg1, arg2)
"""

import time
import functools
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])


def retry(n: int, delay: float = 0.0) -> Callable[[F], F]:
    """Return a decorator that retries the wrapped function up to *n* times.

    The wrapped function is called repeatedly until it either:
    - returns a value successfully (that value is returned to the caller), or
    - raises an exception on every one of the *n* allowed attempts, in which
      case the **last** exception is re-raised.

    A ``delay`` of ``0.0`` (the default) means the retry loop is tight with no
    sleep between attempts.  Any positive value causes ``time.sleep(delay)`` to
    be called **between** attempts (i.e. after a failure and before the next
    try), so it is called at most ``n - 1`` times.

    Parameters
    ----------
    n : int
        Maximum number of attempts.  Must be a positive integer (>= 1).
    delay : float, optional
        Seconds to wait between consecutive attempts.  Must be >= 0.
        Defaults to ``0.0``.

    Returns
    -------
    Callable
        A decorator that wraps *any* callable with the retry logic.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer, or if ``delay`` is negative.

    Examples
    --------
    >>> call_count = 0
    >>> @retry(n=3, delay=0)
    ... def sometimes_fails():
    ...     global call_count
    ...     call_count += 1
    ...     if call_count < 3:
    ...         raise RuntimeError("not yet")
    ...     return "ok"
    >>> sometimes_fails()
    'ok'
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: BaseException | None = None
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc
                    if attempt < n and delay > 0:
                        time.sleep(delay)
            # last_exception is always set here because n >= 1, so the loop
            # body ran at least once; the only exit path that doesn't return
            # is when every attempt raised.
            assert last_exception is not None  # narrow type for mypy
            raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator
