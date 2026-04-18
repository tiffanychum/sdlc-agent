"""Retry decorator utility for handling transient failures."""
import functools
import time
from typing import Callable, Tuple, Type, TypeVar

# Use typing_extensions for Python < 3.10 compatibility, or typing for 3.10+
try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

P = ParamSpec('P')
R = TypeVar('R')


def retry(
    max_retries: int = 3,
    delay: float = 0.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that retries a function up to N times on exception.

    Args:
        max_retries: Maximum number of retry attempts after the initial call
            (default: 3). Total attempts = max_retries + 1.
        delay: Delay in seconds between retries (default: 0.0).
        exceptions: Tuple of exception types to catch and retry on
            (default: (Exception,)).

    Returns:
        Decorated function that implements retry logic.

    Raises:
        The last exception if all retries are exhausted.

    Example:
        @retry(max_retries=3, delay=1.0)
        def unstable_api_call():
            # This will be retried up to 3 times on failure
            return requests.get('https://api.example.com/data')

        @retry(max_retries=2, exceptions=(ConnectionError, TimeoutError))
        def network_operation():
            # Only retries on ConnectionError or TimeoutError
            ...
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        raise

            # This should never be reached, but satisfies type checker
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
