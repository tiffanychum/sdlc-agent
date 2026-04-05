"""
Retry decorator utilities for handling transient failures.

Provides both synchronous and asynchronous retry decorators with configurable
max attempts, delay between retries, and exception filtering.
"""

import asyncio
import functools
import time
from typing import Callable, Type, TypeVar, Union, Tuple

F = TypeVar('F', bound=Callable)


def retry(
    max_attempts: int = 3,
    delay: float = 0.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
) -> Callable[[F], F]:
    """
    Decorator that retries a function up to max_attempts times on exception.

    Args:
        max_attempts: Maximum number of attempts (must be >= 1). Default is 3.
        delay: Delay in seconds between retry attempts. Default is 0.
        exceptions: Exception type(s) to catch and retry. Default is Exception (all exceptions).

    Returns:
        Decorated function that will retry on failure.

    Raises:
        The last exception encountered if all retry attempts fail.
        ValueError: If max_attempts < 1.

    Example:
        @retry(max_attempts=5, delay=1.0, exceptions=ConnectionError)
        def fetch_data():
            return requests.get("https://api.example.com/data")
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if delay > 0:
                            time.sleep(delay)
                    # If this was the last attempt, we'll raise below
            
            # All attempts exhausted, raise the last exception
            if last_exception is not None:
                raise last_exception
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


def aretry(
    max_attempts: int = 3,
    delay: float = 0.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
) -> Callable[[F], F]:
    """
    Async decorator that retries an async function up to max_attempts times on exception.

    Args:
        max_attempts: Maximum number of attempts (must be >= 1). Default is 3.
        delay: Delay in seconds between retry attempts. Default is 0.
        exceptions: Exception type(s) to catch and retry. Default is Exception (all exceptions).

    Returns:
        Decorated async function that will retry on failure.

    Raises:
        The last exception encountered if all retry attempts fail.
        ValueError: If max_attempts < 1.

    Example:
        @aretry(max_attempts=5, delay=1.0, exceptions=aiohttp.ClientError)
        async def fetch_data():
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.example.com/data") as resp:
                    return await resp.json()
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        if delay > 0:
                            await asyncio.sleep(delay)
                    # If this was the last attempt, we'll raise below
            
            # All attempts exhausted, raise the last exception
            if last_exception is not None:
                raise last_exception
        
        return wrapper  # type: ignore[return-value]
    
    return decorator
