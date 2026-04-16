"""
Retry decorator utility for handling function retries with configurable parameters.

This module provides a flexible retry decorator that can retry function calls
on exceptions with customizable retry count and delay between attempts.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Union, Tuple

logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    delay: float = 0.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    backoff_factor: float = 1.0,
    logger_name: Optional[str] = None
) -> Callable:
    """
    Decorator that retries a function call on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 0.0)
        exceptions: Exception type(s) to catch and retry on (default: Exception)
        backoff_factor: Multiplier for delay on each retry (default: 1.0)
        logger_name: Custom logger name for retry messages (default: None)
    
    Returns:
        Decorated function that implements retry logic
        
    Raises:
        ValueError: If max_retries is negative
        
    Example:
        @retry(max_retries=3, delay=1.0)
        def unstable_function():
            # Function that might fail
            pass
            
        @retry(max_retries=5, delay=0.5, exceptions=(ConnectionError, TimeoutError))
        def network_call():
            # Function that might have network issues
            pass
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    
    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_logger = logging.getLogger(logger_name or func.__module__)
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        current_logger.info(
                            f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                        )
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        current_logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last exception: {type(e).__name__}: {e}"
                        )
                        raise e
                    
                    current_logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Exception: {type(e).__name__}: {e}. Retrying in {current_delay}s..."
                    )
                    
                    if current_delay > 0:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                        
                except Exception as e:
                    # Re-raise exceptions that are not in the retry list
                    current_logger.error(
                        f"Function {func.__name__} failed with non-retryable exception: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise e
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_with_condition(
    condition: Callable[[Exception], bool],
    max_retries: int = 3,
    delay: float = 0.0,
    backoff_factor: float = 1.0
) -> Callable:
    """
    Advanced retry decorator that retries based on a condition function.
    
    Args:
        condition: Function that takes an exception and returns True if retry should occur
        max_retries: Maximum number of retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 0.0)
        backoff_factor: Multiplier for delay on each retry (default: 1.0)
    
    Returns:
        Decorated function that implements conditional retry logic
        
    Example:
        def should_retry(exc):
            return isinstance(exc, ConnectionError) and "timeout" in str(exc).lower()
            
        @retry_with_condition(should_retry, max_retries=5, delay=1.0)
        def network_operation():
            # Function that might timeout
            pass
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries or not condition(e):
                        raise e
                    
                    logger.warning(
                        f"Retrying {func.__name__} (attempt {attempt + 1}/{max_retries + 1}) "
                        f"after {type(e).__name__}: {e}"
                    )
                    
                    if current_delay > 0:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # This should never be reached
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator