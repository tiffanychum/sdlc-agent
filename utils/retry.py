"""
Retry decorator utility for handling transient failures.

This module provides a simple retry decorator that can be used to automatically
retry function calls that fail due to exceptions, with configurable maximum
attempts and optional delays between retries.

Example usage:
    @retry(max_attempts=3, delay=1.0)
    def unreliable_api_call():
        # Function that might fail
        response = requests.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Union, Tuple


def retry(
    max_attempts: int = 3,
    delay: Optional[float] = None,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    backoff_factor: float = 1.0
) -> Callable:
    """
    Decorator that retries a function up to max_attempts times on specified exceptions.
    
    Args:
        max_attempts (int): Maximum number of attempts (including the first call).
                           Must be >= 1. Default is 3.
        delay (Optional[float]): Fixed delay in seconds between retries.
                               If None, no delay is applied. Default is None.
        exceptions (Union[Type[Exception], Tuple[Type[Exception], ...]]): 
                   Exception type(s) to catch and retry on. Default is Exception.
        backoff_factor (float): Multiplier for delay on each retry.
                              Only used if delay is specified. Default is 1.0.
    
    Returns:
        Callable: The decorated function with retry logic.
        
    Raises:
        ValueError: If max_attempts < 1 or delay < 0 or backoff_factor <= 0.
        The last exception encountered: If all retry attempts are exhausted.
        
    Example:
        @retry(max_attempts=3, delay=1.0)
        def flaky_function():
            # This will be retried up to 3 times with 1 second delay
            if random.random() < 0.7:
                raise ConnectionError("Network error")
            return "Success"
            
        @retry(max_attempts=5, delay=0.5, exceptions=(requests.RequestException,))
        def api_call():
            # Only retries on requests.RequestException and its subclasses
            return requests.get("https://api.example.com").json()
    """
    # Validate parameters
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if delay is not None and delay < 0:
        raise ValueError("delay must be >= 0")
    if backoff_factor <= 0:
        raise ValueError("backoff_factor must be > 0")
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # If this was the last attempt, re-raise the exception
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Apply delay if specified (but not after the last failed attempt)
                    if current_delay is not None and current_delay > 0:
                        time.sleep(current_delay)
                        # Apply backoff factor for next delay
                        current_delay *= backoff_factor
            
            # This should never be reached due to the raise in the except block,
            # but included for completeness
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
) -> Callable:
    """
    Convenience decorator for exponential backoff retry pattern.
    
    This is a specialized version of the retry decorator that implements
    exponential backoff with a maximum delay cap.
    
    Args:
        max_attempts (int): Maximum number of attempts. Default is 3.
        initial_delay (float): Initial delay in seconds. Default is 1.0.
        max_delay (float): Maximum delay cap in seconds. Default is 60.0.
        backoff_factor (float): Exponential backoff multiplier. Default is 2.0.
        exceptions: Exception type(s) to catch and retry on. Default is Exception.
        
    Returns:
        Callable: The decorated function with exponential backoff retry logic.
        
    Example:
        @retry_with_exponential_backoff(max_attempts=5, initial_delay=0.5)
        def database_operation():
            # Retries with delays: 0.5s, 1.0s, 2.0s, 4.0s
            return db.execute_query()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = initial_delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Sleep with current delay, then calculate next delay
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff_factor, max_delay)
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator