"""
Tests for retry decorator utilities.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock

from utils.retry import retry, aretry


# ============================================================================
# Tests for retry() decorator (synchronous)
# ============================================================================

def test_retry_success_on_first_attempt():
    """Function succeeds on first attempt, no retries needed."""
    mock_func = Mock(return_value="success")
    decorated = retry()(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_success_on_second_attempt():
    """Function fails once, then succeeds on second attempt."""
    mock_func = Mock(side_effect=[ValueError("fail"), "success"])
    decorated = retry(max_attempts=3)(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_exhausts_all_attempts():
    """Function fails all attempts, raises last exception."""
    mock_func = Mock(side_effect=ValueError("persistent failure"))
    decorated = retry(max_attempts=3)(mock_func)
    
    with pytest.raises(ValueError, match="persistent failure"):
        decorated()
    
    assert mock_func.call_count == 3


def test_retry_with_delay():
    """Verify delay is applied between retry attempts."""
    mock_func = Mock(side_effect=[ValueError("fail"), "success"])
    decorated = retry(max_attempts=3, delay=0.1)(mock_func)
    
    start_time = time.time()
    result = decorated()
    elapsed = time.time() - start_time
    
    assert result == "success"
    assert mock_func.call_count == 2
    # Should have at least one delay of 0.1 seconds
    assert elapsed >= 0.1


def test_retry_specific_exception_type():
    """Only retry on specified exception types."""
    mock_func = Mock(side_effect=TypeError("wrong type"))
    decorated = retry(max_attempts=3, exceptions=ValueError)(mock_func)
    
    # TypeError is not in the exceptions list, should fail immediately
    with pytest.raises(TypeError, match="wrong type"):
        decorated()
    
    assert mock_func.call_count == 1


def test_retry_multiple_exception_types():
    """Retry on multiple specified exception types."""
    mock_func = Mock(side_effect=[ValueError("fail1"), TypeError("fail2"), "success"])
    decorated = retry(max_attempts=5, exceptions=(ValueError, TypeError))(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_preserves_function_metadata():
    """Decorator preserves original function name and docstring."""
    @retry(max_attempts=2)
    def my_function():
        """My docstring."""
        return "result"
    
    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring."


def test_retry_with_args_and_kwargs():
    """Decorator properly forwards arguments to wrapped function."""
    mock_func = Mock(return_value="success")
    decorated = retry()(mock_func)
    
    result = decorated(1, 2, key="value")
    
    assert result == "success"
    mock_func.assert_called_once_with(1, 2, key="value")


def test_retry_invalid_max_attempts():
    """Raises ValueError if max_attempts < 1."""
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        retry(max_attempts=0)


def test_retry_invalid_delay():
    """Raises ValueError if delay is negative."""
    with pytest.raises(ValueError, match="delay must be non-negative"):
        retry(delay=-1.0)


def test_retry_default_parameters():
    """Test default parameters: 3 attempts, no delay, catch all exceptions."""
    mock_func = Mock(side_effect=[Exception("1"), Exception("2"), "success"])
    decorated = retry()(mock_func)
    
    result = decorated()
    
    assert result == "success"
    assert mock_func.call_count == 3


# ============================================================================
# Tests for aretry() decorator (asynchronous)
# ============================================================================

@pytest.mark.asyncio
async def test_aretry_success_on_first_attempt():
    """Async function succeeds on first attempt, no retries needed."""
    mock_func = AsyncMock(return_value="success")
    decorated = aretry()(mock_func)
    
    result = await decorated()
    
    assert result == "success"
    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_aretry_success_on_second_attempt():
    """Async function fails once, then succeeds on second attempt."""
    mock_func = AsyncMock(side_effect=[ValueError("fail"), "success"])
    decorated = aretry(max_attempts=3)(mock_func)
    
    result = await decorated()
    
    assert result == "success"
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_aretry_exhausts_all_attempts():
    """Async function fails all attempts, raises last exception."""
    mock_func = AsyncMock(side_effect=ValueError("persistent failure"))
    decorated = aretry(max_attempts=3)(mock_func)
    
    with pytest.raises(ValueError, match="persistent failure"):
        await decorated()
    
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_aretry_with_delay():
    """Verify async delay is applied between retry attempts."""
    mock_func = AsyncMock(side_effect=[ValueError("fail"), "success"])
    decorated = aretry(max_attempts=3, delay=0.1)(mock_func)
    
    start_time = time.time()
    result = await decorated()
    elapsed = time.time() - start_time
    
    assert result == "success"
    assert mock_func.call_count == 2
    # Should have at least one delay of 0.1 seconds
    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_aretry_specific_exception_type():
    """Only retry on specified exception types for async functions."""
    mock_func = AsyncMock(side_effect=TypeError("wrong type"))
    decorated = aretry(max_attempts=3, exceptions=ValueError)(mock_func)
    
    # TypeError is not in the exceptions list, should fail immediately
    with pytest.raises(TypeError, match="wrong type"):
        await decorated()
    
    assert mock_func.call_count == 1


@pytest.mark.asyncio
async def test_aretry_multiple_exception_types():
    """Retry on multiple specified exception types for async functions."""
    mock_func = AsyncMock(side_effect=[ValueError("fail1"), TypeError("fail2"), "success"])
    decorated = aretry(max_attempts=5, exceptions=(ValueError, TypeError))(mock_func)
    
    result = await decorated()
    
    assert result == "success"
    assert mock_func.call_count == 3


@pytest.mark.asyncio
async def test_aretry_preserves_function_metadata():
    """Async decorator preserves original function name and docstring."""
    @aretry(max_attempts=2)
    async def my_async_function():
        """My async docstring."""
        return "result"
    
    assert my_async_function.__name__ == "my_async_function"
    assert my_async_function.__doc__ == "My async docstring."


@pytest.mark.asyncio
async def test_aretry_with_args_and_kwargs():
    """Async decorator properly forwards arguments to wrapped function."""
    mock_func = AsyncMock(return_value="success")
    decorated = aretry()(mock_func)
    
    result = await decorated(1, 2, key="value")
    
    assert result == "success"
    mock_func.assert_called_once_with(1, 2, key="value")


def test_aretry_invalid_max_attempts():
    """Raises ValueError if max_attempts < 1 for async decorator."""
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        aretry(max_attempts=0)


def test_aretry_invalid_delay():
    """Raises ValueError if delay is negative for async decorator."""
    with pytest.raises(ValueError, match="delay must be non-negative"):
        aretry(delay=-1.0)


@pytest.mark.asyncio
async def test_aretry_default_parameters():
    """Test default parameters for async: 3 attempts, no delay, catch all exceptions."""
    mock_func = AsyncMock(side_effect=[Exception("1"), Exception("2"), "success"])
    decorated = aretry()(mock_func)
    
    result = await decorated()
    
    assert result == "success"
    assert mock_func.call_count == 3


# ============================================================================
# Integration tests with real functions
# ============================================================================

def test_retry_with_real_function():
    """Integration test with a real function that tracks state."""
    class Counter:
        def __init__(self):
            self.count = 0
        
        def increment_and_fail_twice(self):
            self.count += 1
            if self.count < 3:
                raise ValueError(f"Attempt {self.count} failed")
            return f"Success on attempt {self.count}"
    
    counter = Counter()
    decorated = retry(max_attempts=5)(counter.increment_and_fail_twice)
    
    result = decorated()
    
    assert result == "Success on attempt 3"
    assert counter.count == 3


@pytest.mark.asyncio
async def test_aretry_with_real_async_function():
    """Integration test with a real async function that tracks state."""
    class AsyncCounter:
        def __init__(self):
            self.count = 0
        
        async def increment_and_fail_twice(self):
            self.count += 1
            await asyncio.sleep(0.01)  # Simulate async work
            if self.count < 3:
                raise ValueError(f"Attempt {self.count} failed")
            return f"Success on attempt {self.count}"
    
    counter = AsyncCounter()
    decorated = aretry(max_attempts=5)(counter.increment_and_fail_twice)
    
    result = await decorated()
    
    assert result == "Success on attempt 3"
    assert counter.count == 3
