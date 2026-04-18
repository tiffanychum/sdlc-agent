"""Tests for the retry decorator utility."""
import time
import pytest
from unittest.mock import Mock, patch
from utils.retry import retry


class CustomError(Exception):
    """Custom exception for testing."""
    pass


class AnotherError(Exception):
    """Another custom exception for testing."""
    pass


def test_retry_success_on_first_attempt():
    """Test that function succeeds on first attempt without retries."""
    mock_func = Mock(return_value="success")
    
    @retry(max_retries=3)
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_success_after_failures():
    """Test that function succeeds after some failures."""
    mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])
    
    @retry(max_retries=3)
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_exhausted_raises_last_exception():
    """Test that last exception is raised when all retries are exhausted."""
    mock_func = Mock(side_effect=Exception("persistent failure"))
    
    @retry(max_retries=2)
    def test_func():
        return mock_func()
    
    with pytest.raises(Exception, match="persistent failure"):
        test_func()
    
    assert mock_func.call_count == 3  # 1 initial + 2 retries


def test_retry_with_zero_retries():
    """Test retry behavior with zero retries (only initial attempt)."""
    mock_func = Mock(side_effect=Exception("immediate failure"))
    
    @retry(max_retries=0)
    def test_func():
        return mock_func()
    
    with pytest.raises(Exception, match="immediate failure"):
        test_func()
    
    assert mock_func.call_count == 1


def test_retry_with_delay():
    """Test that delay is applied between retries."""
    mock_func = Mock(side_effect=[Exception("fail"), "success"])
    
    @retry(max_retries=2, delay=0.1)
    def test_func():
        return mock_func()
    
    start_time = time.time()
    result = test_func()
    end_time = time.time()
    
    assert result == "success"
    assert mock_func.call_count == 2
    # Should have at least one delay of 0.1 seconds
    assert end_time - start_time >= 0.1


def test_retry_specific_exceptions():
    """Test retry only catches specified exception types."""
    mock_func = Mock(side_effect=[CustomError("custom fail"), "success"])
    
    @retry(max_retries=2, exceptions=(CustomError,))
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_ignores_unspecified_exceptions():
    """Test that unspecified exceptions are not caught and retried."""
    mock_func = Mock(side_effect=AnotherError("different error"))
    
    @retry(max_retries=2, exceptions=(CustomError,))
    def test_func():
        return mock_func()
    
    with pytest.raises(AnotherError, match="different error"):
        test_func()
    
    assert mock_func.call_count == 1  # No retries for unspecified exception


def test_retry_multiple_exception_types():
    """Test retry catches multiple specified exception types."""
    mock_func = Mock(side_effect=[CustomError("custom"), AnotherError("another"), "success"])
    
    @retry(max_retries=3, exceptions=(CustomError, AnotherError))
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_preserves_function_metadata():
    """Test that decorator preserves original function metadata."""
    @retry(max_retries=2)
    def original_func():
        """Original function docstring."""
        return "result"
    
    assert original_func.__name__ == "original_func"
    assert original_func.__doc__ == "Original function docstring."


def test_retry_with_function_arguments():
    """Test retry works with functions that have arguments."""
    mock_func = Mock(side_effect=[Exception("fail"), "success"])
    
    @retry(max_retries=2)
    def test_func(arg1, arg2, kwarg1=None):
        mock_func(arg1, arg2, kwarg1=kwarg1)
        return mock_func.return_value
    
    # Set up mock to return success on second call
    mock_func.return_value = "success"
    
    result = test_func("test1", "test2", kwarg1="test3")
    
    assert result == "success"
    assert mock_func.call_count == 2
    # Verify arguments were passed correctly
    mock_func.assert_called_with("test1", "test2", kwarg1="test3")


def test_retry_negative_max_retries_raises_error():
    """Test that negative max_retries raises ValueError."""
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        @retry(max_retries=-1)
        def test_func():
            pass


def test_retry_negative_delay_raises_error():
    """Test that negative delay raises ValueError."""
    with pytest.raises(ValueError, match="delay must be non-negative"):
        @retry(max_retries=1, delay=-0.1)
        def test_func():
            pass


def test_retry_no_delay_between_attempts():
    """Test that no delay is applied when delay=0."""
    mock_func = Mock(side_effect=[Exception("fail"), "success"])
    
    @retry(max_retries=2, delay=0.0)
    def test_func():
        return mock_func()
    
    start_time = time.time()
    result = test_func()
    end_time = time.time()
    
    assert result == "success"
    assert mock_func.call_count == 2
    # Should complete quickly with no delay
    assert end_time - start_time < 0.05


def test_retry_default_parameters():
    """Test retry with default parameters."""
    mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), Exception("fail3"), "success"])
    
    @retry()  # Uses defaults: max_retries=3, delay=0.0, exceptions=(Exception,)
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 4  # 1 initial + 3 retries


def test_retry_with_return_values():
    """Test retry preserves different return value types."""
    test_cases = [
        42,
        "string result",
        [1, 2, 3],
        {"key": "value"},
        None,
        True
    ]
    
    for expected_value in test_cases:
        mock_func = Mock(side_effect=[Exception("fail"), expected_value])
        
        @retry(max_retries=2)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == expected_value


@patch('time.sleep')
def test_retry_delay_timing(mock_sleep):
    """Test that time.sleep is called with correct delay values."""
    mock_func = Mock(side_effect=[Exception("fail1"), Exception("fail2"), "success"])
    
    @retry(max_retries=3, delay=0.5)
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 3
    # Should have called sleep twice (after first two failures)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(0.5)


def test_retry_exception_inheritance():
    """Test retry works with exception inheritance."""
    class BaseCustomError(Exception):
        pass
    
    class DerivedCustomError(BaseCustomError):
        pass
    
    mock_func = Mock(side_effect=[DerivedCustomError("derived error"), "success"])
    
    @retry(max_retries=2, exceptions=(BaseCustomError,))
    def test_func():
        return mock_func()
    
    result = test_func()
    
    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_with_async_compatibility():
    """Test retry decorator doesn't interfere with normal function execution."""
    # This test ensures the decorator works correctly with various function types
    call_count = 0
    
    @retry(max_retries=1)
    def complex_func(x, y=10):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("first attempt fails")
        return x * y + call_count
    
    result = complex_func(5, y=20)
    
    assert result == 102  # 5 * 20 + 2 (call_count)
    assert call_count == 2