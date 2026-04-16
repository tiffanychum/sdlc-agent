"""
Tests for the retry decorator utility.

This module contains comprehensive tests for the retry() decorator and
retry_with_exponential_backoff() decorator, covering happy path, edge cases,
error conditions, and timing behavior.
"""

import pytest
import time
from unittest.mock import Mock, patch
from utils.retry import retry, retry_with_exponential_backoff


class TestRetryDecorator:
    """Test cases for the basic retry() decorator."""
    
    def test_successful_function_no_retries_needed(self):
        """Test that successful functions execute normally without retries."""
        mock_func = Mock(return_value="success")
        
        @retry(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_function_succeeds_after_retries(self):
        """Test that function succeeds after some failures."""
        call_count = 0
        
        @retry(max_attempts=3)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_max_attempts_exhausted_raises_last_exception(self):
        """Test that last exception is raised when all attempts fail."""
        @retry(max_attempts=2)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
    
    def test_retry_with_delay(self):
        """Test that delay is applied between retries."""
        call_times = []
        
        @retry(max_attempts=3, delay=0.1)
        def timed_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise RuntimeError("Fail")
            return "success"
        
        start_time = time.time()
        result = timed_func()
        total_time = time.time() - start_time
        
        assert result == "success"
        assert len(call_times) == 3
        # Should have at least 2 delays of 0.1 seconds each
        assert total_time >= 0.2
        # Check delays between calls
        assert call_times[1] - call_times[0] >= 0.1
        assert call_times[2] - call_times[1] >= 0.1
    
    def test_retry_with_backoff_factor(self):
        """Test that backoff factor increases delay on each retry."""
        call_times = []
        
        @retry(max_attempts=4, delay=0.1, backoff_factor=2.0)
        def backoff_func():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise RuntimeError("Fail")
            return "success"
        
        result = backoff_func()
        
        assert result == "success"
        assert len(call_times) == 4
        # Delays should be: 0.1, 0.2, 0.4
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]
        
        assert 0.09 <= delay1 <= 0.15  # ~0.1s with tolerance
        assert 0.18 <= delay2 <= 0.25  # ~0.2s with tolerance
        assert 0.35 <= delay3 <= 0.45  # ~0.4s with tolerance
    
    def test_specific_exception_types(self):
        """Test retry only catches specified exception types."""
        @retry(max_attempts=3, exceptions=ValueError)
        def specific_exception_func():
            raise TypeError("Wrong exception type")
        
        # Should not retry TypeError, should raise immediately
        with pytest.raises(TypeError, match="Wrong exception type"):
            specific_exception_func()
    
    def test_multiple_exception_types(self):
        """Test retry catches multiple specified exception types."""
        call_count = 0
        
        @retry(max_attempts=3, exceptions=(ValueError, ConnectionError))
        def multi_exception_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First error")
            elif call_count == 2:
                raise ConnectionError("Second error")
            return "success"
        
        result = multi_exception_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_function_with_arguments(self):
        """Test retry works with functions that have arguments."""
        call_count = 0
        
        @retry(max_attempts=2)
        def func_with_args(x, y, z=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Fail once")
            return f"{x}-{y}-{z}"
        
        result = func_with_args("a", "b", z="c")
        
        assert result == "a-b-c"
        assert call_count == 2
    
    def test_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""
        @retry(max_attempts=2)
        def documented_func():
            """This is a test function."""
            return "test"
        
        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a test function."


class TestRetryParameterValidation:
    """Test parameter validation for retry decorator."""
    
    def test_invalid_max_attempts_zero(self):
        """Test that max_attempts=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            @retry(max_attempts=0)
            def test_func():
                pass
    
    def test_invalid_max_attempts_negative(self):
        """Test that negative max_attempts raises ValueError."""
        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            @retry(max_attempts=-1)
            def test_func():
                pass
    
    def test_invalid_delay_negative(self):
        """Test that negative delay raises ValueError."""
        with pytest.raises(ValueError, match="delay must be >= 0"):
            @retry(max_attempts=2, delay=-1.0)
            def test_func():
                pass
    
    def test_invalid_backoff_factor_zero(self):
        """Test that backoff_factor=0 raises ValueError."""
        with pytest.raises(ValueError, match="backoff_factor must be > 0"):
            @retry(max_attempts=2, backoff_factor=0)
            def test_func():
                pass
    
    def test_invalid_backoff_factor_negative(self):
        """Test that negative backoff_factor raises ValueError."""
        with pytest.raises(ValueError, match="backoff_factor must be > 0"):
            @retry(max_attempts=2, backoff_factor=-1.0)
            def test_func():
                pass
    
    def test_valid_delay_zero(self):
        """Test that delay=0 is valid (no delay)."""
        call_count = 0
        
        @retry(max_attempts=2, delay=0.0)
        def zero_delay_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Fail once")
            return "success"
        
        result = zero_delay_func()
        assert result == "success"
        assert call_count == 2


class TestRetryEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_max_attempts_one(self):
        """Test retry with max_attempts=1 (no retries)."""
        @retry(max_attempts=1)
        def single_attempt_func():
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError, match="Always fails"):
            single_attempt_func()
    
    def test_no_delay_specified(self):
        """Test retry without delay (delay=None)."""
        call_count = 0
        
        @retry(max_attempts=3, delay=None)
        def no_delay_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Fail")
            return "success"
        
        start_time = time.time()
        result = no_delay_func()
        total_time = time.time() - start_time
        
        assert result == "success"
        assert call_count == 3
        # Should complete quickly without delays
        assert total_time < 0.1
    
    def test_exception_not_in_retry_list(self):
        """Test that exceptions not in the retry list are not caught."""
        @retry(max_attempts=3, exceptions=ValueError)
        def wrong_exception_func():
            raise KeyError("Not retried")
        
        with pytest.raises(KeyError, match="Not retried"):
            wrong_exception_func()


class TestRetryWithExponentialBackoff:
    """Test cases for retry_with_exponential_backoff decorator."""
    
    def test_exponential_backoff_timing(self):
        """Test that exponential backoff increases delays correctly."""
        call_times = []
        
        @retry_with_exponential_backoff(
            max_attempts=4, 
            initial_delay=0.1, 
            backoff_factor=2.0
        )
        def backoff_func():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise RuntimeError("Fail")
            return "success"
        
        result = backoff_func()
        
        assert result == "success"
        assert len(call_times) == 4
        
        # Check exponential backoff: 0.1, 0.2, 0.4
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]
        
        assert 0.09 <= delay1 <= 0.15  # ~0.1s
        assert 0.18 <= delay2 <= 0.25  # ~0.2s
        assert 0.35 <= delay3 <= 0.45  # ~0.4s
    
    def test_max_delay_cap(self):
        """Test that max_delay caps the exponential growth."""
        call_times = []
        
        @retry_with_exponential_backoff(
            max_attempts=5,
            initial_delay=0.1,
            max_delay=0.2,
            backoff_factor=3.0
        )
        def capped_delay_func():
            call_times.append(time.time())
            if len(call_times) < 5:
                raise RuntimeError("Fail")
            return "success"
        
        result = capped_delay_func()
        
        assert result == "success"
        assert len(call_times) == 5
        
        # Delays should be: 0.1, 0.2 (capped), 0.2 (capped), 0.2 (capped)
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        delay3 = call_times[3] - call_times[2]
        delay4 = call_times[4] - call_times[3]
        
        assert 0.09 <= delay1 <= 0.15  # ~0.1s
        assert 0.18 <= delay2 <= 0.25  # ~0.2s (capped)
        assert 0.18 <= delay3 <= 0.25  # ~0.2s (capped)
        assert 0.18 <= delay4 <= 0.25  # ~0.2s (capped)
    
    def test_exponential_backoff_with_specific_exceptions(self):
        """Test exponential backoff with specific exception types."""
        call_count = 0
        
        @retry_with_exponential_backoff(
            max_attempts=3,
            initial_delay=0.05,
            exceptions=ConnectionError
        )
        def specific_exception_backoff():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network issue")
            return "connected"
        
        result = specific_exception_backoff()
        
        assert result == "connected"
        assert call_count == 3


class TestRetryIntegration:
    """Integration tests combining multiple features."""
    
    def test_nested_decorators(self):
        """Test retry decorator works with other decorators."""
        def logging_decorator(func):
            def wrapper(*args, **kwargs):
                print(f"Calling {func.__name__}")
                return func(*args, **kwargs)
            return wrapper
        
        call_count = 0
        
        @logging_decorator
        @retry(max_attempts=2)
        def nested_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Fail once")
            return "success"
        
        result = nested_func()
        
        assert result == "success"
        assert call_count == 2
    
    @patch('time.sleep')
    def test_retry_with_mocked_sleep(self, mock_sleep):
        """Test retry behavior with mocked sleep for faster testing."""
        call_count = 0
        
        @retry(max_attempts=3, delay=1.0)
        def mock_sleep_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Fail")
            return "success"
        
        result = mock_sleep_func()
        
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2  # 2 delays between 3 attempts
        mock_sleep.assert_called_with(1.0)