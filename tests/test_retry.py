"""
Comprehensive tests for the retry decorator utility.

Tests cover various scenarios including successful retries, maximum retry limits,
different exception types, delay functionality, and edge cases.
"""

import pytest
import time
import logging
from unittest.mock import Mock, patch, call
from typing import List

from utils.retry import retry, retry_with_condition


class TestRetryDecorator:
    """Test cases for the basic retry decorator."""
    
    def test_successful_function_no_retries_needed(self):
        """Test that a successful function executes without retries."""
        call_count = 0
        
        @retry(max_retries=3)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_function()
        assert result == "success"
        assert call_count == 1
    
    def test_function_succeeds_after_retries(self):
        """Test that a function succeeds after some failed attempts."""
        call_count = 0
        
        @retry(max_retries=3)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not ready yet")
            return "success"
        
        result = eventually_successful()
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test that function fails after exceeding max retries."""
        call_count = 0
        
        @retry(max_retries=2)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        
        assert call_count == 3  # Initial call + 2 retries
    
    def test_specific_exception_types(self):
        """Test retry behavior with specific exception types."""
        call_count = 0
        
        @retry(max_retries=2, exceptions=(ValueError, TypeError))
        def specific_exceptions():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First failure")
            elif call_count == 2:
                raise TypeError("Second failure")
            return "success"
        
        result = specific_exceptions()
        assert result == "success"
        assert call_count == 3
    
    def test_non_retryable_exception(self):
        """Test that non-specified exceptions are not retried."""
        call_count = 0
        
        @retry(max_retries=3, exceptions=ValueError)
        def raises_runtime_error():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Not retryable")
        
        with pytest.raises(RuntimeError, match="Not retryable"):
            raises_runtime_error()
        
        assert call_count == 1  # Should not retry
    
    def test_delay_functionality(self):
        """Test that delay works correctly between retries."""
        call_times = []
        
        @retry(max_retries=2, delay=0.1)
        def delayed_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not ready")
            return "success"
        
        start_time = time.time()
        result = delayed_function()
        total_time = time.time() - start_time
        
        assert result == "success"
        assert len(call_times) == 3
        assert total_time >= 0.2  # At least 2 delays of 0.1s each
        
        # Check that delays occurred between calls
        assert call_times[1] - call_times[0] >= 0.1
        assert call_times[2] - call_times[1] >= 0.1
    
    def test_backoff_factor(self):
        """Test exponential backoff functionality."""
        call_times = []
        
        @retry(max_retries=2, delay=0.1, backoff_factor=2.0)
        def backoff_function():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not ready")
            return "success"
        
        result = backoff_function()
        assert result == "success"
        assert len(call_times) == 3
        
        # First delay should be ~0.1s, second delay should be ~0.2s
        first_delay = call_times[1] - call_times[0]
        second_delay = call_times[2] - call_times[1]
        
        assert first_delay >= 0.1
        assert second_delay >= 0.2
        assert second_delay > first_delay
    
    def test_zero_retries(self):
        """Test behavior with zero retries."""
        call_count = 0
        
        @retry(max_retries=0)
        def zero_retries():
            nonlocal call_count
            call_count += 1
            raise ValueError("Immediate failure")
        
        with pytest.raises(ValueError, match="Immediate failure"):
            zero_retries()
        
        assert call_count == 1
    
    def test_negative_retries_raises_error(self):
        """Test that negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            @retry(max_retries=-1)
            def dummy():
                pass
    
    def test_function_with_arguments(self):
        """Test that function arguments are preserved through retries."""
        call_count = 0
        
        @retry(max_retries=2)
        def function_with_args(x, y, z=None):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Not ready")
            return f"{x}-{y}-{z}"
        
        result = function_with_args("a", "b", z="c")
        assert result == "a-b-c"
        assert call_count == 2
    
    def test_function_metadata_preserved(self):
        """Test that function metadata is preserved by the decorator."""
        @retry(max_retries=1)
        def documented_function():
            """This is a test function."""
            return "test"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function."
    
    @patch('utils.retry.logger')
    def test_logging_on_retry(self, mock_logger):
        """Test that appropriate log messages are generated."""
        call_count = 0
        
        @retry(max_retries=2, delay=0.01)
        def logged_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Failure {call_count}")
            return "success"
        
        result = logged_function()
        assert result == "success"
        
        # Should have warning logs for retries and info log for success
        assert mock_logger.warning.call_count == 2
        assert mock_logger.info.call_count == 1
    
    @patch('utils.retry.logger')
    def test_logging_on_final_failure(self, mock_logger):
        """Test logging when all retries are exhausted."""
        @retry(max_retries=1)
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
        
        # Should have warning for retry and error for final failure
        assert mock_logger.warning.call_count == 1
        assert mock_logger.error.call_count == 1
    
    def test_custom_logger_name(self):
        """Test using a custom logger name."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            @retry(max_retries=1, logger_name="custom.logger")
            def test_function():
                raise ValueError("Test")
            
            with pytest.raises(ValueError):
                test_function()
            
            mock_get_logger.assert_called_with("custom.logger")


class TestRetryWithCondition:
    """Test cases for the conditional retry decorator."""
    
    def test_condition_based_retry(self):
        """Test retry based on exception condition."""
        call_count = 0
        
        def should_retry(exc):
            return isinstance(exc, ValueError) and "retryable" in str(exc)
        
        @retry_with_condition(should_retry, max_retries=2)
        def conditional_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("retryable error")
            elif call_count == 2:
                raise ValueError("non-retryable error")
            return "success"
        
        with pytest.raises(ValueError, match="non-retryable error"):
            conditional_function()
        
        assert call_count == 2
    
    def test_condition_prevents_retry(self):
        """Test that condition can prevent retries."""
        call_count = 0
        
        def never_retry(exc):
            return False
        
        @retry_with_condition(never_retry, max_retries=3)
        def no_retry_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Should not retry")
        
        with pytest.raises(ValueError, match="Should not retry"):
            no_retry_function()
        
        assert call_count == 1
    
    def test_condition_with_backoff(self):
        """Test conditional retry with backoff."""
        call_times = []
        
        def always_retry(exc):
            return True
        
        @retry_with_condition(always_retry, max_retries=2, delay=0.1, backoff_factor=2.0)
        def backoff_conditional():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Not ready")
            return "success"
        
        result = backoff_conditional()
        assert result == "success"
        assert len(call_times) == 3
        
        # Verify backoff timing
        first_delay = call_times[1] - call_times[0]
        second_delay = call_times[2] - call_times[1]
        assert first_delay >= 0.1
        assert second_delay >= 0.2
    
    def test_negative_retries_in_conditional(self):
        """Test that negative max_retries raises error in conditional retry."""
        def dummy_condition(exc):
            return True
        
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            @retry_with_condition(dummy_condition, max_retries=-1)
            def dummy():
                pass


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_exception_tuple_handling(self):
        """Test that exception tuples are handled correctly."""
        call_count = 0
        
        @retry(max_retries=2, exceptions=(ValueError, TypeError, RuntimeError))
        def multiple_exception_types():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First")
            elif call_count == 2:
                raise TypeError("Second")
            elif call_count == 3:
                raise RuntimeError("Third")
            return "success"
        
        with pytest.raises(RuntimeError, match="Third"):
            multiple_exception_types()
        
        assert call_count == 3
    
    def test_single_exception_type(self):
        """Test that single exception type (not tuple) works."""
        call_count = 0
        
        @retry(max_retries=1, exceptions=ValueError)
        def single_exception():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            return "success"
        
        result = single_exception()
        assert result == "success"
        assert call_count == 2
    
    def test_zero_delay_with_backoff(self):
        """Test zero delay with backoff factor."""
        call_count = 0
        
        @retry(max_retries=2, delay=0.0, backoff_factor=2.0)
        def zero_delay():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not ready")
            return "success"
        
        start_time = time.time()
        result = zero_delay()
        elapsed = time.time() - start_time
        
        assert result == "success"
        assert call_count == 3
        assert elapsed < 0.1  # Should be very fast with zero delay
    
    def test_large_retry_count(self):
        """Test with a large number of retries."""
        call_count = 0
        
        @retry(max_retries=100)
        def large_retry_count():
            nonlocal call_count
            call_count += 1
            if call_count < 50:
                raise ValueError("Not ready")
            return "success"
        
        result = large_retry_count()
        assert result == "success"
        assert call_count == 50
    
    def test_exception_chaining_preserved(self):
        """Test that exception chaining is preserved."""
        @retry(max_retries=1)
        def chained_exception():
            try:
                raise ValueError("Original")
            except ValueError as e:
                raise RuntimeError("Chained") from e
        
        with pytest.raises(RuntimeError) as exc_info:
            chained_exception()
        
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Original"