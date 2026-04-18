"""Tests for the retry decorator utility."""
import pytest
from unittest.mock import patch

from utils.retry import retry


class TestRetryDecorator:
    """Test cases for the retry() decorator."""

    def test_successful_execution_no_retry_needed(self):
        """Function succeeds on first attempt - no retry triggered."""
        call_count = 0

        @retry(max_retries=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_fails_then_succeeds_within_retry_limit(self):
        """Function fails initially but succeeds within retry limit."""
        call_count = 0

        @retry(max_retries=3)
        def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = fails_twice_then_succeeds()
        assert result == "success"
        assert call_count == 3

    def test_exhausts_all_retries_raises_exception(self):
        """Function exhausts all retries and raises the last exception."""
        call_count = 0

        @retry(max_retries=2)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Persistent failure")

        with pytest.raises(RuntimeError, match="Persistent failure"):
            always_fails()

        # Initial attempt + 2 retries = 3 total calls
        assert call_count == 3

    def test_delay_between_retries(self):
        """Verify delay is applied between retry attempts."""
        call_count = 0

        @retry(max_retries=2, delay=0.5)
        def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail once")
            return "success"

        with patch('utils.retry.time.sleep') as mock_sleep:
            result = fails_then_succeeds()
            assert result == "success"
            mock_sleep.assert_called_once_with(0.5)

    def test_no_delay_when_zero(self):
        """No sleep called when delay is 0."""
        call_count = 0

        @retry(max_retries=2, delay=0.0)
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        with patch('utils.retry.time.sleep') as mock_sleep:
            fails_once()
            mock_sleep.assert_not_called()

    def test_max_retries_zero(self):
        """With max_retries=0, function runs once and raises on failure."""
        call_count = 0

        @retry(max_retries=0)
        def fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Immediate failure")

        with pytest.raises(ValueError, match="Immediate failure"):
            fails()

        assert call_count == 1

    def test_max_retries_zero_success(self):
        """With max_retries=0, successful function works normally."""
        @retry(max_retries=0)
        def succeeds():
            return "ok"

        assert succeeds() == "ok"

    def test_max_retries_one(self):
        """With max_retries=1, function gets one retry after initial failure."""
        call_count = 0

        @retry(max_retries=1)
        def fails_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success"

        result = fails_once()
        assert result == "success"
        assert call_count == 2

    def test_specific_exception_types_caught(self):
        """Only specified exception types trigger retry."""
        call_count = 0

        @retry(max_retries=3, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not caught")

        with pytest.raises(TypeError):
            raises_type_error()

        # No retry for TypeError - only 1 call
        assert call_count == 1

    def test_specific_exception_types_retried(self):
        """Specified exception types do trigger retry."""
        call_count = 0

        @retry(max_retries=3, exceptions=(ValueError,))
        def raises_value_error_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Caught and retried")
            return "success"

        result = raises_value_error_then_succeeds()
        assert result == "success"
        assert call_count == 2

    def test_multiple_exception_types(self):
        """Multiple exception types can be specified."""
        call_count = 0

        @retry(max_retries=3, exceptions=(ValueError, TypeError))
        def alternating_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First error")
            elif call_count == 2:
                raise TypeError("Second error")
            return "success"

        result = alternating_errors()
        assert result == "success"
        assert call_count == 3

    def test_preserves_function_metadata(self):
        """Decorator preserves original function metadata."""
        @retry(max_retries=3)
        def documented_function():
            """This is the docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."

    def test_passes_positional_arguments(self):
        """Positional arguments are passed correctly to decorated function."""
        @retry(max_retries=3)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3
        assert add(10, 20) == 30

    def test_passes_keyword_arguments(self):
        """Keyword arguments are passed correctly to decorated function."""
        @retry(max_retries=3)
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        assert greet("World") == "Hello, World!"
        assert greet("World", greeting="Hi") == "Hi, World!"
        assert greet(name="Alice", greeting="Hey") == "Hey, Alice!"

    def test_passes_mixed_arguments(self):
        """Mixed positional and keyword arguments work correctly."""
        @retry(max_retries=3)
        def calculate(a, b, c=0, d=0):
            return a + b + c + d

        assert calculate(1, 2) == 3
        assert calculate(1, 2, c=3) == 6
        assert calculate(1, 2, 3, d=4) == 10

    def test_negative_max_retries_raises_error(self):
        """Negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            @retry(max_retries=-1)
            def func():
                pass

    def test_negative_delay_raises_error(self):
        """Negative delay raises ValueError."""
        with pytest.raises(ValueError, match="delay must be non-negative"):
            @retry(max_retries=3, delay=-1.0)
            def func():
                pass

    def test_exception_message_preserved(self):
        """The original exception message is preserved when re-raised."""
        @retry(max_retries=1)
        def fails_with_message():
            raise ValueError("Specific error message 12345")

        with pytest.raises(ValueError) as exc_info:
            fails_with_message()

        assert "Specific error message 12345" in str(exc_info.value)

    def test_exception_type_preserved(self):
        """The original exception type is preserved when re-raised."""
        class CustomError(Exception):
            pass

        @retry(max_retries=1, exceptions=(CustomError,))
        def raises_custom():
            raise CustomError("Custom!")

        with pytest.raises(CustomError):
            raises_custom()

    def test_return_value_types(self):
        """Various return value types are handled correctly."""
        @retry(max_retries=1)
        def return_none():
            return None

        @retry(max_retries=1)
        def return_list():
            return [1, 2, 3]

        @retry(max_retries=1)
        def return_dict():
            return {"key": "value"}

        assert return_none() is None
        assert return_list() == [1, 2, 3]
        assert return_dict() == {"key": "value"}
