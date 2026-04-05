"""
tests/test_retry.py
===================
Pytest test-suite for :func:`utils.retry.retry`.

Coverage targets
----------------
* Successful call on the first attempt.
* Success after N-1 failures (function eventually succeeds).
* Failure after exhausting all retries (exception is re-raised).
* Correct total number of call attempts.
* Delay behaviour (``time.sleep`` is called with the right arguments).
* Edge case: ``max_attempts=1`` (no retries, single attempt only).
* Parameter validation: ``max_attempts < 1`` and ``delay < 0``.
* Non-retried exceptions pass through immediately.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call, patch

from utils.retry import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CustomError(Exception):
    """Distinct error type used in selectivity tests."""


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestRetrySuccess:
    def test_succeeds_on_first_attempt(self):
        """Function that never raises should return its value immediately."""
        mock_fn = MagicMock(return_value=42)
        decorated = retry(max_attempts=3, delay=0)(mock_fn)

        result = decorated()

        assert result == 42
        mock_fn.assert_called_once()

    def test_succeeds_after_n_minus_one_failures(self):
        """Function that fails twice then succeeds should return the success value."""
        side_effects = [ValueError("fail"), ValueError("fail"), "ok"]
        mock_fn = MagicMock(side_effect=side_effects)
        decorated = retry(max_attempts=3, delay=0)(mock_fn)

        result = decorated()

        assert result == "ok"
        assert mock_fn.call_count == 3

    def test_passes_args_and_kwargs_correctly(self):
        """Positional and keyword arguments must be forwarded unchanged."""
        mock_fn = MagicMock(return_value="done")
        decorated = retry(max_attempts=2, delay=0)(mock_fn)

        decorated("a", "b", key="value")

        mock_fn.assert_called_once_with("a", "b", key="value")


# ---------------------------------------------------------------------------
# Failure / exhaustion tests
# ---------------------------------------------------------------------------

class TestRetryExhaustion:
    def test_raises_last_exception_after_all_attempts(self):
        """After max_attempts failures the last exception should propagate."""
        mock_fn = MagicMock(side_effect=RuntimeError("boom"))
        decorated = retry(max_attempts=3, delay=0)(mock_fn)

        with pytest.raises(RuntimeError, match="boom"):
            decorated()

    def test_correct_number_of_attempts_on_total_failure(self):
        """The decorated function must be called exactly max_attempts times."""
        mock_fn = MagicMock(side_effect=OSError("io error"))
        max_attempts = 5
        decorated = retry(max_attempts=max_attempts, delay=0)(mock_fn)

        with pytest.raises(OSError):
            decorated()

        assert mock_fn.call_count == max_attempts

    def test_max_attempts_one_does_not_retry(self):
        """With max_attempts=1 the function is called exactly once; no retry."""
        mock_fn = MagicMock(side_effect=ValueError("single shot"))
        decorated = retry(max_attempts=1, delay=0)(mock_fn)

        with pytest.raises(ValueError, match="single shot"):
            decorated()

        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Delay behaviour tests
# ---------------------------------------------------------------------------

class TestRetryDelay:
    @patch("utils.retry.time.sleep")
    def test_sleep_called_between_attempts(self, mock_sleep: MagicMock):
        """time.sleep must be called (max_attempts - 1) times with *delay* seconds."""
        mock_fn = MagicMock(side_effect=Exception("err"))
        delay = 0.5
        max_attempts = 4
        decorated = retry(max_attempts=max_attempts, delay=delay)(mock_fn)

        with pytest.raises(Exception):
            decorated()

        expected_calls = [call(delay)] * (max_attempts - 1)
        mock_sleep.assert_has_calls(expected_calls)
        assert mock_sleep.call_count == max_attempts - 1

    @patch("utils.retry.time.sleep")
    def test_no_sleep_when_delay_is_zero(self, mock_sleep: MagicMock):
        """time.sleep must NOT be called when delay=0."""
        mock_fn = MagicMock(side_effect=Exception("err"))
        decorated = retry(max_attempts=3, delay=0)(mock_fn)

        with pytest.raises(Exception):
            decorated()

        mock_sleep.assert_not_called()

    @patch("utils.retry.time.sleep")
    def test_no_sleep_after_last_attempt(self, mock_sleep: MagicMock):
        """No sleep should occur after the final (exhausting) attempt."""
        mock_fn = MagicMock(side_effect=[Exception("e"), "success"])
        decorated = retry(max_attempts=2, delay=1.0)(mock_fn)

        result = decorated()

        assert result == "success"
        # Sleep only between attempt 1 → 2 (one call), not after success
        mock_sleep.assert_called_once_with(1.0)


# ---------------------------------------------------------------------------
# Selective exception tests
# ---------------------------------------------------------------------------

class TestRetrySelectiveExceptions:
    def test_only_specified_exceptions_are_retried(self):
        """Exceptions not in the *exceptions* tuple must not be retried."""
        mock_fn = MagicMock(side_effect=CustomError("custom"))
        decorated = retry(max_attempts=5, delay=0, exceptions=(ValueError,))(mock_fn)

        with pytest.raises(CustomError):
            decorated()

        # Should bail out on the first attempt — no retry
        mock_fn.assert_called_once()

    def test_specified_subclass_exception_is_retried(self):
        """Subclasses of a listed exception type must also trigger retries."""
        class SubError(ValueError):
            pass

        mock_fn = MagicMock(side_effect=[SubError("sub"), "ok"])
        decorated = retry(max_attempts=2, delay=0, exceptions=(ValueError,))(mock_fn)

        result = decorated()

        assert result == "ok"
        assert mock_fn.call_count == 2


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestRetryValidation:
    def test_raises_value_error_for_zero_max_attempts(self):
        """max_attempts=0 must raise ValueError at decoration time."""
        with pytest.raises(ValueError, match="max_attempts"):
            retry(max_attempts=0)

    def test_raises_value_error_for_negative_max_attempts(self):
        """Negative max_attempts must raise ValueError at decoration time."""
        with pytest.raises(ValueError, match="max_attempts"):
            retry(max_attempts=-3)

    def test_raises_value_error_for_negative_delay(self):
        """Negative delay must raise ValueError at decoration time."""
        with pytest.raises(ValueError, match="delay"):
            retry(max_attempts=3, delay=-1.0)


# ---------------------------------------------------------------------------
# Decorator metadata preservation
# ---------------------------------------------------------------------------

class TestRetryMetadata:
    def test_functools_wraps_preserves_name(self):
        """The wrapped function's __name__ must match the original."""
        @retry(max_attempts=2, delay=0)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_functools_wraps_preserves_docstring(self):
        """The wrapped function's __doc__ must match the original."""
        @retry(max_attempts=2, delay=0)
        def documented():
            """My docstring."""

        assert documented.__doc__ == "My docstring."
