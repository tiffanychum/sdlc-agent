"""
tests/test_retry.py
-------------------
Pytest tests for the `retry` decorator defined in utils/retry.py.

Test matrix
~~~~~~~~~~~
* Succeeds on first attempt            -> returns value, called once
* Succeeds after N-1 failures          -> returns value, called N times
* Fails all attempts                   -> raises the last exception
* Non-matching exception               -> propagates immediately, no retry
* Delay is honoured between attempts   -> time.sleep called with correct args
* Preserves wrapped function metadata  -> __name__, __doc__ intact
* Bad constructor arguments raise early:
  - times < 1
  - delay < 0
  - exceptions not a tuple
"""

import time
import pytest
from unittest.mock import MagicMock, patch, call

from utils.retry import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TransientError(Exception):
    """Raised intentionally to simulate a transient failure."""


class FatalError(Exception):
    """Raised intentionally to simulate a non-retryable failure."""


def make_flaky(total_calls: int, fail_times: int, exc=TransientError):
    """Return a MagicMock that raises *exc* for the first *fail_times* calls,
    then returns ``"ok"``."""
    side_effects = [exc("boom")] * fail_times + ["ok"] * (total_calls - fail_times)
    mock = MagicMock(side_effect=side_effects)
    return mock


# ---------------------------------------------------------------------------
# Basic success / failure
# ---------------------------------------------------------------------------

class TestRetryBasicBehaviour:

    def test_succeeds_on_first_attempt(self):
        """Function that never raises should be called exactly once."""
        mock_fn = MagicMock(return_value=42)

        @retry(times=3)
        def always_ok():
            return mock_fn()

        result = always_ok()
        assert result == 42
        mock_fn.assert_called_once()

    def test_succeeds_after_partial_failures(self):
        """Function should succeed once transient failures are exhausted."""
        fail_times = 2
        mock_fn = make_flaky(total_calls=3, fail_times=fail_times)

        @retry(times=3, exceptions=(TransientError,))
        def flaky():
            return mock_fn()

        result = flaky()
        assert result == "ok"
        assert mock_fn.call_count == 3

    def test_raises_after_all_attempts_exhausted(self):
        """Should raise the last exception when every attempt fails."""
        mock_fn = MagicMock(side_effect=TransientError("always fails"))

        @retry(times=3, exceptions=(TransientError,))
        def always_fails():
            return mock_fn()

        with pytest.raises(TransientError, match="always fails"):
            always_fails()

        assert mock_fn.call_count == 3

    def test_times_equals_one_no_retry(self):
        """With times=1 the function should be tried exactly once and the
        exception re-raised immediately."""
        mock_fn = MagicMock(side_effect=TransientError("once"))

        @retry(times=1, exceptions=(TransientError,))
        def one_shot():
            return mock_fn()

        with pytest.raises(TransientError):
            one_shot()

        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Non-matching exceptions
# ---------------------------------------------------------------------------

class TestRetryExceptionFiltering:

    def test_non_matching_exception_propagates_immediately(self):
        """An exception not in *exceptions* must propagate without retrying."""
        call_count = 0

        @retry(times=5, exceptions=(TransientError,))
        def raises_fatal():
            nonlocal call_count
            call_count += 1
            raise FatalError("fatal")

        with pytest.raises(FatalError, match="fatal"):
            raises_fatal()

        # Must NOT have retried
        assert call_count == 1

    def test_only_specified_exceptions_trigger_retry(self):
        """Multiple exception types: matching ones retry, others propagate."""
        mock_fn = MagicMock(side_effect=[TransientError("t"), FatalError("f")])

        @retry(times=5, exceptions=(TransientError,))
        def mixed():
            return mock_fn()

        with pytest.raises(FatalError):
            mixed()

        # Called twice: first raises TransientError (retried), second FatalError (not)
        assert mock_fn.call_count == 2


# ---------------------------------------------------------------------------
# Delay between attempts
# ---------------------------------------------------------------------------

class TestRetryDelay:

    def test_delay_is_respected_between_attempts(self):
        """time.sleep should be called (times-1) times with the correct delay."""
        mock_fn = make_flaky(total_calls=3, fail_times=2)

        @retry(times=3, delay=0.5, exceptions=(TransientError,))
        def flaky():
            return mock_fn()

        with patch("utils.retry.time.sleep") as mock_sleep:
            result = flaky()

        assert result == "ok"
        # sleep should be called after attempt 1 and attempt 2, but NOT after 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(0.5), call(0.5)])

    def test_no_delay_means_no_sleep(self):
        """With delay=0 (default) time.sleep must never be called."""
        mock_fn = make_flaky(total_calls=2, fail_times=1)

        @retry(times=2, exceptions=(TransientError,))
        def flaky():
            return mock_fn()

        with patch("utils.retry.time.sleep") as mock_sleep:
            flaky()

        mock_sleep.assert_not_called()

    def test_sleep_not_called_after_final_failed_attempt(self):
        """sleep must NOT be called after the last (exhausted) attempt."""
        mock_fn = MagicMock(side_effect=TransientError("fail"))

        @retry(times=3, delay=1.0, exceptions=(TransientError,))
        def always_fails():
            return mock_fn()

        with patch("utils.retry.time.sleep") as mock_sleep:
            with pytest.raises(TransientError):
                always_fails()

        # 3 attempts => sleep between attempt 1→2 and 2→3, NOT after 3
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# Decorator metadata preservation
# ---------------------------------------------------------------------------

class TestRetryMetadata:

    def test_wrapped_function_name_preserved(self):
        """functools.wraps should preserve __name__."""

        @retry(times=2)
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"

    def test_wrapped_function_docstring_preserved(self):
        """functools.wraps should preserve __doc__."""

        @retry(times=2)
        def documented():
            """Important docs."""

        assert documented.__doc__ == "Important docs."


# ---------------------------------------------------------------------------
# Constructor argument validation
# ---------------------------------------------------------------------------

class TestRetryArgumentValidation:

    def test_times_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="`times`"):
            retry(times=0)

    def test_times_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="`times`"):
            retry(times=-1)

    def test_times_non_integer_raises_value_error(self):
        with pytest.raises(ValueError, match="`times`"):
            retry(times=2.5)  # type: ignore[arg-type]

    def test_delay_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="`delay`"):
            retry(times=3, delay=-0.1)

    def test_exceptions_not_tuple_raises_type_error(self):
        with pytest.raises(TypeError, match="`exceptions`"):
            retry(times=3, exceptions=ValueError)  # type: ignore[arg-type]

    def test_exceptions_empty_tuple_raises_type_error(self):
        with pytest.raises(TypeError, match="`exceptions`"):
            retry(times=3, exceptions=())

    def test_valid_arguments_do_not_raise(self):
        """Sanity check: correct arguments produce a decorator without error."""
        dec = retry(times=5, delay=0.1, exceptions=(ValueError, IOError))
        assert callable(dec)


# ---------------------------------------------------------------------------
# Edge / integration cases
# ---------------------------------------------------------------------------

class TestRetryEdgeCases:

    def test_return_value_is_passed_through(self):
        """Decorator must not alter the return value."""

        @retry(times=3)
        def give_dict():
            return {"key": [1, 2, 3]}

        assert give_dict() == {"key": [1, 2, 3]}

    def test_args_and_kwargs_forwarded(self):
        """Positional and keyword arguments must be forwarded correctly."""

        @retry(times=2)
        def add(a, b, *, multiplier=1):
            return (a + b) * multiplier

        assert add(3, 4, multiplier=2) == 14

    def test_retry_multiple_exception_types(self):
        """Retry should trigger for any exception listed in *exceptions*."""
        side_effects = [TransientError("t"), FatalError("f"), "ok"]
        mock_fn = MagicMock(side_effect=side_effects)

        @retry(times=3, exceptions=(TransientError, FatalError))
        def multi_exc():
            return mock_fn()

        result = multi_exc()
        assert result == "ok"
        assert mock_fn.call_count == 3
