"""
tests/test_retry.py
-------------------
Pytest test suite for the `retry` decorator defined in utils/retry.py.

Coverage
--------
Happy-path tests (SDLC-36)
  - Function succeeds on the first call.
  - Function fails twice then succeeds on the third attempt.
  - Function always fails — last exception re-raised after exactly N attempts.
  - Retry count accuracy validated with a call counter.

Edge-case & delay tests (SDLC-37)
  - delay > 0 causes time.sleep() to be called (N-1) times with the right value.
  - delay=0 (default) — time.sleep is NOT called.
  - n=1 — function called exactly once; exception propagates immediately.
  - Invalid n (0, negative, non-integer) — ValueError from the decorator.
  - functools.wraps: __name__ and __doc__ are preserved on the wrapper.
"""

import time
import pytest
from unittest.mock import patch, call

from utils.retry import retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class AlwaysFailsError(Exception):
    """Sentinel exception used in tests."""


# ---------------------------------------------------------------------------
# Happy-path tests (SDLC-36)
# ---------------------------------------------------------------------------

class TestRetryHappyPath:
    """Tests covering successful execution and retry-until-success scenarios."""

    def test_succeeds_on_first_call_returns_value(self):
        """Function that never raises should return its value immediately."""
        @retry(n=3)
        def always_ok():
            return 42

        assert always_ok() == 42

    def test_call_count_is_one_when_first_attempt_succeeds(self):
        """Wrapper must NOT make extra calls if the first attempt succeeds."""
        counter = {"calls": 0}

        @retry(n=5)
        def succeeds_first_time():
            counter["calls"] += 1
            return "success"

        result = succeeds_first_time()

        assert result == "success"
        assert counter["calls"] == 1

    def test_succeeds_on_third_attempt_returns_value(self):
        """Function that fails twice then succeeds should return its value."""
        counter = {"calls": 0}

        @retry(n=5, delay=0)
        def fails_twice_then_ok():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise AlwaysFailsError("not yet")
            return "recovered"

        result = fails_twice_then_ok()

        assert result == "recovered"
        assert counter["calls"] == 3

    def test_exact_retry_count_on_eventual_success(self):
        """Verify the call counter matches expected attempts on eventual success."""
        calls = []

        @retry(n=4, delay=0)
        def fail_three_times():
            calls.append(1)
            if len(calls) < 4:
                raise ValueError("retry me")
            return "done"

        result = fail_three_times()

        assert result == "done"
        assert len(calls) == 4

    def test_always_fails_raises_last_exception(self):
        """After N failures the last exception must be re-raised."""
        @retry(n=3, delay=0)
        def always_fails():
            raise AlwaysFailsError("boom")

        with pytest.raises(AlwaysFailsError, match="boom"):
            always_fails()

    def test_always_fails_calls_function_exactly_n_times(self):
        """The wrapped function must be invoked exactly N times, no more."""
        counter = {"calls": 0}

        @retry(n=4, delay=0)
        def always_fails():
            counter["calls"] += 1
            raise AlwaysFailsError("fail")

        with pytest.raises(AlwaysFailsError):
            always_fails()

        assert counter["calls"] == 4

    def test_passes_args_and_kwargs_through(self):
        """retry must forward positional and keyword arguments unchanged."""
        @retry(n=3)
        def add(a, b, *, multiplier=1):
            return (a + b) * multiplier

        assert add(2, 3, multiplier=4) == 20


# ---------------------------------------------------------------------------
# Edge-case & delay tests (SDLC-37)
# ---------------------------------------------------------------------------

class TestRetryEdgeCases:
    """Tests covering boundary values, delay behaviour, and input validation."""

    def test_delay_calls_sleep_between_attempts(self):
        """time.sleep must be called (N-1) times when delay > 0 and all attempts fail."""
        @retry(n=3, delay=0.1)
        def always_fails():
            raise AlwaysFailsError("x")

        with patch("utils.retry.time.sleep") as mock_sleep:
            with pytest.raises(AlwaysFailsError):
                always_fails()

        # sleep is called between attempts: after attempt 1 and after attempt 2
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(0.1), call(0.1)])

    def test_delay_not_called_after_last_failing_attempt(self):
        """time.sleep must NOT be called after the final (Nth) failed attempt."""
        sleep_calls = []

        @retry(n=3, delay=0.5)
        def always_fails():
            raise AlwaysFailsError("x")

        with patch("utils.retry.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with pytest.raises(AlwaysFailsError):
                always_fails()

        # With n=3, sleep should only be called 2 times (after attempt 1 and 2)
        assert len(sleep_calls) == 2

    def test_zero_delay_does_not_call_sleep(self):
        """Default delay=0 must never call time.sleep."""
        @retry(n=3, delay=0)
        def always_fails():
            raise AlwaysFailsError("x")

        with patch("utils.retry.time.sleep") as mock_sleep:
            with pytest.raises(AlwaysFailsError):
                always_fails()

        mock_sleep.assert_not_called()

    def test_n_equals_one_calls_function_once_and_raises(self):
        """n=1 means a single attempt — exception must propagate immediately."""
        counter = {"calls": 0}

        @retry(n=1, delay=0)
        def always_fails():
            counter["calls"] += 1
            raise AlwaysFailsError("single shot")

        with pytest.raises(AlwaysFailsError, match="single shot"):
            always_fails()

        assert counter["calls"] == 1

    def test_n_equals_one_success(self):
        """n=1 with a function that succeeds should work correctly."""
        @retry(n=1)
        def ok():
            return "fine"

        assert ok() == "fine"

    # --- Input validation ---

    def test_n_zero_raises_value_error(self):
        """n=0 is invalid; ValueError must be raised at decoration time."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            @retry(n=0)
            def func():
                pass

    def test_n_negative_raises_value_error(self):
        """Negative n is invalid; ValueError must be raised at decoration time."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            @retry(n=-5)
            def func():
                pass

    def test_n_non_integer_raises_value_error(self):
        """Non-integer n (e.g. float) is invalid; ValueError must be raised."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            @retry(n=3.0)  # type: ignore[arg-type]
            def func():
                pass

    def test_n_bool_raises_value_error(self):
        """bool is a subclass of int but must be rejected for clarity."""
        with pytest.raises(ValueError, match="n must be a positive integer"):
            @retry(n=True)  # type: ignore[arg-type]
            def func():
                pass

    def test_negative_delay_raises_value_error(self):
        """Negative delay is invalid; ValueError must be raised at decoration time."""
        with pytest.raises(ValueError, match="delay must be >= 0"):
            @retry(n=3, delay=-1.0)
            def func():
                pass

    # --- functools.wraps preservation ---

    def test_wraps_preserves_function_name(self):
        """The wrapper must expose the original function's __name__."""
        @retry(n=3)
        def my_special_function():
            """My docstring."""

        assert my_special_function.__name__ == "my_special_function"

    def test_wraps_preserves_docstring(self):
        """The wrapper must expose the original function's __doc__."""
        @retry(n=3)
        def documented():
            """This is the docstring."""

        assert "This is the docstring." in documented.__doc__

    # --- Misc behavioural edge-cases ---

    def test_non_exception_return_value_none(self):
        """Function returning None should be treated as success, not re-tried."""
        counter = {"calls": 0}

        @retry(n=3)
        def returns_none():
            counter["calls"] += 1
            return None

        result = returns_none()

        assert result is None
        assert counter["calls"] == 1

    def test_retry_can_be_applied_to_lambda_equivalent(self):
        """Decorator applied programmatically (not via @-syntax) must work."""
        def flaky(limit, counter):
            counter["n"] += 1
            if counter["n"] < limit:
                raise RuntimeError("not ready")
            return counter["n"]

        counter = {"n": 0}
        retried = retry(n=5, delay=0)(flaky)

        result = retried(3, counter)

        assert result == 3

    def test_different_exception_types_still_trigger_retry(self):
        """retry must catch any Exception subclass, not just a specific type."""
        errors = [KeyError("k"), IndexError("i"), ValueError("v")]
        call_log = []

        @retry(n=4, delay=0)
        def raises_different_errors():
            call_log.append(1)
            if errors:
                raise errors.pop(0)
            return "finally"

        result = raises_different_errors()

        assert result == "finally"
        assert len(call_log) == 4
