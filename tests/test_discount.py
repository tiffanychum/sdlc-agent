"""
test_discount.py
----------------
Pytest test suite for discount_utils.calculate_discount().

SDLC-11: Implement discount_utils.py with pytest coverage
"""

import pytest
from tests.discount_utils import calculate_discount


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_normal_discount():
    """25% off 200 should return 150.0."""
    assert calculate_discount(200, 25) == 150.0


def test_zero_discount():
    """0% discount leaves the price unchanged."""
    assert calculate_discount(100, 0) == 100.0


def test_full_discount():
    """100% discount reduces price to 0.0."""
    assert calculate_discount(200, 100) == 0.0


def test_fractional_discount():
    """10% off 50 should return 45.0."""
    assert calculate_discount(50, 10) == 45.0


def test_float_price():
    """Works correctly with a float price."""
    result = calculate_discount(99.99, 50)
    assert abs(result - 49.995) < 1e-9


# ---------------------------------------------------------------------------
# Boundary / edge-case tests
# ---------------------------------------------------------------------------

def test_boundary_zero_percent():
    """Boundary: 0% is a valid discount_pct."""
    assert calculate_discount(300, 0) == 300.0


def test_boundary_hundred_percent():
    """Boundary: 100% is a valid discount_pct."""
    assert calculate_discount(300, 100) == 0.0


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_invalid_negative_discount():
    """Negative discount_pct must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_discount(200, -1)


def test_invalid_over_100_discount():
    """discount_pct > 100 must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_discount(200, 101)


def test_invalid_large_negative():
    """A large negative discount_pct must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_discount(100, -50)


def test_invalid_far_over_100():
    """A discount_pct far above 100 must raise ValueError."""
    with pytest.raises(ValueError):
        calculate_discount(100, 200)


def test_error_message_contains_value():
    """ValueError message should mention the offending value."""
    with pytest.raises(ValueError, match="-5"):
        calculate_discount(100, -5)
