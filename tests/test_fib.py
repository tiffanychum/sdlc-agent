import pytest
from utils.fib import fibonacci


def test_fib_zero():
    """fibonacci(0) should return 0 — the first Fibonacci number."""
    assert fibonacci(0) == 0


def test_fib_one():
    """fibonacci(1) should return 1 — the second Fibonacci number."""
    assert fibonacci(1) == 1


def test_fib_ten():
    """fibonacci(10) should return 55 — the eleventh Fibonacci number."""
    assert fibonacci(10) == 55
