"""Tests for the fibonacci function."""

import pytest
from utils.fib import fibonacci


def test_fib_zero():
    """Test that fib(0) returns 0."""
    assert fibonacci(0) == 0


def test_fib_one():
    """Test that fib(1) returns 1."""
    assert fibonacci(1) == 1


def test_fib_ten():
    """Test that fib(10) returns 55."""
    assert fibonacci(10) == 55
