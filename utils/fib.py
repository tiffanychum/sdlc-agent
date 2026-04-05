def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed).

    Args:
        n: A non-negative integer position in the Fibonacci sequence.

    Returns:
        The nth Fibonacci number, where fibonacci(0) = 0 and fibonacci(1) = 1.

    Raises:
        ValueError: If n is negative.
        TypeError: If n is not an integer.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
