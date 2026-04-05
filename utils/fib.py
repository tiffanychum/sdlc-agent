def fibonacci(n):
    """Return the nth Fibonacci number (0-indexed).

    Args:
        n (int): A non-negative integer index into the Fibonacci sequence.

    Returns:
        int: The nth Fibonacci number, where fibonacci(0) == 0
             and fibonacci(1) == 1.

    Raises:
        ValueError: If n is a negative integer.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError(f"n must be a non-negative integer, got {n}")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
