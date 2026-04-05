"""Fibonacci number calculator."""


def fibonacci(n: int) -> int:
    """
    Return the nth Fibonacci number.
    
    The Fibonacci sequence is defined as:
    - fib(0) = 0
    - fib(1) = 1
    - fib(n) = fib(n-1) + fib(n-2) for n > 1
    
    Args:
        n: The position in the Fibonacci sequence (non-negative integer)
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Iterative approach for efficiency
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr
