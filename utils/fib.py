"""Fibonacci number utilities."""


def fibonacci(n):
    """Return the nth Fibonacci number.
    
    Args:
        n: The index of the Fibonacci number to return (0-indexed).
        
    Returns:
        The nth Fibonacci number where fib(0)=0, fib(1)=1, fib(n)=fib(n-1)+fib(n-2).
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
