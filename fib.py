def fib(n):
    """Compute the nth Fibonacci number, for n >= 2."""
    pred, curr = 0, 1 # Fibonacci numbers 1 and 2
    k = 2
    while k < n:
        pred, curr = curr, pred + curr
        k = k + 1
    return curr

def sum_naturals(n):
    """Return the sum of the first n natural numbers.

    >>> sum_naturals(10)
    55
    >>> sum_naturals(100)
    5050
    """
    total, k = 0, 1
    while k <= n:
        total, k = total + k, k + 1
    return total

