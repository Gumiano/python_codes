# square of a number is the number nultiplied by itself
def square(n):
    return n * n

def summation(n, term):
    return summation2(n, term, lambda x: x+1)

def summation2(n, term, next):
    sum, k = 0, 1
    while k <= n:
        sum = sum + term(k)
        k = next(k)
    return sum

def sum_of_n_even(n):
    """Returns the sum of the first N even numbers."""
    return summation(n, lambda x: 2 * x)

def sum_of_n_odd(n):
    """Returns the sum of the first N odd numbers."""
    return summation(n, lambda x: 2 * x + 1)

def sum_of_n_starting_from_m(m, n):
    """Returns the sum of the first n numbers starting from m."""
    return summation(n, lambda x: x + m)

def deriv(f, x, delta_x):
    return (f(x + delta_x) - f(x)) / delta_x

def filterd_count(n, pred):
    """
    Prints all the numbers 1 to N for which PRED returns True.
    """
    k = 1
    while k <= n:
        if pred(k):
            print(k)
        k += 1

def compose(f, g):
    return lambda x: f(g(x))

increment = lambda num: num+1
square = lambda num: num*num
identity = lambda num: num
twice = lambda f: compose(f, f)