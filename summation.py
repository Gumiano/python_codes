def summation(n, term):
    total, k = 0, 1
    while k <= n:
        total, k = total + term(k), k + 1
    return total

def cube(x):
    return x ** 3

def identity(x):
    return x

def pi_term(x):
    return 8 / ((4*x-3) * (4*x-1))

def sum_naturals(n):
    return summation(n, identity)

def sum_cubes(n):
    return summation(n, cube)

def pi_sum(n):
    return summation(n, pi_term)
