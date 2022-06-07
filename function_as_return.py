def funny_function(a, b, c):
    # local helper function
    def a_funny(b):
        (a * b) + (b / a) + (a ** b)
    return a_funny(b) / a_funny(c)

def make_adder(m):
    def add_m_to_twice(n):
        return (2*n) + m
    return add_m_to_twice

def make_summationer(term):
    def summation(n):
        total, k = 0, 1
        while k <= n:
            total += term(k)
        return total

    return summation

