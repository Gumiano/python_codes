def improve(update, isclose, guess=1):
    while not isclose(guess):
        guess = update(guess)
    return guess

"""
newton's method
used to find the roots of a function where the function evaluates to zero
"""