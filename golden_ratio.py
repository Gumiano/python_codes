def improve(update, close, guess=1):
    """Iterative improvement algorithm.

    close - A comparision to check whether the current guess is "close enough" to be considered correct.
    update - A function which is repeatedly used to improve guess.
    guess - A guess of a solution to an equation.
    """
    while not close(guess):
        guess = update(guess)
    return guess

def golden_update(guess):
    return 1/guess + 1

def square_close_to_successor(guess):
    return approx_eq(guess * guess, guess + 1)

def approx_eq(x, y, tolerance=1e-15):
    return abs(x - y) < tolerance
