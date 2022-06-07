def compose1(f, g):
    def h(x):
        return f(g(x))
    return h


