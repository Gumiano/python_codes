def compose(g, f):
    return lambda x: g(f(x))

def add_x(y):
    return lambda x: y + x

"""范畴：元素的集合+变形关系"""
class Category:
    def __init__(self, val) -> None:
        self.val = val

    def addOne(x):
        return x + 1

"""函子：也是一个范畴，但函数是当前范畴到另一个范畴的映射"""
class Functor:
    def __init__(self, val) -> None:
        self.val = val

    def map(self, f):
        return Functor(f(self.val))

if __name__ == '__main__':
    print( (Functor('bombs')).map(str.upper).val )