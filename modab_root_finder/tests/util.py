from functools import wraps


def debug_func(func):
    @wraps(func)
    def inner(x):
        y = func(x)
        print(f"f({x}) = {y}")
        return y
    return inner
