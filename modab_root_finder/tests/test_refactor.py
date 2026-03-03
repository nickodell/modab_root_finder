from modab_root_finder import root_scalar
from modab_root_finder.modab_refactor import (
    sign,
    SolverState,
    SolverArgs,
    modab_single,
)
import numpy as np
from numpy.testing import assert_allclose
from functools import wraps


# Source - https://stackoverflow.com/a/14620633
# Posted by Kimvais, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-02, License - CC BY-SA 4.0

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_bisect_state(args, x1, x2):
    return SolverState(dict(
        bisect=True,
        x1=x1,
        x2=x2,
        y1=args.func(x1),
        y2=args.func(x2),
        func_calls=2,
        x3_prev=x1,
        terminate=False,
    ))


def make_bisect_args(func):
    return SolverArgs(dict(
        xtol=1e-14,
        rtol=0,
        ftol=0,
        func=func,
    ))


def debug_func(func):
    @wraps(func)
    def inner(x):
        y = func(x)
        print(f"f({x}) = {y}")
        return y
    return inner


def assert_state_preconditions(state):
    state = AttrDict(state.asdict())
    assert state.x1 < state.x2
    assert sign(state.y1) == -1 * sign(state.y2)


def assert_state_postconditions(state):
    state = AttrDict(state.asdict())
    assert state.x1 < state.x2
    assert sign(state.y1) == -1 * sign(state.y2)


def test_bisect1():
    @debug_func
    def f(x):
        print("x")
        return x + 1
    args = make_bisect_args(f)
    state = make_bisect_state(args, -5, 5)
    assert_state_preconditions(state)
    modab_single(state, args)
    assert_state_postconditions(state)
